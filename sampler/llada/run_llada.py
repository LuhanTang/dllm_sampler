#!/usr/bin/env python3
# sampler/llada/run_llada.py
# ------------------------------------------------------------
# Oracle LLaDA runner (ReMDM-aligned outputs + metrics)
#
# Replacement version requirements implemented:
#   - Preserve original metrics + plots + csv + jsonl/json outputs
#   - When --remasking low_confidence: automatically dump decoded text + top ngrams (token ngrams)
#   - When --remasking random: (UPDATED) ALSO dump decoded text + top ngrams (token ngrams)
#   - Default --temperature=1.0
#   - Prompt default OFF (prompt_mode=none)
#   - step_seed is ALWAYS args.seed (no seed_mode knob)
#   - AR baseline seed = args.seed + 777
#   - tokenizer_json default: tokenizers/owt_bytebpe_v4096/tokenizer.json
#
# NEW (Feb 2026):
#   - Chunked eval with --N_chunk (default 128) to reduce VRAM.
#   - Save full sampled ids (N x T) for BOTH remasking modes under run_dir/seqs/ (default ON).
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt

from sampler.gt_io import load_gt
from sampler.utils_io import parse_steps, _fmt_float_tag, _sanitize

from sampler.metrics_full import (
    SparseTeleportPrior as MetricsSparseTeleportPrior,
    nll_transition_sparse_teleport,
    full_kl_rate_sparse_teleport,
    unigram_l1,
    unique_ngram_ratio,
    dup_rate,
)

# optional debug print helper (repo-dependent)
try:
    from sampler.metrics_full import top_unigrams_bigrams_print  # type: ignore
except Exception:
    top_unigrams_bigrams_print = None  # noqa

from sampler.ar_baseline_sparse import sample_ar_sparse_teleport
from sampler.llada.generate_llada import generate

from sampler.oracle_hmm_posterior import (
    SparseTeleportPrior as OracleSparseTeleportPrior,
    OracleHMMPosterior_LogRank1Teleport,
)

LOGIT_CLAMP = 1e-30
NINF = -1e30


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _plot_curve(
    xs: List[int],
    ys: List[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
    ylog: bool = False,
    ar_value: Optional[float] = None,
) -> None:
    plt.figure(figsize=(9.2, 4.9))
    plt.plot(
        xs,
        ys,
        marker="o",
        linewidth=2.3,
        markersize=6,
        label="LLaDA",
        color="tab:orange",
    )
    if ar_value is not None:
        plt.axhline(
            ar_value,
            linestyle="--",
            linewidth=2.0,
            label="AR baseline",
            color="tab:blue",
        )

    plt.xscale("log")
    if ylog:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


@torch.no_grad()
def make_oracle_logits_fn(
    oracle: OracleHMMPosterior_LogRank1Teleport,
    *,
    V: int,
    temp_beta: float,
) -> Any:
    """
    logits_fn(x, attention_mask=None) -> logits [B,T,V+1]

    We output logits for TRUE tokens [:V] using oracle posterior,
    and forbid MASK class at index V by setting it to -inf-ish.

    NOTE: generate() itself handles "only update masked positions" via where().
    """
    beta = float(temp_beta)

    def logits_fn(x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        device = x.device
        B, T = x.shape

        p = oracle(x).to(torch.float32)  # [B,T,V]
        logp = torch.log(p.clamp_min(LOGIT_CLAMP))
        if beta != 1.0:
            logp = logp * beta

        logits = torch.empty((B, T, V + 1), device=device, dtype=torch.float32)
        logits[..., :V] = logp
        logits[..., V] = NINF
        return logits

    return logits_fn


@torch.no_grad()
def _fill_remaining_masks_from_pi(
    x: torch.Tensor,  # [N,T]
    *,
    pi: torch.Tensor,  # [V]
    V: int,
    seed: int,
) -> torch.Tensor:
    mask_id = int(V)
    if not (x == mask_id).any():
        return x
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 99991)

    m = (x == mask_id)
    num = int(m.sum().item())
    fill = torch.multinomial(pi.detach().cpu(), num_samples=num, replacement=True, generator=g).to(x.device)
    x2 = x.clone()
    x2[m] = fill
    return x2


def _maybe_load_hf_tokenizer_json(tokenizer_json: str | None):
    """
    Load a HuggingFace `tokenizers` JSON file (ByteBPE etc.) for:
      - decoding dumps
      - optional prompt encoding (prompt_str/text)
      - optional eos/eot inference (best-effort via special tokens)
    """
    if not tokenizer_json:
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "tokenizers is required for tokenizer_json decoding/prompt. "
            "Install `tokenizers` or disable dumping/prompt options."
        ) from e

    if not os.path.exists(tokenizer_json):
        raise FileNotFoundError(f"tokenizer_json not found: {tokenizer_json}")
    tok = Tokenizer.from_file(tokenizer_json)
    return tok


def _infer_eos_eot_ids_from_tokenizers_json(
    *,
    tokenizer,  # tokenizers.Tokenizer | None
    eos_id_cli: int | None,
    eot_id_cli: int | None,
) -> Tuple[int | None, int | None]:
    """
    Best-effort inference for tokenizers-json:
      - if CLI provided: trust it
      - else: try common special strings by encoding
    """
    eos_id = eos_id_cli
    eot_id = eot_id_cli
    if tokenizer is None:
        return eos_id, eot_id

    # Helper: encode a candidate string; if it maps to exactly one token, use that id.
    def _try_single_id(s: str) -> int | None:
        try:
            enc = tokenizer.encode(s)
            ids = list(enc.ids)
            if len(ids) == 1:
                return int(ids[0])
        except Exception:
            return None
        return None

    if eos_id is None:
        for s in ["</s>", "<|endoftext|>", "<eos>", "<EOS>"]:
            cand = _try_single_id(s)
            if cand is not None:
                eos_id = cand
                break

    if eot_id is None:
        for s in ["<|eot_id|>", "<|EOT|>", "<EOT>", "<eot>", "<EOT_ID>"]:
            cand = _try_single_id(s)
            if cand is not None:
                eot_id = cand
                break

    return eos_id, eot_id


def _build_prompt(
    *,
    prompt_mode: str,
    prompt_token_id: int | None,
    prompt_char: str | None,
    prompt_str: str | None,
    prompt_text: str | None,
    tokenizer,  # tokenizers.Tokenizer | None
    gt,
    device: torch.device,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor | None, Dict[str, Any]]:
    """
    Returns:
      prompt_ids: LongTensor [N, Lp]
      attention_mask: LongTensor [N, Lp] (no padding case -> all ones) or None
      prompt_meta: dict to store in header

    NOTE: default is OFF (prompt_mode=none).
    """
    if prompt_mode == "none":
        prompt = torch.empty((N, 0), dtype=torch.long, device=device)
        return prompt, None, {"mode": "none", "prompt_len": 0}

    if prompt_mode == "token":
        ids: List[int] = []

        if prompt_token_id is not None:
            ids = [int(prompt_token_id)]
            meta = {"mode": "token", "source": "prompt_token_id", "ids": ids}

        elif prompt_char is not None:
            if not (hasattr(gt, "stoi") and isinstance(gt.stoi, dict)):
                raise ValueError("--prompt_char requires GT with `stoi` mapping (e.g., text8-char GT).")
            if len(prompt_char) != 1:
                raise ValueError("--prompt_char must be exactly 1 character.")
            if prompt_char not in gt.stoi:
                raise ValueError(f"--prompt_char={prompt_char!r} not in gt.stoi.")
            ids = [int(gt.stoi[prompt_char])]
            meta = {"mode": "token", "source": "prompt_char", "char": prompt_char, "ids": ids}

        elif prompt_str is not None:
            if tokenizer is None:
                raise ValueError("--prompt_str requires --tokenizer_json.")
            enc = tokenizer.encode(prompt_str)
            tok_ids = list(map(int, enc.ids))
            if len(tok_ids) != 1:
                raise ValueError(
                    f"--prompt_mode token requires prompt_str to encode into exactly 1 token, got {len(tok_ids)}."
                )
            ids = tok_ids
            meta = {"mode": "token", "source": "prompt_str", "prompt_str": prompt_str, "ids": ids}

        else:
            raise ValueError("--prompt_mode token requires one of: --prompt_token_id, --prompt_char, --prompt_str")

        prompt_1 = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, Lp]
        prompt = prompt_1.repeat(N, 1)  # [N, Lp]
        attn = torch.ones((N, prompt.shape[1]), dtype=torch.long, device=device)
        meta["prompt_len"] = int(prompt.shape[1])
        return prompt, attn, meta

    if prompt_mode == "text":
        if prompt_text is None:
            raise ValueError("--prompt_mode text requires --prompt_text.")
        if tokenizer is None:
            raise ValueError("--prompt_text requires --tokenizer_json.")
        enc = tokenizer.encode(prompt_text)
        ids = list(map(int, enc.ids))
        ids_1 = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, Lp]
        prompt = ids_1.repeat(N, 1)
        attn = torch.ones((N, prompt.shape[1]), dtype=torch.long, device=device)
        meta = {"mode": "text", "prompt_text": prompt_text, "prompt_len": int(prompt.shape[1])}
        return prompt, attn, meta

    raise ValueError(f"Unknown --prompt_mode {prompt_mode!r}")


def _decode_batch_text(
    x: torch.Tensor,  # [N,T]
    *,
    tokenizer,  # tokenizers.Tokenizer | None
    V: int,
) -> List[str]:
    """
    Decode each sequence into text for dumping.
    If tokenizer is None, fallback: join token ids as space-separated ints.
    """
    x_cpu = x.detach().to("cpu")
    mask_id = int(V)

    outs: List[str] = []
    if tokenizer is None:
        for i in range(x_cpu.shape[0]):
            ids = [int(t) for t in x_cpu[i].tolist() if int(t) != mask_id]
            outs.append(" ".join(map(str, ids)))
        return outs

    for i in range(x_cpu.shape[0]):
        ids = [int(t) for t in x_cpu[i].tolist() if int(t) != mask_id]
        try:
            s = tokenizer.decode(ids)
        except Exception:
            s = " ".join(map(str, ids))
        outs.append(s)
    return outs


def _top_ngrams_token_ids(
    x: torch.Tensor,  # [N,T]
    *,
    n: int,
    V: int,
    k: int,
) -> List[Tuple[Tuple[int, ...], int, float]]:
    """
    Count token-id ngrams across the whole batch (ignoring MASK=V).
    Returns top-k (ngram_ids, count, fraction_of_all_ngrams).
    """
    assert n >= 1
    x_cpu = x.detach().to("cpu")
    mask_id = int(V)

    counts: Dict[Tuple[int, ...], int] = {}
    total = 0

    N, T = x_cpu.shape
    for i in range(N):
        seq = [int(t) for t in x_cpu[i].tolist() if int(t) != mask_id]
        if len(seq) < n:
            continue
        for j in range(len(seq) - n + 1):
            ng = tuple(seq[j : j + n])
            counts[ng] = counts.get(ng, 0) + 1
            total += 1

    if total == 0 or not counts:
        return []

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: int(k)]
    out: List[Tuple[Tuple[int, ...], int, float]] = []
    for ng, c in items:
        out.append((ng, int(c), float(c) / float(total)))
    return out


def _format_top_ngrams_for_dump(
    ngrams: List[Tuple[Tuple[int, ...], int, float]],
    *,
    tokenizer,  # tokenizers.Tokenizer | None
) -> List[Dict[str, Any]]:
    """
    Convert ngram ids to (optionally decoded) strings for readability.
    """
    out: List[Dict[str, Any]] = []
    for ids, c, frac in ngrams:
        decoded = None
        if tokenizer is not None:
            try:
                decoded = tokenizer.decode(list(ids))
            except Exception:
                decoded = None
        out.append(
            {
                "ids": list(map(int, ids)),
                "count": int(c),
                "frac": float(frac),
                "text": decoded,
            }
        )
    return out


def _dump_text_and_ngrams(
    *,
    x: torch.Tensor,
    tokenizer,
    V: int,
    run_dir: str,
    step: int,
    dump_n: int,
    topk_ngrams: int,
    ngram_ns: List[int],
) -> None:
    """
    Dump decoded samples and top ngrams to run_dir (USED FOR BOTH remasking modes now).
    """
    _ensure_dir(run_dir)
    step_tag = f"step{int(step)}"

    # 1) text samples
    texts = _decode_batch_text(x, tokenizer=tokenizer, V=V)[: int(dump_n)]
    txt_path = os.path.join(run_dir, f"samples_{step_tag}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(texts):
            f.write(f"[{i}]\n{s}\n\n")

    # 2) top ngrams (token-id based)
    ngrams_dump: Dict[str, Any] = {"step": int(step), "top_ngrams": {}}
    for n in ngram_ns:
        ng = _top_ngrams_token_ids(x, n=int(n), V=V, k=int(topk_ngrams))
        ngrams_dump["top_ngrams"][f"{n}gram"] = _format_top_ngrams_for_dump(ng, tokenizer=tokenizer)

    json_path = os.path.join(run_dir, f"top_ngrams_{step_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ngrams_dump, f, indent=2, ensure_ascii=False)

    # 3) human-readable ngram text
    pretty_path = os.path.join(run_dir, f"top_ngrams_{step_tag}.txt")
    with open(pretty_path, "w", encoding="utf-8") as f:
        f.write(f"Top ngrams | step={int(step)}\n\n")
        for n in ngram_ns:
            f.write(f"== {n}-grams ==\n")
            rows = ngrams_dump["top_ngrams"].get(f"{n}gram", [])
            if not rows:
                f.write("(empty)\n\n")
                continue
            for r in rows:
                ids = r["ids"]
                c = r["count"]
                frac = r["frac"]
                s = r.get("text", None)
                if s is None:
                    s = ""
                f.write(f"count={c:7d}  frac={frac:.6f}  ids={ids}  text={s}\n")
            f.write("\n")

    print(f"[DUMP] samples: {os.path.abspath(txt_path)}")
    print(f"[DUMP] ngrams : {os.path.abspath(json_path)}")
    print(f"[DUMP] ngrams : {os.path.abspath(pretty_path)}")


def _save_step_seqs_pt(
    *,
    x_cpu: torch.Tensor,  # [N,T] on CPU
    run_dir: str,
    step: int,
    V: int,
    T: int,
    seed: int,
    remasking: str,
    temperature: float,
    temp_beta: float,
    max_seqs: int,
) -> str:
    """
    Save full token-id sequences (N x T) to a .pt file for BOTH random/low_confidence.
    Stored under: run_dir/seqs/seqs_step{step}.pt
    """
    seq_dir = os.path.join(run_dir, "seqs")
    _ensure_dir(seq_dir)
    out_pt = os.path.join(seq_dir, f"seqs_step{int(step)}.pt")

    x_save = x_cpu
    if int(max_seqs) > 0:
        x_save = x_save[: int(max_seqs)]

    # reduce disk: int16 if safe (e.g., V<=4096), else int32
    if int(V) <= 32767:
        x_save = x_save.to(dtype=torch.int16)
    else:
        x_save = x_save.to(dtype=torch.int32)

    payload = {
        "x": x_save.contiguous(),
        "step": int(step),
        "V": int(V),
        "T": int(T),
        "seed": int(seed),
        "remasking": str(remasking),
        "temperature": float(temperature),
        "temp_beta": float(temp_beta),
    }
    torch.save(payload, out_pt)
    return out_pt


@torch.no_grad()
def sample_llada_via_generate(
    *,
    oracle: OracleHMMPosterior_LogRank1Teleport,
    pi0: torch.Tensor,
    N: int,
    T: int,
    V: int,
    steps: int,
    device: torch.device,
    seed: int,
    remasking: str,
    temperature: float,
    temp_beta: float,
    noise_removal: bool,
    use_attention_mask: bool,
    logits_eos_inf: bool,
    confidence_eos_eot_inf: bool,
    eos_id: int | None,
    eot_id: int | None,
    prompt: torch.Tensor,  # [N, Lp]
    prompt_attn: torch.Tensor | None,  # [N, Lp] or None
    prompt_len: int,
) -> torch.Tensor:

    mask_id = int(V)
    logits_fn = make_oracle_logits_fn(oracle, V=V, temp_beta=temp_beta)

    if prompt_len > T:
        raise ValueError(f"prompt_len={prompt_len} > T={T}")
    gen_length = int(T - prompt_len)
    if gen_length <= 0:
        raise ValueError(f"gen_length={gen_length} (T={T}, prompt_len={prompt_len}) must be positive.")

    attn_for_generate = None
    if use_attention_mask:
        attn_for_generate = prompt_attn if prompt_len > 0 else None

    x = generate(
        model=None,
        prompt=prompt,
        attention_mask=attn_for_generate,
        steps=int(steps),
        gen_length=int(gen_length),
        block_length=int(gen_length),
        temperature=float(temperature),
        cfg_scale=0.0,
        remasking=str(remasking),
        mask_id=mask_id,
        logits_fn=logits_fn,
        eos_id=int(eos_id) if eos_id is not None else 0,
        eot_id=int(eot_id) if eot_id is not None else 0,
        logits_eos_inf=bool(logits_eos_inf) if eos_id is not None else False,
        confidence_eos_eot_inf=bool(confidence_eos_eot_inf) if (eos_id is not None and eot_id is not None) else False,
        forbid_mask_prediction=True,
    )  # [N, T]

    x = x[:, :T]

    if noise_removal:
        x = _fill_remaining_masks_from_pi(x, pi=pi0, V=V, seed=int(seed))

    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=str, default="8,16,32,64,128,256")
    p.add_argument("--seed", type=int, default=123)

    # override eval N
    p.add_argument(
        "--N_eval",
        type=int,
        default=-1,
        help="Override eval sample size. -1 => use gt.N",
    )

    p.add_argument(
        "--N_chunk",
        type=int,
        default=128,
        help="Chunk size for sampling to reduce VRAM. 0/neg => no chunking.",
    )

    # NEW: save full ids for BOTH remasking modes (default ON)
    p.add_argument(
        "--save_seqs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save sampled token-id sequences (N x T) as .pt under run_dir/seqs/ (default: True).",
    )
    p.add_argument(
        "--save_max_seqs",
        type=int,
        default=-1,
        help="Max sequences to save per step. -1 => save all N_eval.",
    )

    p.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])

    # REQUIRED change: default temperature=1.0
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temp_beta", type=float, default=1.0)

    p.add_argument("--noise_removal", action="store_true")
    p.add_argument("--run_name", type=str, default="")

    p.add_argument("--use_attention_mask", action="store_true")

    p.add_argument("--logits_eos_inf", action="store_true")
    p.add_argument("--confidence_eos_eot_inf", action="store_true")
    p.add_argument("--eos_id", type=int, default=None)
    p.add_argument("--eot_id", type=int, default=None)

    # REQUIRED change: tokenizer_json default path
    p.add_argument(
        "--tokenizer_json",
        type=str,
        default="tokenizers/owt_bytebpe_v4096/tokenizer.json",
        help="HuggingFace tokenizers JSON (used for prompt encoding + dump decoding).",
    )

    # Prompt (default OFF)
    p.add_argument("--prompt_mode", type=str, default="none", choices=["none", "token", "text"])
    p.add_argument("--prompt_token_id", type=int, default=None)
    p.add_argument("--prompt_char", type=str, default=None, help="For char-GT only (must be length-1).")
    p.add_argument("--prompt_str", type=str, default=None, help="Must encode to exactly 1 token in token-mode.")
    p.add_argument("--prompt_text", type=str, default=None, help="Free-form multi-token prompt (text-mode).")

    # Dump controls (NOW active for BOTH remasking modes)
    p.add_argument("--dump_text_n", type=int, default=20, help="How many sequences to dump as text (per step).")
    p.add_argument("--dump_topk_ngrams", type=int, default=50, help="Top-k ngrams to dump (per n).")
    p.add_argument("--dump_ngram_ns", type=str, default="1,2,3", help="Which n for ngrams to dump, e.g. 1,2,3")

    # Keep this debug print knob
    p.add_argument("--sanity_print", action="store_true", help="print top unigrams/bigrams per step (debug)")
    p.add_argument("--sanity_k", type=int, default=15, help="top-k for sanity print")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    steps_list = parse_steps(args.steps)
    if not steps_list:
        raise ValueError("Empty --steps")
    steps_list = [int(s) for s in steps_list]

    gt = load_gt(args.gt, device=str(device))
    V = int(gt.V)
    T = int(gt.T)
    N_gt = int(gt.N)
    N = int(N_gt if int(args.N_eval) <= 0 else int(args.N_eval))

    nbr_idx = gt.nbr_idx.to(device=device, dtype=torch.long)
    nbr_prob = gt.nbr_prob.to(device=device, dtype=torch.float32)
    nu = gt.nu.to(device=device, dtype=torch.float32)
    eps_tp = float(gt.eps)
    pi0 = gt.pi.to(device=device, dtype=torch.float32)

    K = int(nbr_idx.shape[1])
    mask_id = V

    # tokenizer (tokenizers-json)
    tok_json = (args.tokenizer_json or "").strip()
    tokenizer = None
    try:
        tokenizer = _maybe_load_hf_tokenizer_json(tok_json) if tok_json else None
    except Exception as e:
        # We still allow running metrics without tokenizer; dumps will fallback to ids.
        print(f"[WARN] failed to load tokenizer_json={tok_json!r}: {e}")
        tokenizer = None

    eos_id, eot_id = _infer_eos_eot_ids_from_tokenizers_json(
        tokenizer=tokenizer,
        eos_id_cli=args.eos_id,
        eot_id_cli=args.eot_id,
    )

    # prompt batch size must match N_eval
    prompt, prompt_attn, prompt_meta = _build_prompt(
        prompt_mode=str(args.prompt_mode),
        prompt_token_id=args.prompt_token_id,
        prompt_char=args.prompt_char,
        prompt_str=args.prompt_str,
        prompt_text=args.prompt_text,
        tokenizer=tokenizer,
        gt=gt,
        device=device,
        N=N,
    )
    prompt_len = int(prompt.shape[1])

    # REQUIRED: step_seed is ALWAYS args.seed
    step_seed = int(args.seed)

    print(f"[GT] path={args.gt}")
    print(f"[GT] V={V}, T={T}, gt.N={N_gt}, eval.N={N}, K={K}, eps={eps_tp:g}")
    print(f"[CFG] N_chunk={int(args.N_chunk)} save_seqs={bool(args.save_seqs)} save_max_seqs={int(args.save_max_seqs)}")
    print(
        f"[CFG] remasking={args.remasking} temperature={args.temperature} temp_beta={args.temp_beta} "
        f"noise_removal={args.noise_removal} use_attention_mask={args.use_attention_mask} step_seed={step_seed}"
    )
    print(
        f"[CFG] prompt_mode={args.prompt_mode} prompt_len={prompt_len} "
        f"logits_eos_inf={args.logits_eos_inf} confidence_eos_eot_inf={args.confidence_eos_eot_inf} "
        f"eos_id={eos_id} eot_id={eot_id} tokenizer_json={tok_json}"
    )

    prior_metrics = MetricsSparseTeleportPrior(nbr_idx=nbr_idx, nbr_prob=nbr_prob, nu=nu, eps=eps_tp)
    prior_oracle = OracleSparseTeleportPrior(nbr_idx=nbr_idx, nbr_prob=nbr_prob, nu=nu, eps=eps_tp)

    oracle = OracleHMMPosterior_LogRank1Teleport(
        prior=prior_oracle,
        pi0=pi0,
        mask_id=mask_id,
        store_dtype=torch.float16,
        compute_dtype=torch.float32,
    ).to(device).eval()

    gt_base = os.path.splitext(os.path.basename(args.gt))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    knobs_tag = _sanitize(
        f"llada_rem{args.remasking}"
        f"_temp{_fmt_float_tag(args.temperature)}"
        f"_beta{_fmt_float_tag(args.temp_beta)}"
        f"_nr{int(bool(args.noise_removal))}"
        f"_am{int(bool(args.use_attention_mask))}"
        f"_eosL{int(bool(args.logits_eos_inf))}"
        f"_eosC{int(bool(args.confidence_eos_eot_inf))}"
        f"_pm{args.prompt_mode}_pl{prompt_len}"
        f"_Neval{int(N)}"
        f"_K{K}_eps{_fmt_float_tag(eps_tp)}"
    )

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{gt_base}_{knobs_tag}_seed{args.seed}_{timestamp}"

    out_root = os.path.join("sampler_output", "llada")
    plot_root = os.path.join("sampler_plots", "llada")
    _ensure_dir(out_root)
    _ensure_dir(plot_root)

    run_dir = os.path.join(out_root, run_name)
    run_plot_dir = os.path.join(plot_root, run_name)
    _ensure_dir(run_dir)
    _ensure_dir(run_plot_dir)

    # dumps dir (NOW for BOTH remasking modes)
    dump_dir = os.path.join(run_dir, "dumps")
    _ensure_dir(dump_dir)

    print(f"[OUT] run_dir={os.path.abspath(run_dir)}")
    print(f"[PLOT] plot_dir={os.path.abspath(run_plot_dir)}")

    metrics_json_path = os.path.join(run_dir, "metrics.json")
    metrics_jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")

    # ---- AR baseline ----
    # REQUIRED: AR baseline seed = args.seed + 777
    x_ar = sample_ar_sparse_teleport(pi=pi0, prior=prior_metrics, N=N, T=T, seed=int(args.seed) + 777, device=device)
    ar_nll = nll_transition_sparse_teleport(x_ar, prior_metrics)
    ar_rt = full_kl_rate_sparse_teleport(x_ar, prior_metrics)
    ar_uni = unigram_l1(x_ar, pi=pi0, V=V)
    ar_u2 = unique_ngram_ratio(x_ar, n=2)
    ar_u3 = unique_ngram_ratio(x_ar, n=3)
    ar_dup = dup_rate(x_ar)

    ar_rec: Dict[str, Any] = {
        "type": "baseline_ar",
        "steps": 0,
        "seed": int(args.seed) + 777,
        "nll_token": float(ar_nll),
        "full_kl_rate": float(ar_rt["full_kl_rate"]),
        "full_tv_rate": float(ar_rt["full_tv_rate"]),
        "full_entropy_rate": float(ar_rt["full_entropy_rate"]),
        "unigram_L1": float(ar_uni),
        "unique_2gram_ratio": float(ar_u2),
        "unique_3gram_ratio": float(ar_u3),
        "dup_rate": float(ar_dup),
        "other_mass_rate": float(ar_rt["other_mass_rate"]),
        "support_frac": float(ar_rt["support_frac"]),
    }

    print("\n[AR baseline]")
    print(
        f"  AR | NLL/token={ar_nll:.6f} | fKL={ar_rt['full_kl_rate']:.3e} "
        f"| fTV={ar_rt['full_tv_rate']:.3e} | fH={ar_rt['full_entropy_rate']:.3f} "
        f"| uniL1={ar_uni:.3e} | u2={ar_u2:.4f} u3={ar_u3:.4f} | dup={ar_dup:.4f} "
        f"| other={ar_rt['other_mass_rate']:.4f} | supp={ar_rt['support_frac']:.4f}"
    )

    vocab = gt.vocab if hasattr(gt, "vocab") and isinstance(gt.vocab, list) else None
    meta = gt.config if hasattr(gt, "config") and isinstance(gt.config, dict) else {}

    header: Dict[str, Any] = {
        "type": "header",
        "gt_path": args.gt,
        "device": str(device),
        "seed": int(args.seed),
        "step_seed": int(step_seed),
        "V": int(V),
        "T": int(T),
        "gt_N": int(N_gt),
        "N_eval": int(N),
        "N_chunk": int(args.N_chunk),
        "K": int(K),
        "eps": float(eps_tp),
        "llada_cfg": {
            "remasking": str(args.remasking),
            "temperature": float(args.temperature),
            "temp_beta": float(args.temp_beta),
            "noise_removal": bool(args.noise_removal),
            "use_attention_mask": bool(args.use_attention_mask),
            "logits_eos_inf": bool(args.logits_eos_inf),
            "confidence_eos_eot_inf": bool(args.confidence_eos_eot_inf),
            "eos_id": eos_id,
            "eot_id": eot_id,
            "tokenizer_json": tok_json,
            "prompt": prompt_meta,
            "save_seqs": bool(args.save_seqs),
            "save_max_seqs": int(args.save_max_seqs),
        },
        "gt_meta": meta,
        "ar_baseline": ar_rec,
        "notes": (
            "Oracle LLaDA: logits from exact HMM hard-evidence posterior on sparse teleport Markov prior. "
            "Sampling uses generate_llada.generate with Gumbel-max noise controlled by --temperature. "
            "step_seed is fixed to args.seed for all steps; AR baseline uses args.seed+777."
        ),
    }

    with open(metrics_jsonl_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        f.write(json.dumps(ar_rec) + "\n")

    # ---- steps sweep ----
    print(f"\n[LLaDA] steps sweep: {steps_list}")
    rows: List[Dict[str, Any]] = []

    # dump ngram settings parse
    dump_ngram_ns: List[int] = []
    try:
        dump_ngram_ns = [int(x) for x in str(args.dump_ngram_ns).split(",") if str(x).strip()]
        dump_ngram_ns = [n for n in dump_ngram_ns if n >= 1]
        if not dump_ngram_ns:
            dump_ngram_ns = [1, 2, 3]
    except Exception:
        dump_ngram_ns = [1, 2, 3]

    for s in steps_list:
        # REQUIRED: step_seed恒等于 args.seed
        step_seed = int(args.seed)
        # --- Seed ONCE per step ---
        torch.manual_seed(int(step_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(step_seed))

        # --- chunked sampling to reduce VRAM ---
        N_chunk = int(args.N_chunk)
        if N_chunk <= 0:
            N_chunk = N  # no chunking

        x_parts_cpu: List[torch.Tensor] = []
        for st in range(0, N, N_chunk):
            ed = min(N, st + N_chunk)
            n_i = ed - st

            prompt_i = prompt[st:ed]
            prompt_attn_i = prompt_attn[st:ed] if prompt_attn is not None else None

            x_i = sample_llada_via_generate(
                oracle=oracle,
                pi0=pi0,
                N=int(n_i),
                T=T,
                V=V,
                steps=int(s),
                device=device,
                seed=int(step_seed),
                remasking=str(args.remasking),
                temperature=float(args.temperature),
                temp_beta=float(args.temp_beta),
                noise_removal=bool(args.noise_removal),
                use_attention_mask=bool(args.use_attention_mask),
                logits_eos_inf=bool(args.logits_eos_inf),
                confidence_eos_eot_inf=bool(args.confidence_eos_eot_inf),
                eos_id=eos_id,
                eot_id=eot_id,
                prompt=prompt_i,
                prompt_attn=prompt_attn_i,
                prompt_len=prompt_len,
            )

            # move chunk result off GPU immediately
            x_parts_cpu.append(x_i.detach().to("cpu"))

            # free some cached memory between chunks (helps fragmentation)
            del x_i
            if device.type == "cuda":
                torch.cuda.empty_cache()

        x_ll_cpu = torch.cat(x_parts_cpu, dim=0)  # [N, T] on CPU

        # NEW: save ids for BOTH random and low_confidence (default ON)
        if bool(args.save_seqs):
            try:
                out_pt = _save_step_seqs_pt(
                    x_cpu=x_ll_cpu,
                    run_dir=run_dir,
                    step=int(s),
                    V=V,
                    T=T,
                    seed=int(step_seed),
                    remasking=str(args.remasking),
                    temperature=float(args.temperature),
                    temp_beta=float(args.temp_beta),
                    max_seqs=int(args.save_max_seqs),
                )
                print(f"[OK] Saved seq ids: {os.path.abspath(out_pt)}")
            except Exception as e:
                print(f"[WARN] failed to save seq ids at step={s}: {e}")

        x_ll = x_ll_cpu.to(device=device, dtype=torch.long)

        if (x_ll == mask_id).any():
            raise RuntimeError("Output still has MASK. Increase steps or enable --noise_removal.")

        nll_tok = nll_transition_sparse_teleport(x_ll, prior_metrics)
        rt = full_kl_rate_sparse_teleport(x_ll, prior_metrics)
        uni = unigram_l1(x_ll, pi=pi0, V=V)
        u2 = unique_ngram_ratio(x_ll, n=2)
        u3 = unique_ngram_ratio(x_ll, n=3)
        dr = dup_rate(x_ll)

        rec: Dict[str, Any] = {
            "type": "step",
            "model": "llada-oracle",
            "steps": int(s),
            "seed": int(step_seed),
            "nll_token": float(nll_tok),
            "full_kl_rate": float(rt["full_kl_rate"]),
            "full_tv_rate": float(rt["full_tv_rate"]),
            "full_entropy_rate": float(rt["full_entropy_rate"]),
            "unigram_L1": float(uni),
            "unique_2gram_ratio": float(u2),
            "unique_3gram_ratio": float(u3),
            "dup_rate": float(dr),
            "other_mass_rate": float(rt["other_mass_rate"]),
            "support_frac": float(rt["support_frac"]),
        }
        rows.append(rec)

        with open(metrics_jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        print(
            f"  step={s:4d} | seed={step_seed} | NLL/token={nll_tok:.6f} | fKL={rt['full_kl_rate']:.3e} "
            f"| fTV={rt['full_tv_rate']:.3e} | fH={rt['full_entropy_rate']:.3f} "
            f"| uniL1={uni:.3e} | u2={u2:.4f} u3={u3:.4f} | dup={dr:.4f} "
            f"| other={rt['other_mass_rate']:.4f} | supp={rt['support_frac']:.4f}"
        )

        # Keep existing sanity_print behavior
        if args.sanity_print:
            if top_unigrams_bigrams_print is None:
                print("[WARN] sanity_print requested but top_unigrams_bigrams_print import failed.")
            else:
                top_unigrams_bigrams_print(x_ll, V=V, k=args.sanity_k, vocab=vocab)

        # UPDATED: dump for BOTH random + low_confidence
        try:
            _dump_text_and_ngrams(
                x=x_ll,
                tokenizer=tokenizer,
                V=V,
                run_dir=dump_dir,
                step=int(s),
                dump_n=int(args.dump_text_n),
                topk_ngrams=int(args.dump_topk_ngrams),
                ngram_ns=dump_ngram_ns,
            )
        except Exception as e:
            print(f"[WARN] dump failed at step={s}: {e}")

    summary = {**header, "results": rows}
    with open(metrics_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n[OK] Saved metrics:\n  - {os.path.abspath(metrics_json_path)}\n  - {os.path.abspath(metrics_jsonl_path)}\n  - {os.path.abspath(metrics_csv_path)}"
    )
    print(f"[OK] Dumps (ALL remasking): {os.path.abspath(dump_dir)}")
    if bool(args.save_seqs):
        print(f"[OK] Seqs  (ALL remasking): {os.path.abspath(os.path.join(run_dir, 'seqs'))}")

    xs = [r["steps"] for r in rows]

    def _plot(ykey: str, title: str, ylog: bool = False, ar_value: Optional[float] = None):
        ys = [float(r[ykey]) for r in rows]
        outpath = os.path.join(run_plot_dir, f"{ykey}_vs_steps_{knobs_tag}.png")
        _plot_curve(
            xs,
            ys,
            title=title,
            xlabel="steps",
            ylabel=ykey,
            outpath=outpath,
            ylog=ylog,
            ar_value=ar_value,
        )
        print(f"[OK] Saved plot: {os.path.abspath(outpath)}")

    _plot(
        "nll_token",
        f"NLL/token under P' (sparse+teleport) | V={V} T={T} N={N} K={K}",
        ylog=False,
        ar_value=ar_rec["nll_token"],
    )
    _plot("full_kl_rate", f"FULL KL-rate | V={V} T={T} N={N} K={K}", ylog=True, ar_value=ar_rec["full_kl_rate"])
    _plot("full_tv_rate", f"FULL TV-rate | V={V} T={T} N={N} K={K}", ylog=False, ar_value=ar_rec["full_tv_rate"])
    _plot(
        "full_entropy_rate",
        f"FULL entropy-rate | V={V} T={T} N={N} K={K}",
        ylog=False,
        ar_value=ar_rec["full_entropy_rate"],
    )
    _plot("support_frac", f"support fraction (unique edges / M) | V={V} T={T} N={N}", ylog=True, ar_value=ar_rec["support_frac"])

    _plot("unigram_L1", f"unigram L1 vs pi | V={V} T={T} N={N}", ylog=True, ar_value=ar_rec["unigram_L1"])
    _plot("unique_2gram_ratio", f"unique 2-gram ratio | V={V} T={T} N={N}", ylog=False, ar_value=ar_rec["unique_2gram_ratio"])
    _plot("unique_3gram_ratio", f"unique 3-gram ratio | V={V} T={T} N={N}", ylog=False, ar_value=ar_rec["unique_3gram_ratio"])
    _plot("dup_rate", f"duplicate sequence rate | V={V} T={T} N={N}", ylog=False, ar_value=ar_rec["dup_rate"])
    _plot("other_mass_rate", f"other-mass rate | V={V} T={T} N={N} K={K}", ylog=False, ar_value=ar_rec["other_mass_rate"])


if __name__ == "__main__":
    main()
