#!/usr/bin/env python3
# sampler/llada/run_llada_prompt.py
# ------------------------------------------------------------
# Oracle LLaDA runner (ReMDM-aligned outputs + prompt-aware metrics), using author's generate().
#
# Keeps the old interfaces, but in --conditional mode we focus on prompt_len=1:
#   - Always run:
#       (1) AR baseline (L=0)
#       (2) LLaDA unconditional (L=0)
#       (3) LLaDA conditional (L=1) with prompt token sampled from:
#           - pi   : token ~ stationary pi0
#           - head : token ~ head(pi0) (top-mass subset, default 30%)
#           - tail : token ~ tail(pi0) (low-mass subset, default 30%)
#
# NEW:
#   --N_eval : override evaluation sample size (like run_llada.py)
#
# Outputs:
#   - metrics.csv / metrics.json / metrics.jsonl : per-step full results
#   - summary_laststep.csv / summary_laststep.tex : one table at max(steps)
#   - plots: for EVERY metric, a single plot with curves:
#       AR + LLaDA-uncond + LLaDA(pi/head/tail)
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
    nll_transition_sparse_teleport_conditional,
    full_kl_rate_sparse_teleport_conditional,
    unigram_l1_suffix,
    unique_ngram_ratio_suffix,
    dup_rate_suffix,
)

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


# -------------------------
# dataset tag helper
# -------------------------
def _nice_dataset_name(meta: Dict[str, Any]) -> str:
    ds = str(meta.get("dataset", "") or meta.get("data", "") or meta.get("corpus", "")).strip().lower()
    if ds in ("text8", "text8char", "text8-char", "text8_char"):
        return "text8-char"
    if ds in ("owt", "openwebtext", "open_web_text"):
        return "OWT"
    if ds in ("stack", "the_stack", "the-stack", "stack_py", "stack-python", "stack_python", "stackpy"):
        return "Stack"
    return str(meta.get("dataset", "dataset")).strip() or "dataset"


def _make_dataset_tag(meta: Dict[str, Any], *, V_eff: int) -> Tuple[str, str]:
    """
    Returns:
      ds_tag_human: e.g., "Stack (BPE, tokV=4096, Veff=4096)" or "text8-char (char, Veff=27)"
      ds_tag_file : sanitized short tag for filenames
    """
    name = _nice_dataset_name(meta)
    tok = str(meta.get("tokenizer", "") or meta.get("tok", "") or "").strip().lower()
    tokV = meta.get("V", None)
    V_keep = meta.get("V_keep", meta.get("Vkeep", None))

    if not tok:
        if "text8" in name.lower():
            tok = "char"
        else:
            tok = "bpe"

    if tok in ("char", "character"):
        ds_tag_human = f"{name} (char, Veff={int(V_eff)})"
    else:
        if tokV is not None:
            try:
                tokV_i = int(tokV)
            except Exception:
                tokV_i = None
        else:
            tokV_i = None

        extra = ""
        if V_keep is not None:
            try:
                V_keep_i = int(V_keep)
                extra = f", Vkeep={V_keep_i}"
            except Exception:
                extra = ""

        if tokV_i is None:
            ds_tag_human = f"{name} (BPE, Veff={int(V_eff)})"
        else:
            ds_tag_human = f"{name} (BPE, tokV={tokV_i}, Veff={int(V_eff)}{extra})"

    ds_tag_file = _sanitize(
        ds_tag_human.replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("=", "")
        .replace(" ", "")
        .replace("-", "")
    )
    return ds_tag_human, ds_tag_file


# -------------------------
# utils
# -------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_int_list(s: str) -> List[int]:
    xs: List[int] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        xs.append(int(tok))
    return xs


def _maybe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _plot_curve_multi(
    xs: List[int],
    series: List[Tuple[str, List[float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
    ylog: bool = False,
    ar_value: Optional[float] = None,
) -> None:
    plt.figure(figsize=(9.8, 5.3))
    for name, ys in series:
        plt.plot(xs, ys, marker="o", linewidth=2.2, markersize=5.5, label=name)
    if ar_value is not None:
        plt.axhline(ar_value, linestyle="--", linewidth=2.0, label="AR baseline")

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


def _write_tex_table(path: str, rows: List[Dict[str, Any]], caption: str, label: str) -> None:
    cols = [
        ("curve", "Curve"),
        ("prompt_len", "L"),
        ("steps", "Steps"),
        ("nll_token", "NLL/token"),
        ("full_kl_rate", "Full KL-rate"),
        ("full_tv_rate", "Full TV-rate"),
        ("full_entropy_rate", "Full H-rate"),
        ("unigram_L1", "Unigram L1"),
        ("unique_2gram_ratio", "U-2gram"),
        ("unique_3gram_ratio", "U-3gram"),
        ("dup_rate", "Dup"),
        ("other_mass_rate", "OtherMass"),
        ("support_frac", "SuppFrac"),
    ]

    def fmt(k: str, v: Any) -> str:
        if k in ("curve",):
            return str(v)
        if k in ("prompt_len", "steps"):
            return str(int(v))
        x = _maybe_float(v)
        if k == "nll_token":
            return f"{x:.6f}"
        if "ratio" in k or k in ("dup_rate", "support_frac", "other_mass_rate"):
            return f"{x:.4f}"
        if "entropy" in k:
            return f"{x:.3f}"
        if "kl" in k or "tv" in k:
            return f"{x:.3e}"
        return f"{x:.6g}"

    with open(path, "w") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write("  \\begin{tabular}{lrr" + "r" * (len(cols) - 3) + "}\n")
        f.write("    \\toprule\n")
        f.write("    " + " & ".join(h for _, h in cols) + " \\\\\n")
        f.write("    \\midrule\n")
        for r in rows:
            f.write("    " + " & ".join(fmt(k, r.get(k, "")) for k, _ in cols) + " \\\\\n")
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write(f"  \\caption{{{caption}}}\n")
        f.write(f"  \\label{{{label}}}\n")
        f.write("\\end{table*}\n")


# -------------------------
# oracle logits for generate()
# -------------------------
@torch.no_grad()
def make_oracle_logits_fn(
    oracle: OracleHMMPosterior_LogRank1Teleport,
    *,
    V: int,
    temp_beta: float,
) -> Any:
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
        logits[..., V] = NINF  # forbid mask prediction
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


def _maybe_load_tokenizer(name_or_path: str | None):
    if not name_or_path:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "transformers is required for tokenizer-based prompt/eos handling. "
            "Install transformers or avoid tokenizer-dependent options."
        ) from e

    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if getattr(tok, "padding_side", None) != "left":
        tok.padding_side = "left"
    return tok


def _infer_eos_eot_ids(
    *,
    tokenizer,
    eos_id_cli: int | None,
    eot_id_cli: int | None,
) -> Tuple[int | None, int | None]:
    eos_id = eos_id_cli
    eot_id = eot_id_cli

    if tokenizer is None:
        return eos_id, eot_id

    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)

    if eot_id is None:
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = {}

        candidates = [
            "<|eot_id|>",
            "<|EOT|>",
            "<EOT>",
            "<|endoftext|>",
        ]
        for k in candidates:
            if k in vocab:
                eot_id = int(vocab[k])
                break

    return eos_id, eot_id


# -------------------------
# manual prompt (kept for backwards compatibility)
# -------------------------
def _build_prompt(
    *,
    prompt_mode: str,
    prompt_token_id: int | None,
    prompt_char: str | None,
    prompt_str: str | None,
    prompt_text: str | None,
    tokenizer,
    gt,
    device: torch.device,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor | None, Dict[str, Any]]:
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
                raise ValueError("--prompt_char must be exactly 1 character for Markov-chain conditioning.")
            if prompt_char not in gt.stoi:
                raise ValueError(f"--prompt_char={prompt_char!r} not found in gt.stoi vocab.")
            ids = [int(gt.stoi[prompt_char])]
            meta = {"mode": "token", "source": "prompt_char", "char": prompt_char, "ids": ids}

        elif prompt_str is not None:
            if tokenizer is None:
                raise ValueError("--prompt_str requires --tokenizer_name_or_path.")
            enc = tokenizer(prompt_str, add_special_tokens=False, return_tensors=None)
            tok_ids = enc["input_ids"]
            if isinstance(tok_ids[0], list):
                tok_ids = tok_ids[0]
            tok_ids = list(map(int, tok_ids))
            if len(tok_ids) != 1:
                raise ValueError(
                    f"--prompt_mode token requires prompt_str to tokenize into exactly 1 token, "
                    f"but got {len(tok_ids)} tokens. Use --prompt_mode text instead."
                )
            ids = tok_ids
            meta = {"mode": "token", "source": "prompt_str", "prompt_str": prompt_str, "ids": ids}

        else:
            raise ValueError("--prompt_mode token requires one of: --prompt_token_id, --prompt_char, --prompt_str")

        prompt_1 = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, Lp]
        prompt = prompt_1.repeat(N, 1)
        attn = torch.ones((N, prompt.shape[1]), dtype=torch.long, device=device)
        meta["prompt_len"] = int(prompt.shape[1])
        return prompt, attn, meta

    if prompt_mode == "text":
        if prompt_text is None:
            raise ValueError("--prompt_mode text requires --prompt_text.")
        if tokenizer is None:
            raise ValueError("--prompt_text requires --tokenizer_name_or_path.")
        enc = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
        ids_1 = enc["input_ids"].to(device=device, dtype=torch.long)  # [1, Lp]
        prompt = ids_1.repeat(N, 1)
        attn = torch.ones((N, prompt.shape[1]), dtype=torch.long, device=device)
        meta = {"mode": "text", "prompt_text": prompt_text, "prompt_len": int(prompt.shape[1])}
        return prompt, attn, meta

    raise ValueError(f"Unknown --prompt_mode {prompt_mode!r}")


# -------------------------
# conditional prompt sampling (L=1)
# -------------------------
@torch.no_grad()
def _sample_prompt_len1_pi(*, pi0: torch.Tensor, N: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    ids = torch.multinomial(pi0.detach().cpu(), num_samples=N, replacement=True, generator=g)
    return ids.to(device=device, dtype=torch.long).unsqueeze(1)


@torch.no_grad()
def _subset_by_cum_mass_desc(pi0: torch.Tensor, mass: float) -> Tuple[torch.Tensor, torch.Tensor]:
    mass = float(mass)
    if not (0.0 < mass <= 1.0):
        raise ValueError(f"mass must be in (0,1], got {mass}")
    p = pi0.detach().float().cpu()
    p = p / p.sum().clamp_min(1e-12)
    probs, idx = torch.sort(p, descending=True)
    c = torch.cumsum(probs, dim=0)
    keep = c <= mass
    if not keep.any():
        keep[0] = True
    last = int(keep.sum().item())
    if last < probs.numel():
        keep[last] = True
    head_idx = idx[keep]
    head_probs = probs[keep]
    head_probs = head_probs / head_probs.sum().clamp_min(1e-12)
    return head_idx, head_probs


@torch.no_grad()
def _subset_by_cum_mass_asc(pi0: torch.Tensor, mass: float) -> Tuple[torch.Tensor, torch.Tensor]:
    mass = float(mass)
    if not (0.0 < mass <= 1.0):
        raise ValueError(f"mass must be in (0,1], got {mass}")
    p = pi0.detach().float().cpu()
    p = p / p.sum().clamp_min(1e-12)
    probs, idx = torch.sort(p, descending=False)
    c = torch.cumsum(probs, dim=0)
    keep = c <= mass
    if not keep.any():
        keep[0] = True
    last = int(keep.sum().item())
    if last < probs.numel():
        keep[last] = True
    tail_idx = idx[keep]
    tail_probs = probs[keep]
    tail_probs = tail_probs / tail_probs.sum().clamp_min(1e-12)
    return tail_idx, tail_probs


@torch.no_grad()
def _sample_prompt_len1_head(
    *, pi0: torch.Tensor, N: int, seed: int, device: torch.device, head_mass: float
) -> torch.Tensor:
    head_idx, head_probs = _subset_by_cum_mass_desc(pi0, head_mass)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    draw = torch.multinomial(head_probs, num_samples=N, replacement=True, generator=g)
    ids = head_idx[draw]
    return ids.to(device=device, dtype=torch.long).unsqueeze(1)


@torch.no_grad()
def _sample_prompt_len1_tail(
    *, pi0: torch.Tensor, N: int, seed: int, device: torch.device, tail_mass: float
) -> torch.Tensor:
    tail_idx, tail_probs = _subset_by_cum_mass_asc(pi0, tail_mass)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    draw = torch.multinomial(tail_probs, num_samples=N, replacement=True, generator=g)
    ids = tail_idx[draw]
    return ids.to(device=device, dtype=torch.long).unsqueeze(1)


# -------------------------
# core sampling
# -------------------------
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
    prompt: torch.Tensor,              # [N, Lp]
    prompt_attn: torch.Tensor | None,  # [N, Lp] or None
    prompt_len: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    mask_id = int(V)
    logits_fn = make_oracle_logits_fn(oracle, V=V, temp_beta=temp_beta)

    if prompt_len > T:
        raise ValueError(f"prompt_len={prompt_len} > T={T}")
    gen_length = int(T - prompt_len)
    if gen_length <= 0:
        raise ValueError(f"gen_length={gen_length} must be positive (T={T}, prompt_len={prompt_len}).")

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
        x = _fill_remaining_masks_from_pi(x, pi=pi0, V=V, seed=seed)

    return x


def _compute_metrics(
    x: torch.Tensor,
    *,
    prior: MetricsSparseTeleportPrior,
    pi0: torch.Tensor,
    V: int,
    prompt_len: int,
) -> Dict[str, float]:
    nll_tok = nll_transition_sparse_teleport_conditional(x, prior, prompt_len=prompt_len)
    rt = full_kl_rate_sparse_teleport_conditional(x, prior, prompt_len=prompt_len)
    uni = unigram_l1_suffix(x, pi=pi0, V=V, prompt_len=prompt_len)
    u2 = unique_ngram_ratio_suffix(x, n=2, prompt_len=prompt_len)
    u3 = unique_ngram_ratio_suffix(x, n=3, prompt_len=prompt_len)
    dr = dup_rate_suffix(x, prompt_len=prompt_len)
    out = {
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
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=str, default="8,16,32,64,128,256")
    p.add_argument("--seed", type=int, default=123)

    # NEW: override evaluation N
    p.add_argument(
        "--N_eval",
        type=int,
        default=-1,
        help="Override eval sample size. -1 => use gt.N",
    )

    # fixed   -> same seed for all steps
    # perstep -> seed + 1000*steps
    p.add_argument("--seed_mode", type=str, default="fixed", choices=["fixed", "perstep"])

    p.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--temp_beta", type=float, default=1.0)

    p.add_argument("--noise_removal", action="store_true")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--use_attention_mask", action="store_true")

    p.add_argument("--logits_eos_inf", action="store_true")
    p.add_argument("--confidence_eos_eot_inf", action="store_true")
    p.add_argument("--eos_id", type=int, default=None)
    p.add_argument("--eot_id", type=int, default=None)
    p.add_argument("--tokenizer_name_or_path", type=str, default="", help="Needed for prompt_str/text or eos/eot auto.")

    # manual prompt (only when not --conditional)
    p.add_argument("--prompt_mode", type=str, default="none", choices=["none", "token", "text"])
    p.add_argument("--prompt_token_id", type=int, default=None)
    p.add_argument("--prompt_char", type=str, default=None, help="For char-GT only (must be length-1).")
    p.add_argument("--prompt_str", type=str, default=None, help="Must tokenize to exactly 1 token in token-mode.")
    p.add_argument("--prompt_text", type=str, default=None, help="Free-form multi-token prompt (text-mode).")

    # conditional protocol
    p.add_argument("--conditional", action="store_true", help="Enable conditional protocol (we focus on L=1).")
    p.add_argument("--prompt_lens", type=str, default="0,1", help="Comma-separated prompt lengths (kept).")
    p.add_argument("--prompt_seed", type=int, default=-1, help="Seed for conditional prompt sampling. -1 => seed+4242.")

    # head/tail masses
    p.add_argument("--prompt_head_mass", type=float, default=0.30, help="For head prompts: top-mass subset size.")
    p.add_argument("--prompt_tail_mass", type=float, default=0.30, help="For tail prompts: low-mass subset size.")

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
    max_steps = int(max(steps_list))

    gt = load_gt(args.gt, device=str(device))
    V = int(gt.V)  # V_eff
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

    tok_name = args.tokenizer_name_or_path.strip() or None
    tokenizer = _maybe_load_tokenizer(tok_name)

    eos_id, eot_id = _infer_eos_eot_ids(
        tokenizer=tokenizer,
        eos_id_cli=args.eos_id,
        eot_id_cli=args.eot_id,
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

    # manual prompt (only used when not --conditional)
    prompt_manual, prompt_attn_manual, prompt_meta_manual = _build_prompt(
        prompt_mode=str(args.prompt_mode),
        prompt_token_id=args.prompt_token_id,
        prompt_char=args.prompt_char,
        prompt_str=args.prompt_str,
        prompt_text=args.prompt_text,
        tokenizer=tokenizer,
        gt=gt,
        device=device,
        N=N,  # IMPORTANT: match eval N
    )
    prompt_len_manual = int(prompt_manual.shape[1])

    # conditional prompt seed base
    prompt_seed_base = (int(args.seed) + 4242) if int(args.prompt_seed) == -1 else int(args.prompt_seed)

    # parse prompt lens (kept)
    prompt_lens_list = _parse_int_list(args.prompt_lens)
    if len(prompt_lens_list) == 0:
        prompt_lens_list = [0, 1]

    # precompute per-step seeds once
    step_seed_map: Dict[int, int] = {}
    for s in steps_list:
        step_seed_map[int(s)] = int(args.seed) if args.seed_mode == "fixed" else int(args.seed + 1000 * int(s))

    # ------------------------------------------------------------
    # tags / run dirs
    # ------------------------------------------------------------
    gt_base = os.path.splitext(os.path.basename(args.gt))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta = gt.config if hasattr(gt, "config") and isinstance(gt.config, dict) else {}
    ds_tag_human, ds_tag_file = _make_dataset_tag(meta, V_eff=V)

    knobs_tag = _sanitize(
        f"ds{ds_tag_file}_"
        f"llada_rem{args.remasking}"
        f"_temp{_fmt_float_tag(args.temperature)}"
        f"_beta{_fmt_float_tag(args.temp_beta)}"
        f"_nr{int(bool(args.noise_removal))}"
        f"_am{int(bool(args.use_attention_mask))}"
        f"_eosL{int(bool(args.logits_eos_inf))}"
        f"_eosC{int(bool(args.confidence_eos_eot_inf))}"
        f"_cond{int(bool(args.conditional))}"
        f"_hm{_fmt_float_tag(args.prompt_head_mass)}"
        f"_tm{_fmt_float_tag(args.prompt_tail_mass)}"
        f"_pm{args.prompt_mode}_pl{prompt_len_manual}"
        f"_Neval{int(N)}"
        f"_sm{args.seed_mode}"
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

    print(f"[GT] path={args.gt}")
    print(f"[GT] tag={ds_tag_human}")
    print(f"[GT] Veff={V}, T={T}, gt.N={N_gt}, eval.N={N}, K={K}, eps={eps_tp:g}")
    print(
        f"[CFG] remasking={args.remasking} temperature={args.temperature} temp_beta={args.temp_beta} "
        f"noise_removal={args.noise_removal} use_attention_mask={args.use_attention_mask} seed_mode={args.seed_mode}"
    )
    print(
        f"[CFG] conditional={bool(args.conditional)} prompt_lens={prompt_lens_list} prompt_seed_base={prompt_seed_base} "
        f"(manual prompt_mode={args.prompt_mode}, manual_prompt_len={prompt_len_manual})"
    )
    print(
        f"[CFG] head_mass={args.prompt_head_mass} tail_mass={args.prompt_tail_mass} "
        f"logits_eos_inf={args.logits_eos_inf} confidence_eos_eot_inf={args.confidence_eos_eot_inf} "
        f"eos_id={eos_id} eot_id={eot_id} tokenizer={tok_name}"
    )
    print(f"[OUT]  run_dir={os.path.abspath(run_dir)}")
    print(f"[PLOT] plot_dir={os.path.abspath(run_plot_dir)}")

    metrics_json_path = os.path.join(run_dir, "metrics.json")
    metrics_jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    summary_csv_path = os.path.join(run_dir, "summary_laststep.csv")
    summary_tex_path = os.path.join(run_dir, "summary_laststep.tex")

    vocab = gt.vocab if hasattr(gt, "vocab") and isinstance(gt.vocab, list) else None

    # ------------------------------------------------------------
    # Baseline AR (L=0)
    # ------------------------------------------------------------
    x_ar = sample_ar_sparse_teleport(pi=pi0, prior=prior_metrics, N=N, T=T, seed=args.seed + 777, device=device)
    ar_m = _compute_metrics(x_ar, prior=prior_metrics, pi0=pi0, V=V, prompt_len=0)
    ar_rec: Dict[str, Any] = {
        "type": "baseline_ar",
        "curve": "AR",
        "steps": 0,
        "seed": int(args.seed + 777),
        "prompt_len": 0,
        **ar_m,
    }

    print("\n[AR baseline]")
    print(
        f"  AR | NLL/token={ar_rec['nll_token']:.6f} | fKL={ar_rec['full_kl_rate']:.3e} "
        f"| fTV={ar_rec['full_tv_rate']:.3e} | fH={ar_rec['full_entropy_rate']:.3f} "
        f"| uniL1={ar_rec['unigram_L1']:.3e} | u2={ar_rec['unique_2gram_ratio']:.4f} u3={ar_rec['unique_3gram_ratio']:.4f} "
        f"| dup={ar_rec['dup_rate']:.4f}"
    )

    header: Dict[str, Any] = {
        "type": "header",
        "gt_path": args.gt,
        "gt_tag": ds_tag_human,
        "device": str(device),
        "seed": int(args.seed),
        "seed_mode": str(args.seed_mode),
        "V": int(V),
        "T": int(T),
        "gt_N": int(N_gt),
        "N_eval": int(N),
        "K": int(K),
        "eps": float(eps_tp),
        "max_steps": int(max_steps),
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
            "tokenizer_name_or_path": tok_name,
            "manual_prompt": prompt_meta_manual,
            "conditional": bool(args.conditional),
            "prompt_lens": prompt_lens_list,
            "prompt_seed_base": int(prompt_seed_base),
            "prompt_head_mass": float(args.prompt_head_mass),
            "prompt_tail_mass": float(args.prompt_tail_mass),
        },
        "gt_meta": meta,
        "ar_baseline": ar_rec,
        "notes": (
            "Oracle LLaDA: logits from exact HMM hard-evidence posterior on sparse teleport Markov prior. "
            "Sampling uses generate_llada.generate with Gumbel-max noise controlled by --temperature. "
            "Prompt-aware metrics: transition metrics include boundary transition; distribution/ngram/dup are suffix-only."
        ),
    }

    with open(metrics_jsonl_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        f.write(json.dumps(ar_rec) + "\n")

    # ------------------------------------------------------------
    # LLaDA sweep
    # ------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    print(f"\n[LLaDA] steps sweep: {steps_list}")

    def _run_one_curve(
        *,
        curve_name: str,
        prompt_len: int,
        prompt: torch.Tensor,
        prompt_attn: torch.Tensor | None,
    ) -> None:
        nonlocal rows
        print(f"\n[Curve] {curve_name} (L={prompt_len})")
        for s in steps_list:
            s_int = int(s)
            step_seed = step_seed_map[s_int]

            x_ll = sample_llada_via_generate(
                oracle=oracle,
                pi0=pi0,
                N=N,
                T=T,
                V=V,
                steps=s_int,
                device=device,
                seed=step_seed,
                remasking=str(args.remasking),
                temperature=float(args.temperature),
                temp_beta=float(args.temp_beta),
                noise_removal=bool(args.noise_removal),
                use_attention_mask=bool(args.use_attention_mask),
                logits_eos_inf=bool(args.logits_eos_inf),
                confidence_eos_eot_inf=bool(args.confidence_eos_eot_inf),
                eos_id=eos_id,
                eot_id=eot_id,
                prompt=prompt,
                prompt_attn=prompt_attn,
                prompt_len=int(prompt_len),
            )

            if (x_ll == mask_id).any():
                raise RuntimeError("Output still has MASK. Increase steps or enable --noise_removal (not author-aligned).")

            m = _compute_metrics(x_ll, prior=prior_metrics, pi0=pi0, V=V, prompt_len=int(prompt_len))

            rec: Dict[str, Any] = {
                "type": "step",
                "model": "llada-oracle",
                "curve": str(curve_name),
                "prompt_len": int(prompt_len),
                "steps": int(s_int),
                "seed": int(step_seed),
                "seed_mode": str(args.seed_mode),
                **m,
            }
            rows.append(rec)

            with open(metrics_jsonl_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

            print(
                f"  {curve_name:12s} step={s_int:4d} | seed={step_seed} | NLL/token={rec['nll_token']:.6f} "
                f"| fKL={rec['full_kl_rate']:.3e} | fTV={rec['full_tv_rate']:.3e} | fH={rec['full_entropy_rate']:.3f} "
                f"| uniL1={rec['unigram_L1']:.3e} | u2={rec['unique_2gram_ratio']:.4f} u3={rec['unique_3gram_ratio']:.4f} "
                f"| dup={rec['dup_rate']:.4f}"
            )

        if args.sanity_print and top_unigrams_bigrams_print is not None:
            step_seed = step_seed_map[int(max_steps)]
            x_rep = sample_llada_via_generate(
                oracle=oracle,
                pi0=pi0,
                N=N,
                T=T,
                V=V,
                steps=int(max_steps),
                device=device,
                seed=step_seed,
                remasking=str(args.remasking),
                temperature=float(args.temperature),
                temp_beta=float(args.temp_beta),
                noise_removal=bool(args.noise_removal),
                use_attention_mask=bool(args.use_attention_mask),
                logits_eos_inf=bool(args.logits_eos_inf),
                confidence_eos_eot_inf=bool(args.confidence_eos_eot_inf),
                eos_id=eos_id,
                eot_id=eot_id,
                prompt=prompt,
                prompt_attn=prompt_attn,
                prompt_len=int(prompt_len),
            )
            top_unigrams_bigrams_print(x_rep, V=V, k=args.sanity_k, vocab=vocab)

    if args.conditional:
        # Always run uncond(L=0) + conditional (L=1 pi/head/tail)
        prompt0 = torch.empty((N, 0), dtype=torch.long, device=device)
        _run_one_curve(curve_name="LLaDA-uncond", prompt_len=0, prompt=prompt0, prompt_attn=None)

        attn1 = torch.ones((N, 1), dtype=torch.long, device=device)

        prompt_pi = _sample_prompt_len1_pi(pi0=pi0, N=N, seed=prompt_seed_base + 0, device=device)
        _run_one_curve(curve_name="LLaDA-pi", prompt_len=1, prompt=prompt_pi, prompt_attn=attn1)

        prompt_head = _sample_prompt_len1_head(
            pi0=pi0, N=N, seed=prompt_seed_base + 1, device=device, head_mass=float(args.prompt_head_mass)
        )
        _run_one_curve(curve_name="LLaDA-head", prompt_len=1, prompt=prompt_head, prompt_attn=attn1)

        prompt_tail = _sample_prompt_len1_tail(
            pi0=pi0, N=N, seed=prompt_seed_base + 2, device=device, tail_mass=float(args.prompt_tail_mass)
        )
        _run_one_curve(curve_name="LLaDA-tail", prompt_len=1, prompt=prompt_tail, prompt_attn=attn1)

    else:
        mode = f"manual({args.prompt_mode})"
        _run_one_curve(
            curve_name=f"LLaDA-{mode}",
            prompt_len=int(prompt_len_manual),
            prompt=prompt_manual,
            prompt_attn=prompt_attn_manual,
        )

    # ------------------------------------------------------------
    # save metrics
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # summary_laststep (one table at max(steps))
    # ------------------------------------------------------------
    curves = sorted(set([r["curve"] for r in rows]))
    last_rows: List[Dict[str, Any]] = []

    # For AR, we display it at steps=max_steps (even though it doesn't depend on steps)
    ar_row = {
        "curve": "AR",
        "prompt_len": 0,
        "steps": int(max_steps),
        **{k: ar_rec[k] for k in [
            "nll_token", "full_kl_rate", "full_tv_rate", "full_entropy_rate",
            "unigram_L1", "unique_2gram_ratio", "unique_3gram_ratio", "dup_rate",
            "other_mass_rate", "support_frac"
        ]},
    }
    last_rows.append(ar_row)

    for c in curves:
        cand = [r for r in rows if r["curve"] == c and int(r["steps"]) == int(max_steps)]
        if len(cand) == 0:
            continue
        r = cand[-1]
        last_rows.append({
            "curve": r["curve"],
            "prompt_len": int(r["prompt_len"]),
            "steps": int(r["steps"]),
            "nll_token": float(r["nll_token"]),
            "full_kl_rate": float(r["full_kl_rate"]),
            "full_tv_rate": float(r["full_tv_rate"]),
            "full_entropy_rate": float(r["full_entropy_rate"]),
            "unigram_L1": float(r["unigram_L1"]),
            "unique_2gram_ratio": float(r["unique_2gram_ratio"]),
            "unique_3gram_ratio": float(r["unique_3gram_ratio"]),
            "dup_rate": float(r["dup_rate"]),
            "other_mass_rate": float(r["other_mass_rate"]),
            "support_frac": float(r["support_frac"]),
        })

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(last_rows[0].keys()))
        writer.writeheader()
        writer.writerows(last_rows)

    cap = f"LLaDA oracle (uncond + L=1 pi/head/tail), {ds_tag_human}, summarized at steps={max_steps}."
    lab = f"tab:llada_prompt_summary_laststep_{ds_tag_file}"
    _write_tex_table(summary_tex_path, last_rows, caption=cap, label=lab)

    print(f"[OK] Saved summary:\n  - {os.path.abspath(summary_csv_path)}\n  - {os.path.abspath(summary_tex_path)}")

    # ------------------------------------------------------------
    # plots: for every metric, ONE plot with curves
    #   AR + LLaDA-uncond + LLaDA(pi/head/tail)
    # ------------------------------------------------------------
    if not args.conditional:
        print("[INFO] Not conditional; skipping multi-curve overlay plots.")
        return

    want_order = ["LLaDA-uncond", "LLaDA-pi", "LLaDA-head", "LLaDA-tail"]
    have_curves = {c: [r for r in rows if r["curve"] == c] for c in want_order}
    for c in want_order:
        if len(have_curves[c]) == 0:
            print(f"[WARN] Missing curve {c}; overlay plots will omit it.")

    xs = [int(s) for s in steps_list]

    def _series_for_metric(metric_key: str) -> List[Tuple[str, List[float]]]:
        series: List[Tuple[str, List[float]]] = []
        for c in want_order:
            rr = sorted(have_curves[c], key=lambda z: int(z["steps"]))
            if len(rr) == 0:
                continue
            ys = [float(r[metric_key]) for r in rr]
            if len(ys) != len(xs):
                mp = {int(r["steps"]): float(r[metric_key]) for r in rr}
                ys = [mp.get(int(s), float("nan")) for s in xs]
            series.append((c, ys))
        return series

    plot_specs = [
        ("nll_token", False, "NLL/token under P'"),
        ("full_kl_rate", True, "FULL KL-rate"),
        ("full_tv_rate", False, "FULL TV-rate"),
        ("full_entropy_rate", False, "FULL entropy-rate"),
        ("support_frac", True, "support fraction"),
        ("unigram_L1", True, "unigram L1 vs pi"),
        ("unique_2gram_ratio", False, "unique 2-gram ratio"),
        ("unique_3gram_ratio", False, "unique 3-gram ratio"),
        ("dup_rate", False, "duplicate sequence rate"),
        ("other_mass_rate", False, "other-mass rate"),
    ]

    for key, ylog, title_short in plot_specs:
        outpath = os.path.join(run_plot_dir, f"{key}_all_{knobs_tag}.png")
        series = _series_for_metric(key)
        _plot_curve_multi(
            xs,
            series,
            title=f"{title_short} | {ds_tag_human} | T={T} N={N} K={K} (overlay)",
            xlabel="steps",
            ylabel=key,
            outpath=outpath,
            ylog=bool(ylog),
            ar_value=float(ar_rec[key]) if key in ar_rec else None,
        )
        print(f"[OK] Saved plot: {os.path.abspath(outpath)}")


if __name__ == "__main__":
    main()
