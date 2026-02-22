#!/usr/bin/env python3
# sampler/llada/run_llada_lowconf.py
# ------------------------------------------------------------
# Qualitative LLaDA runner (low_confidence OR random) to dump samples + top ngrams.
#
# - Uses author's generate() and SAME oracle logits_fn signature as run_llada.py:
#     logits_fn(x, attention_mask=None) -> [B,T,V+1]
# - Saves:
#     sampler_output/llada/<run_name>/samples/*_samples.txt
#     sampler_output/llada/<run_name>/samples/*_topngrams.txt
#
# NEW (OWT/Stack-friendly decoding):
#   --tokenizer_json  path/to/tokenizer.json (HF tokenizers format)
#   If provided, *_samples.txt will be decoded into readable text via tokenizer.decode().
#
# NOTE:
#   For OWT/Stack GT, gt usually doesn't store id2tok, so tokenizer_json is the best.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from sampler.gt_io import load_gt
from sampler.utils_io import parse_steps, _fmt_float_tag, _sanitize
from sampler.ar_baseline_sparse import sample_ar_sparse_teleport
from sampler.llada.generate_llada import generate

from sampler.oracle_hmm_posterior import (
    SparseTeleportPrior as OracleSparseTeleportPrior,
    OracleHMMPosterior_LogRank1Teleport,
)

from sampler.metrics_full import SparseTeleportPrior as MetricsSparseTeleportPrior

LOGIT_CLAMP = 1e-30
NINF = -1e30


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _maybe_load_hf_tokenizer_json(tokenizer_json: str | None):
    """
    Load HF `tokenizers` Tokenizer from tokenizer.json (your shared_bytebpe_V4096 output).
    Returns None if tokenizer_json is empty/None.
    """
    if not tokenizer_json:
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Need `tokenizers` to load tokenizer.json. Install with: pip install tokenizers"
        ) from e
    return Tokenizer.from_file(tokenizer_json)


@torch.no_grad()
def make_oracle_logits_fn(
    oracle: OracleHMMPosterior_LogRank1Teleport,
    *,
    V: int,
    temp_beta: float,
) -> Any:
    """
    logits_fn(x, attention_mask=None) -> logits [B,T,V+1]
    This matches generate_llada.generate() which calls logits_fn(x, attention_mask).
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
        logits[..., V] = NINF  # forbid MASK class
        return logits

    return logits_fn


def _maybe_get_id2tok(gt) -> Optional[List[str]]:
    # char GT: typically has itos or vocab
    if hasattr(gt, "itos") and isinstance(gt.itos, list) and len(gt.itos) > 0:
        return [str(x) for x in gt.itos]
    if hasattr(gt, "vocab") and isinstance(gt.vocab, list) and len(gt.vocab) > 0:
        return [str(x) for x in gt.vocab]
    return None


def _decode_seq(
    seq: List[int],
    *,
    id2tok: Optional[List[str]],
    hf_tok=None,
) -> str:
    """
    Prefer tokenizer_json decoding (best for OWT/Stack BPE).
    Fallback to id2tok (char-level).
    Final fallback: join ids.
    """
    if hf_tok is not None:
        try:
            # tokenizers.Tokenizer.decode expects List[int]
            return hf_tok.decode(seq, skip_special_tokens=True)
        except Exception:
            pass

    if id2tok is None:
        return " ".join(map(str, seq))

    # char-level: join directly
    try:
        return "".join(id2tok[i] for i in seq)
    except Exception:
        return " ".join(id2tok[i] if 0 <= i < len(id2tok) else str(i) for i in seq)


def _save_samples_txt(
    *,
    x: torch.Tensor,         # [N,T]
    out_txt: str,
    id2tok: Optional[List[str]],
    hf_tok,
    k: int,
    title: str,
) -> None:
    x_cpu = x.detach().cpu()
    N = x_cpu.shape[0]
    k = min(int(k), int(N))
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append(f"# N={N}  showing first k={k}")
    lines.append("")
    for i in range(k):
        seq = x_cpu[i].tolist()
        lines.append(f"[{i}]")
        lines.append(_decode_seq(seq, id2tok=id2tok, hf_tok=hf_tok))
        lines.append("")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _top_ngrams(
    x: torch.Tensor,  # [N,T]
    *,
    V: int,
    topk: int,
    id2tok: Optional[List[str]],
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Compute top unigrams/bigrams/trigrams across the whole sample batch.
    Note: for OWT/Stack (N=128,T=1024) this is fine.
    """
    x_cpu = x.detach().cpu()
    N, T = x_cpu.shape
    topk = int(topk)

    def tok(i: int) -> str:
        if id2tok is None:
            return str(i)
        if 0 <= i < len(id2tok):
            return id2tok[i]
        return str(i)

    # unigrams
    flat = x_cpu.reshape(-1)
    uni_counts = torch.bincount(flat, minlength=V).tolist()
    uni_items = [(i, c) for i, c in enumerate(uni_counts) if c > 0]
    uni_items.sort(key=lambda t: t[1], reverse=True)
    uni_items = uni_items[:topk]
    uni_named = [(tok(i), int(c)) for i, c in uni_items]

    # bigrams / trigrams via Counter
    bi = Counter()
    tri = Counter()
    for n in range(N):
        seq = x_cpu[n].tolist()
        for t in range(T - 1):
            bi[(seq[t], seq[t + 1])] += 1
        for t in range(T - 2):
            tri[(seq[t], seq[t + 1], seq[t + 2])] += 1

    bi_items = bi.most_common(topk)
    tri_items = tri.most_common(topk)

    if id2tok is not None:
        bi_named = [(f"{tok(a)}{tok(b)}", int(c)) for (a, b), c in bi_items]
        tri_named = [(f"{tok(a)}{tok(b)}{tok(c_)}", int(cnt)) for (a, b, c_), cnt in tri_items]
    else:
        bi_named = [(f"{a} {b}", int(c)) for (a, b), c in bi_items]
        tri_named = [(f"{a} {b} {c_}", int(cnt)) for (a, b, c_), cnt in tri_items]

    return {"unigram": uni_named, "bigram": bi_named, "trigram": tri_named}


def _save_topngrams_txt(
    *,
    x: torch.Tensor,
    out_txt: str,
    V: int,
    topk: int,
    id2tok: Optional[List[str]],
    title: str,
) -> None:
    stats = _top_ngrams(x, V=V, topk=topk, id2tok=id2tok)
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    for key in ["unigram", "bigram", "trigram"]:
        lines.append(f"## top-{topk} {key}s")
        for s, c in stats[key]:
            lines.append(f"{c:10d}  {s}")
        lines.append("")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@torch.no_grad()
def _sample_llada(
    *,
    oracle: OracleHMMPosterior_LogRank1Teleport,
    N: int,
    T: int,
    V: int,
    steps: int,
    device: torch.device,
    seed: int,
    remasking: str,
    temperature: float,
    temp_beta: float,
) -> torch.Tensor:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    mask_id = int(V)
    logits_fn = make_oracle_logits_fn(oracle, V=V, temp_beta=temp_beta)

    # no prompt here (pure unconditional)
    prompt = torch.empty((N, 0), device=device, dtype=torch.long)

    x = generate(
        model=None,
        prompt=prompt,
        attention_mask=None,
        steps=int(steps),
        gen_length=int(T),
        block_length=int(T),
        temperature=float(temperature),
        cfg_scale=0.0,
        remasking=str(remasking),
        mask_id=mask_id,
        logits_fn=logits_fn,
        eos_id=0,
        eot_id=0,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        forbid_mask_prediction=True,
    )
    x = x[:, :T]
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=str, default="8,16,32,64,128,256")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--N_eval", type=int, default=-1, help="-1 => use gt.N (recommended override to 128 for large)")
    p.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temp_beta", type=float, default=1.0)

    # NEW: decode for OWT/Stack using your shared tokenizer.json
    p.add_argument(
        "--tokenizer_json",
        type=str,
        default="",
        help="Path to HF `tokenizers` tokenizer.json (for decoding samples into readable text).",
    )

    # qualitative controls
    p.add_argument("--dump_k", type=int, default=20, help="How many sequences to dump into *_samples.txt")
    p.add_argument("--topk_ngrams", type=int, default=80, help="Top-k for unigram/bigram/trigram tables")
    p.add_argument("--run_name", type=str, default="", help="Optional fixed run name. Default auto.")

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

    # build oracle prior
    nbr_idx = gt.nbr_idx.to(device=device, dtype=torch.long)
    nbr_prob = gt.nbr_prob.to(device=device, dtype=torch.float32)
    nu = gt.nu.to(device=device, dtype=torch.float32)
    eps_tp = float(gt.eps)
    pi0 = gt.pi.to(device=device, dtype=torch.float32)

    prior_metrics = MetricsSparseTeleportPrior(nbr_idx=nbr_idx, nbr_prob=nbr_prob, nu=nu, eps=eps_tp)
    prior_oracle = OracleSparseTeleportPrior(nbr_idx=nbr_idx, nbr_prob=nbr_prob, nu=nu, eps=eps_tp)

    oracle = OracleHMMPosterior_LogRank1Teleport(
        prior=prior_oracle,
        pi0=pi0,
        mask_id=V,
        store_dtype=torch.float16,
        compute_dtype=torch.float32,
    ).to(device).eval()

    # output dirs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gt_base = os.path.splitext(os.path.basename(args.gt))[0]
    knobs_tag = _sanitize(
        f"llada_rem{args.remasking}"
        f"_temp{_fmt_float_tag(args.temperature)}"
        f"_beta{_fmt_float_tag(args.temp_beta)}"
        f"_Neval{int(N)}"
        f"_dump{int(args.dump_k)}"
        f"_top{int(args.topk_ngrams)}"
    )
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{gt_base}_{knobs_tag}_seed{int(args.seed)}_{timestamp}"

    out_root = os.path.join("sampler_output", "llada", run_name)
    samples_dir = os.path.join(out_root, "samples")
    _ensure_dir(samples_dir)

    id2tok = _maybe_get_id2tok(gt)

    tok_json = args.tokenizer_json.strip() or None
    hf_tok = _maybe_load_hf_tokenizer_json(tok_json)

    print(f"[GT] path={args.gt}")
    print(f"[GT] V={V} T={T} gt.N={N_gt} eval.N={N}")
    print(f"[CFG] remasking={args.remasking} temperature={args.temperature} temp_beta={args.temp_beta}")
    print(f"[CFG] tokenizer_json={tok_json}")
    print(f"[OUT] {os.path.abspath(samples_dir)}")

    # ---- AR baseline samples
    x_ar = sample_ar_sparse_teleport(
        pi=pi0,
        prior=prior_metrics,
        N=N,
        T=T,
        seed=int(args.seed) + 777,
        device=device,
    )

    ar_samples_txt = os.path.join(samples_dir, "AR_baseline_samples.txt")
    ar_top_txt = os.path.join(samples_dir, "AR_baseline_topngrams.txt")
    _save_samples_txt(
        x=x_ar,
        out_txt=ar_samples_txt,
        id2tok=id2tok,
        hf_tok=hf_tok,
        k=int(args.dump_k),
        title="AR baseline samples",
    )
    _save_topngrams_txt(
        x=x_ar,
        out_txt=ar_top_txt,
        V=V,
        topk=int(args.topk_ngrams),
        id2tok=id2tok,
        title="AR baseline top n-grams",
    )
    print(f"[OK] qualitative samples saved: {ar_samples_txt}")
    print(f"[OK] top ngrams saved       : {ar_top_txt}")

    # ---- steps sweep (LLaDA)
    mask_id = V
    for s in steps_list:
        step_seed = int(args.seed)  # fixed seed is fine for qualitative comparison
        x_ll = _sample_llada(
            oracle=oracle,
            N=N,
            T=T,
            V=V,
            steps=int(s),
            device=device,
            seed=step_seed,
            remasking=str(args.remasking),
            temperature=float(args.temperature),
            temp_beta=float(args.temp_beta),
        )

        # Warn if any MASK remains (important for low steps)
        if (x_ll == mask_id).any():
            num = int((x_ll == mask_id).sum().item())
            print(f"[WARN] step={s:4d} still has MASK ({num} positions). Consider increasing steps.")

        # Save
        tag = f"llada_steps{s:04d}_{args.remasking}"
        samp_txt = os.path.join(samples_dir, f"{tag}_samples.txt")
        top_txt = os.path.join(samples_dir, f"{tag}_topngrams.txt")

        _save_samples_txt(
            x=x_ll,
            out_txt=samp_txt,
            id2tok=id2tok,
            hf_tok=hf_tok,
            k=int(args.dump_k),
            title=f"LLaDA samples | steps={s} remasking={args.remasking} temp={args.temperature} beta={args.temp_beta}",
        )
        _save_topngrams_txt(
            x=x_ll,
            out_txt=top_txt,
            V=V,
            topk=int(args.topk_ngrams),
            id2tok=id2tok,
            title=f"LLaDA top n-grams | steps={s} remasking={args.remasking} temp={args.temperature} beta={args.temp_beta}",
        )
        print(f"[OK] step={s:4d} samples: {samp_txt}")
        print(f"[OK] step={s:4d} ngrams : {top_txt}")


if __name__ == "__main__":
    main()
