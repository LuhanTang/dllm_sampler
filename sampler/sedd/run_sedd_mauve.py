#!/usr/bin/env python3
# sampler/run_sedd_mauve.py
# ------------------------------------------------------------
# Compute MAUVE only (plus AR baseline) for SEDD under oracle posterior.
#
# "Real data" (P)  : samples from the ground-truth Markov chain P' (AR sampler)
# "Generated" (Q)  : samples from SEDD (oracle-driven) at each step count
#
# Outputs:
#   - metrics.json
#   - metrics.jsonl
#   - metrics.csv
#   - mauve_vs_steps_*.png
#
# Notes:
# - Requires `mauve-text` package:
#     pip install mauve-text
# - MAUVE is computed in embedding space of a pretrained LM (default: gpt2-large).
# ------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch

# --- Local modules
from sampler.gt_io import load_gt

# --- AR baseline sparse (this is our ground-truth sampler for P')
from sampler.ar_baseline_sparse import sample_ar_sparse_teleport

# --- SEDD bits
from sampler.sedd.graph_lib import Absorbing
from sampler.sedd.noise_lib import LogLinearNoise
from sampler.sedd.sampling import get_pc_sampler

# --- Metrics: only need SparseTeleportPrior type
from sampler.metrics_full import SparseTeleportPrior

# --- utils_io
from sampler.utils_io import parse_steps, _fmt_float_tag, _sanitize


# =========================================================
# Numerics
# =========================================================
NORM_CLAMP = 1e-30
MASKED_SCORE_MODE = "ratio"  # "ratio" or "posterior"
NINF = -1e30  # practical -inf (fp16-safe)


# =========================================================
# Utility: dataset tag
# =========================================================
def make_ds_tag(meta: dict, V_eff: int) -> str:
    if not isinstance(meta, dict):
        return f"Veff={int(V_eff)}"

    ds = str(meta.get("dataset", "")).strip().lower()
    tok = str(meta.get("tokenizer", "")).strip().lower()
    tokV = meta.get("V", None)

    tokV_str = ""
    if tokV is not None:
        try:
            tokV_str = str(int(tokV))
        except Exception:
            tokV_str = str(tokV)

    if ds in ("text8", "text8_char", "text8-char", "text8char"):
        name = "text8-char"
        tok = "char" if not tok else tok
    elif ds == "owt":
        name = "OWT"
        tok = "bpe" if not tok else tok
    elif ds in ["stack_py", "stack", "the_stack", "the-stack", "stack-python", "stack_python"]:
        name = "Stack-Python"
        tok = "bpe" if not tok else tok
    elif ds:
        name = ds
    else:
        name = "dataset"

    if tok == "char":
        return f"{name} (char, Veff={int(V_eff)})"

    if tokV_str:
        return f"{name} (BPE, tokV={tokV_str}, Veff={int(V_eff)})"
    return f"{name} (BPE, Veff={int(V_eff)})"


def _ensure_distribution(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(0.0)
    return p / p.sum().clamp_min(NORM_CLAMP)


# =========================================================
# Text reconstruction for MAUVE
# =========================================================
def _infer_join_mode(vocab: List[str]) -> str:
    if any(("Ġ" in t) for t in vocab):
        return "gpt2_marker"
    if any(("▁" in t) for t in vocab):
        return "sp_marker"
    return "concat"


def detokenize_ids(
    x: torch.Tensor,
    vocab: Optional[List[str]],
    join_mode: str = "auto",
) -> List[str]:
    if vocab is None:
        raise ValueError("MAUVE requires gt.vocab to exist")

    mode = join_mode.lower()
    if mode == "auto":
        mode = _infer_join_mode(vocab)

    texts: List[str] = []
    for seq in x.cpu().tolist():
        toks = [vocab[i] for i in seq]
        if mode in ("char", "concat"):
            texts.append("".join(toks))
        elif mode == "space":
            texts.append(" ".join(toks))
        elif mode == "gpt2_marker":
            texts.append("".join(t.replace("Ġ", " ") for t in toks))
        elif mode == "sp_marker":
            texts.append("".join(t.replace("▁", " ") for t in toks))
        else:
            raise ValueError(f"Unknown join_mode={join_mode}")
    return texts


def compute_mauve_score(
    *,
    p_text: List[str],
    q_text: List[str],
    device: torch.device,
    featurize_model_name: str = "gpt2-large",
    num_buckets: int = 25,
    max_text_length: int = 256,
    seed: int = 0,
    verbose: bool = False,
) -> float:
    import mauve

    device_id = -1
    if device.type == "cuda":
        device_id = device.index or 0

    out = mauve.compute_mauve(
        p_text=p_text,
        q_text=q_text,
        device_id=device_id,
        featurize_model_name=featurize_model_name,
        num_buckets=num_buckets,
        max_text_length=max_text_length,
        seed=seed,
        verbose=verbose,
    )
    return float(out.mauve)


# =========================================================
# Adapters (to tolerate different internal APIs)
# =========================================================
def _get_oracle_fn_from_gt(gt: Any):
    """
    Returns a callable oracle_fn(x_t, t_index) that should output oracle log-probs / scores.

    We try common method names. If your project uses a different name,
    edit this function to point to your actual oracle posterior function.
    """
    # Most explicit names (recommended)
    for name in [
        "oracle_logp_x0_given_xt",
        "oracle_logp",
        "oracle_logprob",
        "oracle_posterior_logp",
        "oracle_posterior_logprob",
        "oracle_score",
        "oracle_score_fn",
    ]:
        if hasattr(gt, name) and callable(getattr(gt, name)):
            fn = getattr(gt, name)
            return lambda x_t, t_index: fn(x_t, t_index)

    raise RuntimeError(
        "[ERROR] Cannot find oracle posterior/score function on the loaded GT object.\n"
        "I looked for method names:\n"
        "  oracle_logp_x0_given_xt / oracle_logp / oracle_posterior_logp / oracle_score(_fn)\n\n"
        "Fix: edit _get_oracle_fn_from_gt() to call your project's oracle function.\n"
        "Tip: search your repo for 'oracle' or 'forward_backward' or 'gamma_tu'."
    )


def _call_pc_sampler(pc_sampler, **kwargs):
    """
    Tries multiple common call conventions for pc_sampler.
    Returns x: Tensor [N, T].
    """
    # Common conventions to try
    attempts = []

    # 1) keyword-based with score_fn
    attempts.append(("score_fn_kw", lambda: pc_sampler(**kwargs)))

    # 2) positional first arg = score_fn
    if "score_fn" in kwargs:
        sf = kwargs["score_fn"]
        rest = {k: v for k, v in kwargs.items() if k != "score_fn"}
        attempts.append(("score_fn_pos0", lambda: pc_sampler(sf, **rest)))

    # 3) some codebases use model=score_fn
    if "score_fn" in kwargs and "model" not in kwargs:
        alt = dict(kwargs)
        alt["model"] = alt.pop("score_fn")
        attempts.append(("model_kw", lambda: pc_sampler(**alt)))

    last_err = None
    for tag, fn in attempts:
        try:
            out = fn()
            if isinstance(out, torch.Tensor):
                return out
            return torch.as_tensor(out)
        except Exception as e:
            last_err = (tag, e)

    tag, e = last_err
    raise RuntimeError(
        f"[ERROR] Failed to call pc_sampler (last attempt='{tag}').\n"
        f"Exception: {repr(e)}\n\n"
        "Fix: open sampler/sedd/sampling.py and check get_pc_sampler()'s returned function signature.\n"
        "Then adjust the kwargs in _call_pc_sampler(...) usage inside main()."
    )


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    # IMPORTANT: GT is IMPORT-ONLY (no on-the-fly generation)
    parser.add_argument(
        "--gt",
        type=str,
        default="gt_text8_char_withSpace_T1024_N128_topk27_eps0.pt",
        help="Path to precomputed GT .pt file (required; no GT generation).",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--N_eval", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="sampler_output")
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument(
        "--accuracy",
        type=str,
        default="accurate",
        choices=["accurate", "inaccurate"],
    )
    parser.add_argument("--temp_beta", type=float, default=1.0)
    parser.add_argument("--truncate_eps", type=float, default=0.0)
    parser.add_argument("--topm_trunc", type=int, default=0)
    parser.add_argument("--posterior_quant", type=str, default="none")

    parser.add_argument("--mauve_model", type=str, default="gpt2-large")
    parser.add_argument("--mauve_num_buckets", type=int, default=25)
    parser.add_argument("--mauve_max_text_length", type=int, default=256)
    parser.add_argument("--mauve_verbose", action="store_true")

    parser.add_argument(
        "--text_join",
        type=str,
        default="auto",
        choices=["auto", "char", "concat", "space", "gpt2_marker", "sp_marker"],
    )

    args = parser.parse_args()

    # hard check: GT must exist
    if not os.path.isfile(args.gt):
        raise FileNotFoundError(
            f"[ERROR] GT file not found: {args.gt}\n"
            "This script does NOT generate GT. Please provide a valid gt_*.pt."
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    if args.accuracy == "accurate":
        args.temp_beta = 1.0
        args.truncate_eps = 0.0
        args.topm_trunc = 0
        args.posterior_quant = "none"

    # =========================================================
    # Load GT (IMPORT ONLY)
    # =========================================================
    gt = load_gt(args.gt, device=str(device))

    V = int(gt.V)
    T = int(gt.T)
    K = int(gt.nbr_idx.shape[1])
    N = int(args.N_eval)

    pi0 = gt.pi.to(device)
    prior = SparseTeleportPrior(
        nbr_idx=gt.nbr_idx.to(device),
        nbr_prob=gt.nbr_prob.to(device),
        nu=gt.nu.to(device),
        eps=float(gt.eps),
    )

    vocab = gt.vocab if hasattr(gt, "vocab") else None
    meta = gt.config if hasattr(gt, "config") else {}
    ds_tag = make_ds_tag(meta, V_eff=V)

    print(f"[GT] loaded from {args.gt}")
    print(f"[DS] {ds_tag} | V={V} T={T} K={K}")

    # =========================================================
    # Reference AR samples (P_text): sample from GT Markov chain P'
    # =========================================================
    x_real = sample_ar_sparse_teleport(
        pi=pi0, prior=prior, N=N, T=T, seed=args.seed + 777, device=device
    )
    p_text = detokenize_ids(x_real, vocab, args.text_join)
    print(f"[P] AR reference sampled: {len(p_text)} texts")

    # =========================================================
    # Prepare output directory
    # =========================================================
    steps_list = parse_steps(args.steps)

    acc_tag = _sanitize(args.accuracy)
    beta_tag = _fmt_float_tag(args.temp_beta)
    tr_tag = _fmt_float_tag(args.truncate_eps)
    topm_tag = str(int(args.topm_trunc))
    q_tag = _sanitize(args.posterior_quant)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = []
    if args.run_name:
        parts.append(_sanitize(args.run_name))
    parts += [
        "sedd_mauve",
        _sanitize(meta.get("dataset", "dataset")) if isinstance(meta, dict) else "dataset",
        f"V{V}_T{T}",
        f"{acc_tag}_beta{beta_tag}_tr{tr_tag}_topm{topm_tag}_q{q_tag}",
        f"seed{int(args.seed)}",
        ts,
    ]
    run_tag = "_".join([p for p in parts if p])
    out_dir = os.path.join(args.out_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[OUT] {out_dir}")

    # =========================================================
    # Build SEDD sampler
    # =========================================================
    graph = Absorbing(V)
    noise = LogLinearNoise()
    pc_sampler = get_pc_sampler(graph=graph, noise=noise)

    # Oracle score adapter (you may need to adjust 1 line if your method name differs)
    oracle_fn = _get_oracle_fn_from_gt(gt)

    # =========================================================
    # Run SEDD @ each step, compute MAUVE
    # =========================================================
    records: List[Dict[str, Any]] = []

    for S in steps_list:
        print(f"[RUN] oracle-SEDD sampling: steps={S} N={N} T={T}")

        # We pass a superset of args; if your pc_sampler ignores extras, it's fine.
        # If your signature is strict, adjust the kwargs below to match your implementation.
        x_sedd = _call_pc_sampler(
            pc_sampler,
            score_fn=oracle_fn,
            prior=prior,
            pi0=pi0,
            N=N,
            T=T,
            steps=int(S),
            seed=int(args.seed),
            device=device,
            temp_beta=float(args.temp_beta),
            truncate_eps=float(args.truncate_eps),
            topm_trunc=int(args.topm_trunc),
            posterior_quant=str(args.posterior_quant),
            masked_score_mode=MASKED_SCORE_MODE,
            ninf=NINF,
        ).to(device)

        q_text = detokenize_ids(x_sedd, vocab, args.text_join)

        mauve_val = compute_mauve_score(
            p_text=p_text,
            q_text=q_text,
            device=device,
            featurize_model_name=args.mauve_model,
            num_buckets=int(args.mauve_num_buckets),
            max_text_length=int(args.mauve_max_text_length),
            seed=int(args.seed),
            verbose=bool(args.mauve_verbose),
        )

        rec = {
            "dataset": ds_tag,
            "gt": os.path.basename(args.gt),
            "method": "oracle_sedd",
            "accuracy": args.accuracy,
            "temp_beta": float(args.temp_beta),
            "truncate_eps": float(args.truncate_eps),
            "topm_trunc": int(args.topm_trunc),
            "posterior_quant": str(args.posterior_quant),
            "steps": int(S),
            "N_eval": int(N),
            "T": int(T),
            "mauve": float(mauve_val),
            "mauve_model": str(args.mauve_model),
            "mauve_num_buckets": int(args.mauve_num_buckets),
            "mauve_max_text_length": int(args.mauve_max_text_length),
            "text_join": str(args.text_join),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        records.append(rec)
        print(f"[MAUVE] steps={S} mauve={mauve_val:.4f}")

    # =========================================================
    # Save metrics
    # =========================================================
    metrics_json = {
        "run_tag": run_tag,
        "out_dir": out_dir,
        "gt": args.gt,
        "dataset": ds_tag,
        "records": records,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    with open(os.path.join(out_dir, "metrics.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # =========================================================
    # Plot
    # =========================================================
    xs = [r["steps"] for r in records]
    ys = [r["mauve"] for r in records]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    # steps often are powers of 2
    try:
        plt.xscale("log", base=2)
    except TypeError:
        plt.xscale("log")  # older matplotlib
    plt.xlabel("Steps")
    plt.ylabel("MAUVE")
    plt.title(f"MAUVE vs steps | {ds_tag} | {args.accuracy} | beta={args.temp_beta}")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"mauve_vs_steps_{acc_tag}_beta{beta_tag}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[DONE] Saved metrics to: {out_dir}")
    print(f"[DONE] Plot saved to: {fig_path}")


if __name__ == "__main__":
    main()