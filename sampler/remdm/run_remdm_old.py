#!/usr/bin/env python3
# sampler/remdm/run_remdm.py
#
# Oracle ReMDM sampler using GT oracle p*(x0 | x_t)
# Strictly aligned to the authors' diffusion.py sampling behavior:
#   - remdm-rescale
#   - remdm-conf  (paper default for large scale)
#
# IMPORTANT semantics for oracle:
#   - oracle(x) returns p*(x0 | x_t) over clean tokens (V-dim), with:
#       * masked positions: HMM gamma posterior
#       * unmasked positions: one-hot (fixed tokens)
#   - We treat this as a drop-in replacement for p_x0 = forward(x, sigma_t).exp()
#
# Usage examples:
#   # rescale only
#   CUDA_VISIBLE_DEVICES=2 python -m sampler.remdm.run_remdm \
#     --gt gt_text8_char_T128_N1000_topk27_lam1e-4.pt \
#     --device cuda:0 \
#     --steps "8,16,32,64,128,256" \
#     --sampler rescale \
#     --eta 0.9 \
#     --noise_removal \
#     --seed 123 \
#     --N_eval 128
#
#   # conf (rescale + conf)
#   CUDA_VISIBLE_DEVICES=2 python -m sampler.remdm.run_remdm \
#     --gt gt_text8_char_T128_N1000_topk27_lam1e-4.pt \
#     --device cuda:0 \
#     --steps "8,16,32,64,128,256" \
#     --sampler conf \
#     --eta 0.9 \
#     --noise_removal \
#     --seed 123 \
#     --N_eval 128

from __future__ import annotations

import argparse
import os
import json
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt

# -------------------------------
# GT + metrics (FULL)
# -------------------------------
from sampler.gt_io import load_gt
from sampler.metrics_full import (
    SparseTeleportPrior as MetricsSparseTeleportPrior,
    nll_transition_sparse_teleport,
    full_kl_rate_sparse_teleport,
    unigram_l1,
    unique_ngram_ratio,
    dup_rate,
    top_unigrams_bigrams_print,
)
from sampler.ar_baseline_sparse import sample_ar_sparse_teleport
from sampler.utils_io import parse_steps, _fmt_float_tag, _sanitize

# -------------------------------
# Oracle posterior (HMM hard evidence)
# -------------------------------
from sampler.oracle_hmm_posterior import (
    SparseTeleportPrior,
    OracleHMMPosterior_LogRank1Teleport,
)

# -------------------------------
# ReMDM diffusion update (paper-aligned)  (DO NOT CHANGE)
# -------------------------------
from sampler.remdm.updates_from_diffusion import (
    ReMDMSamplerConfig,
    ddpm_caching_update_from_diffusion,
)

# ============================================================
# Noise schedule (author LogLinearNoise)
#   sigma(t) = -log(1 - (1-eps)*t)
#   where eps=1e-3 in the author's code
# ============================================================
NOISE_EPS = 1e-3


def loglinear_sigma(t: torch.Tensor, *, eps: float = NOISE_EPS) -> torch.Tensor:
    """
    Author's LogLinearNoise.total_noise(t).
    t: shape [B,1] or [B]
    returns sigma with same batch shape.
    """
    if t.ndim == 2 and t.shape[1] == 1:
        t_ = t[:, 0]
    else:
        t_ = t
    return -torch.log1p(-(1.0 - eps) * t_)


# ============================================================
# Dataset tag helper (shared style across all runners)
# ============================================================
def make_ds_tag(meta: dict, V_eff: int) -> str:
    """
    Build a clean dataset tag for plots/tables.

    We distinguish:
      - tokV: tokenizer vocabulary size stored in GT config (often 4096 for BPE)
      - Veff: oracle effective state space size used in evaluation (gt.V)

    Examples:
      Stack-Python (BPE, tokV=4096, Veff=4096)
      OWT (BPE, tokV=4096, Veff=2049)
      text8-char (char, Veff=27)
    """
    if not isinstance(meta, dict):
        return f"Veff={int(V_eff)}"

    ds = str(meta.get("dataset", "")).strip().lower()
    tokV = meta.get("V", None)

    tokV_str = ""
    if tokV is not None:
        try:
            tokV_str = str(int(tokV))
        except Exception:
            tokV_str = str(tokV)

    # dataset display name
    if ds == "owt":
        name = "OWT"
        kind = "BPE"
    elif ds in ["stack_py", "stack", "the_stack", "the-stack", "stack-python", "stack_python"]:
        name = "Stack-Python"
        kind = "BPE"
    elif ds in ["text8", "text8_char", "text8-char", "text8_char_level", "text8char"]:
        name = "text8-char"
        kind = "char"
    elif ds:
        name = ds
        kind = "BPE"
    else:
        name = "dataset"
        kind = "BPE"

    if tokV_str and kind == "BPE":
        return f"{name} ({kind}, tokV={tokV_str}, Veff={int(V_eff)})"
    return f"{name} ({kind}, Veff={int(V_eff)})"


# ============================================================
# Utils
# ============================================================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _plot_curve(
    xs: List[int],
    ys: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
    *,
    ylog: bool = False,
    ar_value: Optional[float] = None,
):
    plt.figure(figsize=(9.2, 4.9))

    # ReMDM
    plt.plot(
        xs,
        ys,
        marker="o",
        linewidth=2.3,
        markersize=6,
        label="ReMDM",
        color="tab:orange",
    )

    # AR
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


# ============================================================
# Oracle ReMDM sampler (paper-aligned)
# ============================================================
@torch.no_grad()
def run_remdm_oracle_steps(
    *,
    oracle: OracleHMMPosterior_LogRank1Teleport,
    steps: int,
    N: int,
    T: int,
    V: int,
    device: torch.device,
    sampler_cfg: ReMDMSamplerConfig,
    eps: float = 1e-5,
    noise_removal: bool = True,
    debug_stats: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Oracle ReMDM sampler.
    x starts from all-mask; each step applies ReMDM update
    using oracle p*(x0 | x_t) as a replacement of forward(x, sigma_t).exp().

    Returns:
      x_final: [N,T]
      dbg: simple stats (mask_frac etc.) for sanity
    """
    mask_index = V
    x = torch.full((N, T), mask_index, dtype=torch.long, device=device)

    # diffusion.py schedule: timesteps from 1 -> eps
    timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
    dt = float((1.0 - eps) / steps)

    # token-wise confidence state (ReMDM-conf only)
    conf = None
    if sampler_cfg.sampler == "remdm-conf":
        conf = torch.full((N, T), -torch.inf, device=device, dtype=torch.float32)

    last_same_frac = 0.0

    for i in range(steps):
        t = timesteps[i] * torch.ones((N, 1), device=device)  # [N,1]
        _sigma_t = loglinear_sigma(t).to(torch.float32)  # [N]
        if not torch.isfinite(_sigma_t).all():
            raise RuntimeError("Non-finite sigma_t encountered in loglinear noise schedule.")

        x_prev = x

        # oracle posterior over clean tokens
        p_clean = oracle(x)  # [N, T, V], sums to 1 over V at every position

        # build p_x0 over V+1 (mask prob = 0)
        p_x0 = torch.zeros((N, T, V + 1), device=device, dtype=torch.float32)
        p_x0[:, :, :V] = p_clean
        p_x0[:, :, mask_index] = 0.0

        x, conf = ddpm_caching_update_from_diffusion(
            x=x,
            t=t,
            dt=dt,
            p_x0=p_x0,
            mask_index=mask_index,
            cfg=sampler_cfg,
            conf=conf,
        )

        if debug_stats:
            last_same_frac = float((x == x_prev).float().mean().item())

    # optional final noise removal (author-aligned, deterministic argmax)
    if noise_removal:
        if (x == mask_index).any():
            p_clean = oracle(x)  # [N,T,V]
            fill = p_clean.argmax(dim=-1)  # [N,T]
            x = torch.where(x == mask_index, fill, x)

    dbg = {
        "mask_frac_final": float((x == mask_index).float().mean().item()),
        "same_frac_last_step": float(last_same_frac),
    }
    return x, dbg


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--seed", type=int, default=123)

    # --- NEW: N_eval override (like your SEDD runner) ---
    parser.add_argument(
        "--N_eval",
        type=int,
        default=128,
        help="Number of sequences used for evaluation (override gt.N).",
    )

    # ReMDM params (paper-aligned)
    parser.add_argument(
        "--sampler",
        type=str,
        default="conf",
        choices=["rescale", "conf"],
        help="rescale: remdm-rescale only; conf: remdm-conf (i.e., rescale + confidence).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.9,
        help="Used by rescale; ignored by conf (as in the authors' remdm-conf).",
    )
    parser.add_argument(
        "--noise_removal",
        action="store_true",
        help="Author-aligned deterministic final denoise (argmax) to remove remaining masks.",
    )

    # output roots
    parser.add_argument("--out_dir", type=str, default="sampler_output")
    parser.add_argument("--plot_dir", type=str, default="sampler_plots")
    parser.add_argument("--run_name", type=str, default="")

    # diagnostics
    parser.add_argument("--sanity_print", action="store_true", help="print top unigrams/bigrams per step (debug)")
    parser.add_argument("--sanity_k", type=int, default=15, help="top-k for sanity print")
    parser.add_argument("--debug_stats", action="store_true", help="print mask_frac and same_frac diagnostics per step")

    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    # IMPORTANT: do not re-seed per step
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------
    gt = load_gt(args.gt, device=str(device))
    V, T = int(gt.V), int(gt.T)
    N_gt = int(gt.N)          # for logging only
    N = int(args.N_eval)      # USE THIS for sampling/eval (override gt.N)

    nbr_idx = gt.nbr_idx.to(device)
    nbr_prob = gt.nbr_prob.to(device)
    nu = gt.nu.to(device)
    eps_tp = float(gt.eps)
    pi0 = gt.pi.to(device)

    K = int(nbr_idx.shape[1])
    mask_index = V

    meta = gt.config if hasattr(gt, "config") and isinstance(gt.config, dict) else {}
    ds_tag = make_ds_tag(meta, V_eff=V)
    tokV_val = None
    if isinstance(meta, dict) and ("V" in meta):
        try:
            tokV_val = int(meta["V"])
        except Exception:
            tokV_val = meta["V"]

    # map CLI sampler -> internal sampler name
    sampler_name = "remdm-conf" if args.sampler == "conf" else "remdm-rescale"

    print(f"[GT] path={args.gt}")
    print(f"[DS] {ds_tag}")
    print(f"[GT] Veff={V}, T={T}, gt.N={N_gt}, eval.N={N}, K={K}, eps={eps_tp:g}")
    print(f"[CFG] sampler={args.sampler} (internal={sampler_name}) eta={args.eta} noise_removal={bool(args.noise_removal)}")
    print(f"[NOISE] LogLinearNoise eps={NOISE_EPS:g} (sigma(t)=-log(1-(1-eps)*t))")

    # --------------------------------------------------------
    # Priors + oracle
    # --------------------------------------------------------
    prior_metrics = MetricsSparseTeleportPrior(nbr_idx, nbr_prob, nu, eps_tp)
    prior_oracle = SparseTeleportPrior(nbr_idx, nbr_prob, nu, eps_tp)

    oracle = OracleHMMPosterior_LogRank1Teleport(
        prior=prior_oracle,
        pi0=pi0,
        mask_id=mask_index,
        store_dtype=torch.float16,
        compute_dtype=torch.float32,
    ).to(device).eval()

    # --------------------------------------------------------
    # Output dirs
    # --------------------------------------------------------
    gt_base = os.path.splitext(os.path.basename(args.gt))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    knobs_tag = _sanitize(
        f"{args.sampler}_eta{_fmt_float_tag(args.eta)}"
        f"_nr{int(bool(args.noise_removal))}"
        f"_Ne{int(N)}"
        f"_K{K}_eps{_fmt_float_tag(eps_tp)}"
    )

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{gt_base}_{knobs_tag}_seed{args.seed}_{timestamp}"

    out_root = os.path.join(args.out_dir, "remdm")
    plot_root = os.path.join(args.plot_dir, "remdm")
    _ensure_dir(out_root)
    _ensure_dir(plot_root)

    run_dir = os.path.join(out_root, run_name)
    run_plot_dir = os.path.join(plot_root, run_name)
    _ensure_dir(run_dir)
    _ensure_dir(run_plot_dir)

    print(f"[OUT] run_dir={os.path.abspath(run_dir)}")
    print(f"[PLOT] plot_dir={os.path.abspath(run_plot_dir)}")

    metrics_json_path = os.path.join(run_dir, "metrics.json")
    metrics_jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")

    # --------------------------------------------------------
    # AR baseline (use eval.N!)
    # --------------------------------------------------------
    x_ar = sample_ar_sparse_teleport(
        pi=pi0,
        prior=prior_metrics,
        N=N,
        T=T,
        seed=args.seed + 777,
        device=device,
    )
    ar_nll = nll_transition_sparse_teleport(x_ar, prior_metrics)
    ar_rt = full_kl_rate_sparse_teleport(x_ar, prior_metrics)
    ar_uni = unigram_l1(x_ar, pi=pi0, V=V)
    ar_u2 = unique_ngram_ratio(x_ar, n=2)
    ar_u3 = unique_ngram_ratio(x_ar, n=3)
    ar_dup = dup_rate(x_ar)

    ar_rec: Dict[str, Any] = {
        "type": "baseline_ar",
        "dataset_tag": ds_tag,
        "tokV": tokV_val,
        "V_eff": int(V),
        "steps": 0,
        "seed": int(args.seed + 777),
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

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    header: Dict[str, Any] = {
        "type": "header",
        "gt_path": args.gt,
        "device": str(device),
        "seed": int(args.seed),
        "dataset_tag": ds_tag,
        "tokV": tokV_val,
        "V_eff": int(V),
        "V": int(V),  # kept for backward compatibility with older scripts
        "T": int(T),
        "gt_N": int(N_gt),
        "N_eval": int(N),
        "K": int(K),
        "eps": float(eps_tp),
        "remdm": {
            "sampler_cli": args.sampler,           # rescale | conf
            "sampler_internal": sampler_name,      # remdm-rescale | remdm-conf
            "eta": float(args.eta),
            "noise_removal": bool(args.noise_removal),
            "noise_schedule": {"type": "loglinear", "eps": float(NOISE_EPS)},
            "notes": (
                "Oracle provides p*(x0|xt) as a replacement of forward(x, sigma_t).exp(); "
                "sigma(t) computed via author LogLinearNoise for alignment."
            ),
        },
        "gt_meta": meta,
        "ar_baseline": ar_rec,
    }

    with open(metrics_jsonl_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        f.write(json.dumps(ar_rec) + "\n")

    print("\n[AR baseline]")
    print(
        f"  AR | NLL/token={ar_nll:.6f} | fKL={ar_rt['full_kl_rate']:.3e} "
        f"| fTV={ar_rt['full_tv_rate']:.3e} | fH={ar_rt['full_entropy_rate']:.3f} "
        f"| uniL1={ar_uni:.3e} | u2={ar_u2:.4f} u3={ar_u3:.4f} | dup={ar_dup:.4f} "
        f"| other={ar_rt['other_mass_rate']:.4f} | supp={ar_rt['support_frac']:.4f}"
    )

    # --------------------------------------------------------
    # Sampler config (paper-aligned core; conf is a distinct branch)
    # --------------------------------------------------------
    sampler_cfg = ReMDMSamplerConfig(
        sampler=sampler_name,
        eta=float(args.eta),  # used by remdm-rescale; ignored by remdm-conf
        t_on=1.0,
        t_off=0.0,
        alpha_on=0.0,
    )

    # --------------------------------------------------------
    # Run steps
    # --------------------------------------------------------
    steps_list = parse_steps(args.steps)
    if not steps_list:
        raise ValueError("Empty --steps")
    print(f"\n[ReMDM] steps sweep: {steps_list}")

    rows: List[Dict[str, Any]] = []
    vocab = gt.vocab if hasattr(gt, "vocab") and isinstance(gt.vocab, list) else None

    for s in steps_list:
        x, dbg = run_remdm_oracle_steps(
            oracle=oracle,
            steps=int(s),
            N=N,          # USE eval.N
            T=T,
            V=V,
            device=device,
            sampler_cfg=sampler_cfg,
            noise_removal=bool(args.noise_removal),
            debug_stats=bool(args.debug_stats),
        )

        nll_tok = nll_transition_sparse_teleport(x, prior_metrics)
        rt = full_kl_rate_sparse_teleport(x, prior_metrics)
        uni = unigram_l1(x, pi=pi0, V=V)
        u2 = unique_ngram_ratio(x, n=2)
        u3 = unique_ngram_ratio(x, n=3)
        dr = dup_rate(x)

        rec: Dict[str, Any] = {
            "type": "step",
            "dataset_tag": ds_tag,
            "tokV": tokV_val,
            "V_eff": int(V),
            "steps": int(s),
            "seed": int(args.seed),
            "N_eval": int(N),

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
            "dbg_mask_frac_final": float(dbg["mask_frac_final"]),
            "dbg_same_frac_last_step": float(dbg["same_frac_last_step"]),
        }
        rows.append(rec)

        with open(metrics_jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        extra = ""
        if args.debug_stats:
            extra = f" | maskFrac={dbg['mask_frac_final']:.4f} sameFrac(last)={dbg['same_frac_last_step']:.4f}"

        print(
            f"  step={s:4d} | NLL/token={nll_tok:.6f} | fKL={rt['full_kl_rate']:.3e} "
            f"| fTV={rt['full_tv_rate']:.3e} | fH={rt['full_entropy_rate']:.3f} "
            f"| uniL1={uni:.3e} | u2={u2:.4f} u3={u3:.4f} | dup={dr:.4f} "
            f"| other={rt['other_mass_rate']:.4f} | supp={rt['support_frac']:.4f}{extra}"
        )

        if args.sanity_print:
            top_unigrams_bigrams_print(x, V=V, k=args.sanity_k, vocab=vocab)

    # --------------------------------------------------------
    # Save summary + CSV
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Plots (titles include dataset tag + eval.N)
    # --------------------------------------------------------
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

    title_suffix = f"T={T} N={N} K={K}"

    _plot("nll_token",         f"{ds_tag} | NLL/token under P' | {title_suffix}",        ylog=False, ar_value=ar_rec["nll_token"])
    _plot("full_kl_rate",      f"{ds_tag} | FULL KL-rate | {title_suffix}",              ylog=True,  ar_value=ar_rec["full_kl_rate"])
    _plot("full_tv_rate",      f"{ds_tag} | FULL TV-rate | {title_suffix}",              ylog=False, ar_value=ar_rec["full_tv_rate"])
    _plot("full_entropy_rate", f"{ds_tag} | FULL entropy-rate | {title_suffix}",         ylog=False, ar_value=ar_rec["full_entropy_rate"])
    _plot("support_frac",      f"{ds_tag} | support fraction | T={T} N={N}",             ylog=True,  ar_value=ar_rec["support_frac"])

    _plot("unigram_L1",        f"{ds_tag} | unigram L1 vs pi | T={T} N={N}",             ylog=True,  ar_value=ar_rec["unigram_L1"])
    _plot("unique_2gram_ratio", f"{ds_tag} | unique 2-gram ratio | T={T} N={N}",         ylog=False, ar_value=ar_rec["unique_2gram_ratio"])
    _plot("unique_3gram_ratio", f"{ds_tag} | unique 3-gram ratio | T={T} N={N}",         ylog=False, ar_value=ar_rec["unique_3gram_ratio"])
    _plot("dup_rate",          f"{ds_tag} | duplicate sequence rate | T={T} N={N}",      ylog=False, ar_value=ar_rec["dup_rate"])
    _plot("other_mass_rate",   f"{ds_tag} | OTHER-mass rate | {title_suffix}",           ylog=False, ar_value=ar_rec["other_mass_rate"])


if __name__ == "__main__":
    main()
