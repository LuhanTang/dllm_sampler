#!/usr/bin/env python3
# sampler/remdm/run_remdm.py
#
# Oracle ReMDM sampler using GT oracle p*(x0 | x_t)
# Now aligned to the authors' *diffusion.py* sampling branches:
#   - remdm-conf
#   - remdm-loop
# and optionally applies nucleus/top-p to the oracle p*(x0|xt).
#


from __future__ import annotations

import argparse
import os
import json
import csv
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tokenizers import Tokenizer

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

# ============================================================
# Noise schedule 
#   sigma(t) = -log(1 - (1-eps)*t), eps=1e-3 i

# ============================================================
NOISE_EPS = 1e-3


def loglinear_sigma(t: torch.Tensor, *, eps: float = NOISE_EPS) -> torch.Tensor:
    if t.ndim == 2 and t.shape[1] == 1:
        t_ = t[:, 0]
    else:
        t_ = t
    return -torch.log1p(-(1.0 - eps) * t_)


# ============================================================
# Helpers: sampling + nucleus
# ============================================================
def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    """
     gumbel-max categorical sampling:
      categorical_probs: [..., K] with probs that sum to 1 on last dim
    returns argmax indices on last dim
    """
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@torch.no_grad()
def apply_nucleus_probs(p: torch.Tensor, nucleus_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to a probability tensor over last dim.
    p: [..., V] probs (assumed non-negative; will be renormalized)
    nucleus_p: in (0,1]; if >=1, returns p unchanged.
    """
    if nucleus_p >= 1.0:
        return p
    # sort descending
    sorted_probs, sorted_indices = torch.sort(p, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative <= nucleus_p
    # ensure at least 1 token kept
    top_p_mask[..., 0] = True
    nucleus_probs = sorted_probs * top_p_mask
    denom = nucleus_probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    nucleus_probs = nucleus_probs / denom
    # scatter back
    out = torch.zeros_like(p)
    out.scatter_(-1, sorted_indices, nucleus_probs)
    return out


# ============================================================
# Dataset tag helper (shared style across all runners)
# ============================================================
def make_ds_tag(meta: dict, V_eff: int) -> str:
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

    if ds == "owt":
        name, kind = "OWT", "BPE"
    elif ds in ["stack_py", "stack", "the_stack", "the-stack", "stack-python", "stack_python"]:
        name, kind = "Stack-Python", "BPE"
    elif ds in ["text8", "text8_char", "text8-char", "text8_char_level", "text8char"]:
        name, kind = "text8-char", "char"
    elif ds:
        name, kind = ds, "BPE"
    else:
        name, kind = "dataset", "BPE"

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
    plt.plot(xs, ys, marker="o", linewidth=2.3, markersize=6, label="ReMDM", color="tab:orange")
    if ar_value is not None:
        plt.axhline(ar_value, linestyle="--", linewidth=2.0, label="AR baseline", color="tab:blue")
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
# caching update (subset we need)
#   Implements:
#     - remdm-conf
#     - remdm-loop
#   and expects p_x0 already computed (and nucleus-applied if desired).
# ============================================================
@torch.no_grad()
def ddpm_caching_update_author_like(
    *,
    x: torch.Tensor,              # [N,T]
    t: torch.Tensor,              # [N,1]
    dt: float,
    p_x0: torch.Tensor,           # [N,T,V+1] over clean+mask_index
    mask_index: int,
    sampler: str,                 # "remdm_conf" | "remdm_loop"
    conf: Optional[torch.Tensor], # [N,T] for remdm_conf
    eta: float,
    t_on: float,
    t_off: float,
    alpha_on: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      xs: new tokens [N,T]
      conf: updated conf (for remdm_conf), else unchanged None
    """
 
    # Shapes: [N,1,1]
    if t.ndim > 1:
        t1 = t.squeeze(-1)
    else:
        t1 = t
    assert t1.ndim == 1

    move_chance_t = t1[:, None, None]
    move_chance_s = (t1 - dt)[:, None, None]

    if sampler == "remdm_conf":
   
        # compute alpha_t/alpha_s and sigma_max
        alpha_t = (1 - move_chance_t)[0].item()
        alpha_s = (1 - move_chance_s)[0].item()
        if alpha_t > 0:
            sigma_max = min(1.0, (1 - alpha_s) / alpha_t)
        else:
            sigma_max = 1.0

        if conf is None:
            conf = torch.full_like(x, -torch.inf, dtype=torch.float32)

        eta_vec = conf.softmax(dim=-1)  # [N,T]
        masked_flag = (x == mask_index)
        eta_vec[masked_flag] = 0.0
        sigma = eta_vec * sigma_max  # [N,T]

        q_xs = p_x0 * (1 - sigma[:, :, None])
        q_xs[..., mask_index] = sigma
        q_xs_2 = p_x0 * ((alpha_s - (1 - sigma[:, :, None]) * alpha_t) / (1 - alpha_t))
        q_xs_2[..., mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)

        copy_flag = (x != mask_index).to(torch.bool)
        q = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q)

        # update conf 
        unmask_mask = (x == mask_index) & (xs != mask_index)
        batch_indices = torch.arange(xs.shape[0], device=xs.device)[:, None]
        feature_indices = torch.arange(xs.shape[1], device=xs.device)[None, :]
        xs_clamped = xs.clamp(0, mask_index)
        conf_values = -p_x0[batch_indices, feature_indices, xs_clamped]
        conf = conf.to(conf_values.dtype)
        conf[unmask_mask] = conf_values[unmask_mask]

        remask_mask = (x != mask_index) & (xs == mask_index)
        conf[remask_mask] = -torch.inf
        return xs, conf

    if sampler == "remdm_loop":
        # ===  remdm-loop branch ===
        # time is scalar from first element
        time = float(t1[0].item())
        sigma = torch.tensor(float(eta), device=x.device, dtype=p_x0.dtype)

        # piecewise redefinition of move_chance for outer segments
        # (copied from your pasted diffusion.py)
        if time > t_on:
            move_chance_t2 = (1 - (1 - t1) * alpha_on / (1 - t_on))[:, None, None]
            move_chance_s2 = (1 - (1 - t1 + dt) * alpha_on / (1 - t_on))[:, None, None]
        elif time <= t_off:
            move_chance_t2 = (t1 * (1 - alpha_on) / t_off)[:, None, None]
            move_chance_s2 = ((t1 - dt) * (1 - alpha_on) / t_off)[:, None, None]
        else:
            move_chance_t2, move_chance_s2 = None, None

        # outer segments: use MDLM update
        if time > t_on or time <= t_off:
            # MDLM: q_xs = p_x0*(move_chance_t - move_chance_s), mask prob = move_chance_s
            q_xs = p_x0 * (move_chance_t2 - move_chance_s2)
            q_xs[:, :, mask_index] = move_chance_s2[:, :, 0]
            _x = _sample_categorical(q_xs)
            copy_flag = (x != mask_index).to(x.dtype)
            xs = copy_flag * x + (1 - copy_flag) * _x
            return xs, conf

        # middle segment: use ReMDM with fixed sigma=eta 
        
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., mask_index] = sigma
        q_xs_2 = p_x0 * ((alpha_on - (1 - sigma) * alpha_on) / (1 - alpha_on))
        q_xs_2[..., mask_index] = (1 - alpha_on - alpha_on * sigma) / (1 - alpha_on)

        copy_flag = (x != mask_index).to(torch.bool)
        q = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q)
        return xs, conf

    raise ValueError(f"Unknown sampler={sampler!r}")


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
    sampler: str,          # "remdm_conf" | "remdm_loop"
    nucleus_p: float,
    eta: float,
    t_on: float,
    t_off: float,
    alpha_on: float,
    eps: float = 1e-5,
    noise_removal: bool = True,
    debug_stats: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask_index = V
    x = torch.full((N, T), mask_index, dtype=torch.long, device=device)

    timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
    dt = float((1.0 - eps) / steps)

    conf: Optional[torch.Tensor] = None
    if sampler == "remdm_conf":
        conf = torch.full((N, T), -torch.inf, device=device, dtype=torch.float32)

    last_same_frac = 0.0

    for i in range(steps):
        t = timesteps[i] * torch.ones((N, 1), device=device)  # [N,1]

        # keep for sanity: sigma finite
        _sigma_t = loglinear_sigma(t).to(torch.float32)
        if not torch.isfinite(_sigma_t).all():
            raise RuntimeError("Non-finite sigma_t encountered in loglinear noise schedule.")

        x_prev = x

        # oracle posterior over clean tokens
        p_clean = oracle(x)  # [N,T,V], sums to 1
        if nucleus_p < 1.0:
            p_clean = apply_nucleus_probs(p_clean, nucleus_p)

        # build p_x0 over V+1 (mask prob = 0)
        p_x0 = torch.zeros((N, T, V + 1), device=device, dtype=torch.float32)
        p_x0[:, :, :V] = p_clean
        p_x0[:, :, mask_index] = 0.0

        x, conf = ddpm_caching_update_author_like(
            x=x,
            t=t,
            dt=dt,
            p_x0=p_x0,
            mask_index=mask_index,
            sampler=sampler,
            conf=conf,
            eta=float(eta),
            t_on=float(t_on),
            t_off=float(t_off),
            alpha_on=float(alpha_on),
        )

        if debug_stats:
            last_same_frac = float((x == x_prev).float().mean().item())

    # optional final noise removal 
    if noise_removal and (x == mask_index).any():
        p_clean = oracle(x)  # [N,T,V]
        if nucleus_p < 1.0:
            p_clean = apply_nucleus_probs(p_clean, nucleus_p)
        fill = p_clean.argmax(dim=-1)
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

    parser.add_argument(
        "--N_chunk",
        type=int,
        default=0,
        help="If >0, generate in chunks of this size to reduce VRAM peak. 0 disables chunking.",
    )


    parser.add_argument(
        "--N_eval",
        type=int,
        default=128,
        help="Number of sequences used for evaluation (override gt.N).",
    )

    # === NEW: sampler choices ===
    parser.add_argument(
        "--sampler",
        type=str,
        default="remdm_conf",
        choices=["remdm_conf", "remdm_loop"],
        help="sampler branches: remdm_conf or remdm_loop.",
    )

    # === NEW: nucleus/top-p ===
    parser.add_argument(
        "--nucleus_p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling on p*(x0|xt). 1.0 disables; e.g. 0.9 enables.",
    )

    # loop params (used only if --sampler remdm_loop)
    parser.add_argument("--eta", type=float, default=0.9, help="Loop middle-segment fixed sigma (authors' eta).")
    parser.add_argument("--t_on", type=float, default=1.0, help="Loop: switch-on time threshold (authors' t_on).")
    parser.add_argument("--t_off", type=float, default=0.0, help="Loop: switch-off time threshold (authors' t_off).")
    parser.add_argument("--alpha_on", type=float, default=0.0, help="Loop: alpha_on (authors' alpha_on).")

    parser.add_argument(
        "--noise_removal",
        action="store_true",
        help="Author-aligned deterministic final denoise (argmax) to remove remaining masks.",
    )

    # -------------------------------
    # Sample dumping (decode for OWT)
    # -------------------------------
    parser.add_argument(
        "--dump_samples",
        action="store_true",
        help="If set, decode and dump generated samples to text files (OWT only).",
    )
    parser.add_argument(
        "--tokenizer_json",
        type=str,
        default="tokenizers/owt_bytebpe_v4096/tokenizer.json",
        help="Tokenizer JSON path used to decode OWT token ids.",
    )
    parser.add_argument(
        "--dump_steps",
        type=str,
        default="",
        help="Comma-separated steps to dump samples for. Empty => dump for all steps in --steps.",
    )
    parser.add_argument(
        "--dump_limit",
        type=int,
        default=0,
        help="If >0, only dump first dump_limit samples (for quick debugging). 0 => dump all N.",
    )
    parser.add_argument(
        "--dump_prefix",
        type=str,
        default="samples",
        help="Filename prefix for dumped sample files.",
    )

    parser.add_argument(
        "--dump_ids",
        action="store_true",
        help="If set, save token ids (x) as a .pt file for each dumped step.",
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

    if not (0.0 < args.nucleus_p <= 1.0):
        raise ValueError("--nucleus_p must be in (0,1].")

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

    print(f"[GT] path={args.gt}")
    print(f"[DS] {ds_tag}")
    print(f"[GT] Veff={V}, T={T}, gt.N={N_gt}, eval.N={N}, K={K}, eps={eps_tp:g}")
    print(
        f"[CFG] sampler={args.sampler} nucleus_p={args.nucleus_p:g} "
        f"eta={args.eta} t_on={args.t_on} t_off={args.t_off} alpha_on={args.alpha_on} "
        f"noise_removal={bool(args.noise_removal)}"
    )
    print(f"[NOISE] LogLinearNoise eps={NOISE_EPS:g} (sigma(t)=-log(1-(1-eps)*t))")

    # --------------------------------------------------------
    # Optional tokenizer for OWT decoding
    # --------------------------------------------------------
    ds_name = str(meta.get("dataset", "")).strip().lower() if isinstance(meta, dict) else ""
    is_owt = (ds_name == "owt")

    tok = None
    if args.dump_samples:
        if not is_owt:
            print("[WARN] --dump_samples is set but dataset is not OWT; will not decode/dump.")
        else:
            if not os.path.isfile(args.tokenizer_json):
                raise FileNotFoundError(f"Tokenizer JSON not found: {args.tokenizer_json}")
            tok = Tokenizer.from_file(args.tokenizer_json)
            print(f"[TOK] Loaded tokenizer: {args.tokenizer_json}")



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
        f"{args.sampler}"
        f"_p{_fmt_float_tag(args.nucleus_p)}"
        f"_eta{_fmt_float_tag(args.eta)}"
        f"_ton{_fmt_float_tag(args.t_on)}"
        f"_toff{_fmt_float_tag(args.t_off)}"
        f"_aon{_fmt_float_tag(args.alpha_on)}"
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

    header: Dict[str, Any] = {
        "type": "header",
        "gt_path": args.gt,
        "device": str(device),
        "seed": int(args.seed),
        "dataset_tag": ds_tag,
        "tokV": tokV_val,
        "V_eff": int(V),
        "V": int(V),
        "T": int(T),
        "gt_N": int(N_gt),
        "N_eval": int(N),
        "K": int(K),
        "eps": float(eps_tp),
        "remdm": {
            "sampler": args.sampler,                 # remdm_conf | remdm_loop
            "nucleus_p": float(args.nucleus_p),
            "eta": float(args.eta),
            "t_on": float(args.t_on),
            "t_off": float(args.t_off),
            "alpha_on": float(args.alpha_on),
            "noise_removal": bool(args.noise_removal),
            "noise_schedule": {"type": "loglinear", "eps": float(NOISE_EPS)},
            "notes": (
                "Oracle provides p*(x0|xt) as a replacement of forward(x, sigma_t).exp(). "
                "Update branches follow authors' diffusion.py (remdm-conf/remdm-loop). "
                "Optional nucleus is applied on p*(x0|xt) before the update."
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
    # Run steps
    # --------------------------------------------------------
    steps_list = parse_steps(args.steps)
    if not steps_list:
        raise ValueError("Empty --steps")
    print(f"\n[ReMDM] steps sweep: {steps_list}")

    dump_steps = None
    if (args.dump_samples and is_owt) or args.dump_ids:
        if str(args.dump_steps).strip():
            dump_steps = set(parse_steps(args.dump_steps))
        else:
            dump_steps = set(steps_list) # default: dump all steps


    rows: List[Dict[str, Any]] = []
    vocab = gt.vocab if hasattr(gt, "vocab") and isinstance(gt.vocab, list) else None

    def _dump_owt_samples(step: int, x_ids: torch.Tensor) -> str:
        """
        Dump decoded OWT samples for a given step.
        Writes one sample per line (plain text) to:
          {run_dir}/{dump_prefix}_step{step}.txt
        Returns output path.
        """
        assert tok is not None
        outpath = os.path.join(run_dir, f"{args.dump_prefix}_step{int(step)}.txt")

        # move to CPU once
        x_cpu = x_ids.detach().to("cpu")

        n_dump = int(args.dump_limit) if int(args.dump_limit) > 0 else int(x_cpu.shape[0])
        n_dump = min(n_dump, int(x_cpu.shape[0]))

        with open(outpath, "w", encoding="utf-8") as f:
            for i in range(n_dump):
                ids = x_cpu[i].tolist()
                # important: tokenizer expects List[int]
                txt = tok.decode(ids)
                # keep it 1-sample-per-line for downstream tools (mauve etc.)
                f.write(txt.replace("\n", " ") + "\n")
        return outpath


    for s in steps_list:
    # ----------------------------------------------------
    # Chunked sampling: keep global RNG stream (NO per-chunk reseed)
    # ----------------------------------------------------
        N_total = int(N)
        N_chunk = int(args.N_chunk) if int(args.N_chunk) > 0 else N_total
        if N_chunk > N_total:
            N_chunk = N_total
        if N_chunk <= 0:
            raise ValueError("--N_chunk must be >0, or leave it as 0 to disable chunking.")

        x_chunks: List[torch.Tensor] = []
        dbg_chunks: List[Dict[str, float]] = []

        for b0 in range(0, N_total, N_chunk):
            bN = min(N_chunk, N_total - b0)

            x_b, dbg_b = run_remdm_oracle_steps(
                oracle=oracle,
                steps=int(s),
                N=int(bN),
                T=T,
                V=V,
                device=device,
                sampler=args.sampler,
                nucleus_p=float(args.nucleus_p),
                eta=float(args.eta),
                t_on=float(args.t_on),
                t_off=float(args.t_off),
                alpha_on=float(args.alpha_on),
                noise_removal=bool(args.noise_removal),
                debug_stats=bool(args.debug_stats),
            )
            x_chunks.append(x_b)
            dbg_chunks.append(dbg_b)

        # (optional) free cached blocks between chunks to reduce peak
        # if device.type == "cuda":
        #     torch.cuda.empty_cache()

        x = torch.cat(x_chunks, dim=0)

        # aggregate dbg (final mask frac from concatenated x; last-step same_frac averaged)
        dbg = {
            "mask_frac_final": float((x == mask_index).float().mean().item()),
            "same_frac_last_step": float(
                sum(float(d.get("same_frac_last_step", 0.0)) for d in dbg_chunks) / max(1, len(dbg_chunks))
            ),
        }

        # ----------------------------------------------------
        # Optional: dump decoded samples (OWT only)
        # ----------------------------------------------------
        if (dump_steps is not None) and (int(s) in dump_steps):
            if args.dump_samples and is_owt:
                out_txt = _dump_owt_samples(int(s), x)
                print(f"[OK] Dumped OWT decoded samples: {os.path.abspath(out_txt)}")

            if args.dump_ids:
                x_cpu = x.detach().to("cpu")
                if int(args.dump_limit) > 0:
                    x_cpu = x_cpu[: int(args.dump_limit)]
                out_pt = os.path.join(run_dir, f"{args.dump_prefix}_step{int(s)}.pt")
                torch.save({"x": x_cpu}, out_pt)




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

            "sampler": args.sampler,
            "nucleus_p": float(args.nucleus_p),
            "eta": float(args.eta),
            "t_on": float(args.t_on),
            "t_off": float(args.t_off),
            "alpha_on": float(args.alpha_on),

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
    # Plots
    # --------------------------------------------------------
    xs = [r["steps"] for r in rows]

    def _plot(ykey: str, title: str, ylog: bool = False, ar_value: Optional[float] = None):
        ys = [float(r[ykey]) for r in rows]
        outpath = os.path.join(run_plot_dir, f"{ykey}_vs_steps_{knobs_tag}.png")
        _plot_curve(xs, ys, title=title, xlabel="steps", ylabel=ykey, outpath=outpath, ylog=ylog, ar_value=ar_value)
        print(f"[OK] Saved plot: {os.path.abspath(outpath)}")

    title_suffix = f"T={T} N={N} K={K}"

    _plot("nll_token",          f"{ds_tag} | NLL/token under P' | {title_suffix}",        ylog=False, ar_value=ar_rec["nll_token"])
    _plot("full_kl_rate",       f"{ds_tag} | FULL KL-rate | {title_suffix}",              ylog=True,  ar_value=ar_rec["full_kl_rate"])
    _plot("full_tv_rate",       f"{ds_tag} | FULL TV-rate | {title_suffix}",              ylog=False, ar_value=ar_rec["full_tv_rate"])
    _plot("full_entropy_rate",  f"{ds_tag} | FULL entropy-rate | {title_suffix}",         ylog=False, ar_value=ar_rec["full_entropy_rate"])
    _plot("support_frac",       f"{ds_tag} | support fraction | T={T} N={N}",             ylog=True,  ar_value=ar_rec["support_frac"])

    _plot("unigram_L1",         f"{ds_tag} | unigram L1 vs pi | T={T} N={N}",             ylog=True,  ar_value=ar_rec["unigram_L1"])
    _plot("unique_2gram_ratio", f"{ds_tag} | unique 2-gram ratio | T={T} N={N}",          ylog=False, ar_value=ar_rec["unique_2gram_ratio"])
    _plot("unique_3gram_ratio", f"{ds_tag} | unique 3-gram ratio | T={T} N={N}",          ylog=False, ar_value=ar_rec["unique_3gram_ratio"])
    _plot("dup_rate",           f"{ds_tag} | duplicate sequence rate | T={T} N={N}",      ylog=False, ar_value=ar_rec["dup_rate"])
    _plot("other_mass_rate",    f"{ds_tag} | OTHER-mass rate | {title_suffix}",           ylog=False, ar_value=ar_rec["other_mass_rate"])


if __name__ == "__main__":
    main()