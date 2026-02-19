#!/usr/bin/env python3
# sampler/run_ar_temp_genppl_gpt2.py
# ------------------------------------------------------------
# Load sparse bigram prior (top-K + teleport), sample AR sequences
# (beta=1 vs beta>1), then evaluate GenPPL using gpt2-large.
#
# Paper-ready outputs:
#   - summary.json
#   - results.csv
#   - genppl_vs_beta.png
#   - unique2_vs_beta.png
#   - batch_entropy_vs_beta.png
#   - sentence_entropy_vs_beta.png
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# AR sampler: top-K + teleport, with beta sharpening on the top-K part
# ----------------------------
@torch.no_grad()
def sample_ar_sparse_teleport(
    *,
    pi0: torch.Tensor,             # [V]
    nbr_idx: torch.Tensor,         # [V,K]
    nbr_prob: torch.Tensor,        # [V,K]
    nu: torch.Tensor,              # [V]
    eps: float,
    N: int,
    T: int,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """
    AR sampling from:
      P'(j|i) = (1-eps) P_topK(j|i) + eps * nu(j)

    with beta-sharpening applied ONLY to the top-K component:
      P_beta(j|i) ∝ P_topK(j|i)^beta
    """
    V, K = nbr_idx.shape
    pi0 = pi0.to(device=device, dtype=torch.float32)
    nu = nu.to(device=device, dtype=torch.float32)
    nbr_idx = nbr_idx.to(device=device)
    nbr_prob = nbr_prob.to(device=device, dtype=torch.float32)

    if abs(beta - 1.0) < 1e-8:
        nbr_prob_beta = nbr_prob
    else:
        p = torch.clamp(nbr_prob, min=1e-30)
        p = p ** float(beta)
        p = p / torch.clamp(p.sum(dim=1, keepdim=True), min=1e-30)
        nbr_prob_beta = p

    x = torch.empty((N, T), device=device, dtype=torch.long)
    x[:, 0] = torch.multinomial(pi0, num_samples=N, replacement=True)

    for t in range(1, T):
        prev = x[:, t - 1]
        u = torch.rand((N,), device=device)
        do_tele = u < float(eps)

        if do_tele.any():
            x[do_tele, t] = torch.multinomial(
                nu, num_samples=int(do_tele.sum()), replacement=True
            )

        if (~do_tele).any():
            idx = prev[~do_tele]
            nbrs = nbr_idx[idx]
            probs = nbr_prob_beta[idx]
            kk = torch.multinomial(probs, num_samples=1).squeeze(1)
            x[~do_tele, t] = nbrs[torch.arange(len(idx)), kk]

    return x


# ----------------------------
# Diversity / collapse metrics
# ----------------------------
def unique_n(x: np.ndarray, n: int) -> float:
    """Unique-n ratio (same as distinct-n)."""
    N, T = x.shape
    if T < n:
        return 0.0
    total = N * (T - n + 1)
    s = set()
    for i in range(N):
        for j in range(T - n + 1):
            s.add(tuple(x[i, j:j + n]))
    return len(s) / total


def sentence_entropy(x: np.ndarray, V: int) -> float:
    """
    Sentence entropy (unigram-based).

    Entropy of the empirical token unigram distribution aggregated over all
    generated sequences in the batch. This matches the common usage of
    "sentence entropy" in prior work and reflects token-level diversity.
    """
    flat = x.reshape(-1)
    counts = np.bincount(flat, minlength=V).astype(np.float64)
    p = counts / max(1.0, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def batch_entropy(x: np.ndarray) -> float:
    """
    Batch entropy over full sequences.

    Empirical entropy of the distribution over complete sequences in the batch.
    Mainly sensitive to exact-duplicate / few-mode collapse.
    """
    N = x.shape[0]
    if N <= 1:
        return 0.0
    counts: Dict[bytes, int] = {}
    for i in range(N):
        key = x[i].tobytes()
        counts[key] = counts.get(key, 0) + 1
    H = 0.0
    for c in counts.values():
        p = c / N
        H -= p * np.log(p)
    return float(H)


def dup_rate(x: np.ndarray) -> float:
    """Fraction of duplicate sequences in the batch."""
    seen = set()
    dup = 0
    for i in range(x.shape[0]):
        key = x[i].tobytes()
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    return dup / max(1, x.shape[0])


# ----------------------------
# GenPPL (gpt2-large)
# ----------------------------
@torch.no_grad()
def eval_genppl_gpt2_large(samples: torch.Tensor, device: torch.device, batch_size: int = 8):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()

    N, T = samples.shape
    total_nll, total_tokens = 0.0, 0

    for st in range(0, N, batch_size):
        ed = min(N, st + batch_size)
        x = samples[st:ed].to(device)
        out = model(input_ids=x, labels=x)
        num = (T - 1) * (ed - st)
        total_nll += out.loss.item() * num
        total_tokens += num

    mean_nll = total_nll / total_tokens
    return {"genppl_nll": mean_nll, "genppl": float(np.exp(mean_nll))}


# ----------------------------
# Plot helper
# ----------------------------
def save_simple_plot(x, y, xlabel, ylabel, out_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_pt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--betas", default="1.0,1.5,2.0")
    ap.add_argument("--genppl_batch", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    prior = torch.load(args.prior_pt, map_location="cpu")
    V, K, eps = int(prior["V"]), int(prior["K"]), float(prior["eps"])

    betas = [float(b) for b in args.betas.split(",")]
    results = []

    for beta in betas:
        x = sample_ar_sparse_teleport(
            pi0=prior.get("pi0", prior["nu"]),
            nbr_idx=prior["nbr_idx"],
            nbr_prob=prior["nbr_prob"],
            nu=prior["nu"],
            eps=eps,
            N=args.N,
            T=args.T,
            beta=beta,
            device=device,
        )

        x_cpu = x.cpu().numpy().astype(np.int32)
        rec = {
            "beta": beta,
            "genppl": eval_genppl_gpt2_large(x, device, args.genppl_batch)["genppl"],
            "unique_2": unique_n(x_cpu, 2),
            "sentence_entropy": sentence_entropy(x_cpu, V),
            "batch_entropy": batch_entropy(x_cpu),
            "dup_rate": dup_rate(x_cpu),
        }
        results.append(rec)

        print(
            f"beta={beta:.2f} | GenPPL={rec['genppl']:.2f} | "
            f"unique2={rec['unique_2']:.3f} | "
            f"H_sent={rec['sentence_entropy']:.3f} | "
            f"H_batch={rec['batch_entropy']:.3f}"
        )

    # save csv/json
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"results": results}, f, indent=2)

    bet = [r["beta"] for r in results]
    save_simple_plot(bet, [r["genppl"] for r in results],
                     "beta (sharpening)", "GenPPL", os.path.join(args.out_dir, "genppl_vs_beta.png"))
    save_simple_plot(bet, [r["unique_2"] for r in results],
                     "beta (sharpening)", "unique-2", os.path.join(args.out_dir, "unique2_vs_beta.png"))
    save_simple_plot(bet, [r["sentence_entropy"] for r in results],
                     "beta (sharpening)", "Sentence entropy (unigram)",
                     os.path.join(args.out_dir, "sentence_entropy_vs_beta.png"))
    save_simple_plot(bet, [r["batch_entropy"] for r in results],
                     "beta (sharpening)", "Batch entropy (full sequences)",
                     os.path.join(args.out_dir, "batch_entropy_vs_beta.png"))


if __name__ == "__main__":
    main()
