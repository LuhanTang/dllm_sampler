#!/usr/bin/env python3
# sampler/sanity_check_gt.py
from __future__ import annotations

import argparse
import torch

NORM_CLAMP = 1e-30


def _find_other_id(vocab_list: list[str], other_token: str) -> int | None:
    for i, tok in enumerate(vocab_list):
        if tok == other_token:
            return i
    return None


@torch.no_grad()
def power_iter(P: torch.Tensor, pi0: torch.Tensor, iters: int = 500) -> torch.Tensor:
    pi = pi0.clone().to(torch.float64)
    pi = pi / pi.sum().clamp_min(NORM_CLAMP)
    for _ in range(iters):
        pi = pi @ P.to(torch.float64)
        pi = pi / pi.sum().clamp_min(NORM_CLAMP)
    return pi.to(torch.float32)


@torch.no_grad()
def tv_dist(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p - q).abs().sum().item())


@torch.no_grad()
def summarize_other_mass(P: torch.Tensor, other_id: int) -> dict:
    col = P[:, other_id].detach().cpu()
    return {
        "p50": float(col.median().item()),
        "p90": float(torch.quantile(col, 0.90).item()),
        "p99": float(torch.quantile(col, 0.99).item()),
        "max": float(col.max().item()),
        "mean": float(col.mean().item()),
    }


@torch.no_grad()
def stationary_consistency(P: torch.Tensor, unigram: torch.Tensor, trials: int = 3, iters: int = 800) -> dict:
    V = P.shape[0]
    base = power_iter(P, unigram, iters=iters)

    out = {"tv_vs_unigram_init": []}

    for k in range(trials):
        g = torch.Generator().manual_seed(1234 + k)
        rand = torch.rand((V,), generator=g)
        rand = rand / rand.sum().clamp_min(NORM_CLAMP)
        pi_k = power_iter(P, rand, iters=iters)
        out["tv_vs_unigram_init"].append(tv_dist(base, pi_k))

    out["tv_max"] = float(max(out["tv_vs_unigram_init"])) if out["tv_vs_unigram_init"] else 0.0
    out["tv_mean"] = float(sum(out["tv_vs_unigram_init"]) / max(1, len(out["tv_vs_unigram_init"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True, help="path to saved .pt GT artifact")
    ap.add_argument("--other_token", type=str, default="<other>", help="token string used for OTHER (if exists)")
    ap.add_argument("--pi_iters", type=int, default=800)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    art = torch.load(args.gt, map_location="cpu")

    P = art.get("P", None)
    if P is None:
        P = art.get("P_prime_dense", None)
    if P is None:
        raise KeyError("Cannot find P or P_prime_dense in artifact.")

    unig = art.get("unigram", None)
    if unig is None:
        raise KeyError("Cannot find unigram in artifact.")

    vocab = art.get("vocab", None)
    if vocab is None:
        # fallback: try itos dict
        itos = art.get("itos", None)
        if isinstance(itos, dict):
            vocab = [itos[i] for i in range(len(itos))]
    if vocab is None:
        raise KeyError("Cannot find vocab/itos in artifact.")

    V = int(P.shape[0])
    print(f"[GT] V={V}  eps={art.get('eps', 'NA')}  nu={art.get('config', {}).get('nu', 'NA')}")
    print(f"[GT] file={args.gt}")

    # 1) OTHER transition diagnostics
    other_id = _find_other_id(vocab, args.other_token)
    if other_id is None:
        print(f"[OTHER] token '{args.other_token}' not found in vocab; skip OTHER diagnostics.")
    else:
        stats = summarize_other_mass(P, other_id)
        print(f"[OTHER] id={other_id} token={args.other_token}")
        print(f"[OTHER] P(i->OTHER): mean={stats['mean']:.6f} p50={stats['p50']:.6f} p90={stats['p90']:.6f} p99={stats['p99']:.6f} max={stats['max']:.6f}")

    # 2) Stationary consistency from different inits
    cons = stationary_consistency(P, unig, trials=int(args.trials), iters=int(args.pi_iters))
    print(f"[PI] iters={args.pi_iters} trials={args.trials}")
    print(f"[PI] TV(pi_unigram_init, pi_rand_init): {cons['tv_vs_unigram_init']}")
    print(f"[PI] TV max={cons['tv_max']:.6e}  mean={cons['tv_mean']:.6e}")

    # 3) Quick row-sum check
    row_sums = P.sum(dim=1)
    print(f"[CHK] row_sums: min={float(row_sums.min().item()):.6f} max={float(row_sums.max().item()):.6f}")

    # 4) Optional: how much mass in OTHER unigram
    if other_id is not None:
        other_mass = float(unig[other_id].item())
        print(f"[UNI] OTHER mass={other_mass:.6f}")

if __name__ == "__main__":
    main()
