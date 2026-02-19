#!/usr/bin/env python3
# sampler/build_text8_char_gt_k27_eps0.py
# ------------------------------------------------------------
# Fixed-config builder for text8-char GT (WITH SPACES):
#   - vocab: 26 letters + space  (V=27)
#   - top-k: fixed K=27
#   - teleport: fixed eps=0.0 (no teleport mixture)
#
# Also saves:
#   - empirical unigram in artifact["unigram"]
#   - sparse top-k transitions nbr_idx/nbr_prob
#   - dense P (P' == P_topk since eps=0)
#   - stationary distribution pi
#   - sampled sequences "samples" from the stationary chain
#
# Usage:
#   python -u -m sampler.build_text8_char_gt_k27_eps0 \
#     --out gt_text8_char_withSpace_T1024_N128_topk27_eps0.pt \
#     --T 1024 --N 128 --max_chars 5000000 --seed 123 \
#     --text8_path ./text8
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch


# -------------------------
# text preprocessing
# -------------------------
def load_text8_chars(*, text8_path: str, max_chars: int | None = None) -> str:
    """
    Load local text8 file as a raw character stream.
    The official text8 file contains spaces between words.
    """
    if not text8_path:
        raise ValueError("--text8_path is required (e.g., ./text8)")
    if not os.path.isfile(text8_path):
        raise FileNotFoundError(f"text8 file not found: {text8_path}")

    with open(text8_path, "r", encoding="utf-8") as f:
        s = f.read()

    # Optional truncate
    if max_chars is not None and max_chars > 0:
        s = s[: int(max_chars)]
    return s


def make_vocab() -> List[str]:
    # 26 letters + space
    return [chr(ord("a") + i) for i in range(26)] + [" "]


def text_to_ids(s: str, stoi: Dict[str, int]) -> torch.Tensor:
    ids: List[int] = []
    for ch in s:
        if ch in stoi:
            ids.append(int(stoi[ch]))
        # ignore other chars (should not happen for official text8)
    return torch.tensor(ids, dtype=torch.long)


# -------------------------
# build bigram counts
# -------------------------
def bigram_counts(x: torch.Tensor, V: int) -> torch.Tensor:
    prev = x[:-1]
    nxt = x[1:]
    idx = prev * V + nxt
    cnt = torch.bincount(idx, minlength=V * V).reshape(V, V).to(torch.float32)
    return cnt


def row_normalize(mat: torch.Tensor, clamp: float = 1e-30) -> torch.Tensor:
    s = mat.sum(dim=1, keepdim=True).clamp_min(clamp)
    return mat / s


def topk_rows(P: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return per-row top-K indices and renormalized probs.
    For K=V, this is basically the dense row (but still returned in topk form).
    """
    V = P.shape[0]
    K = int(min(max(1, K), V))
    vals, idx = torch.topk(P, k=K, dim=1)
    vals = vals / vals.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return idx.to(torch.long), vals.to(torch.float32)


def power_iteration_stationary(P: torch.Tensor, pi0: torch.Tensor, iters: int = 500) -> torch.Tensor:
    pi = pi0.clone()
    pi = pi / pi.sum().clamp_min(1e-30)
    for _ in range(int(iters)):
        pi = pi @ P
        pi = pi / pi.sum().clamp_min(1e-30)
    return pi


def sample_markov(pi: torch.Tensor, P: torch.Tensor, N: int, T: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    x = torch.empty((int(N), int(T)), dtype=torch.long)
    x[:, 0] = torch.multinomial(pi, num_samples=int(N), replacement=True, generator=g)
    for t in range(1, int(T)):
        prev = x[:, t - 1]
        probs = P[prev]  # [N,V]
        x[:, t] = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).squeeze(1)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="output .pt path")

    # dataset / sampling
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--max_chars", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=123)

    # stationary
    ap.add_argument("--pi_iters", type=int, default=500)

    # local file
    ap.add_argument("--text8_path", type=str, default="./text8", help="Path to local text8 file (with spaces)")

    args = ap.parse_args()

    # -------------------------
    # fixed knobs
    # -------------------------
    K_final = 27
    eps = 0.0
    nu_mode = "uniform"  # stored for record only

    # -------------------------
    # build vocab + read data
    # -------------------------
    vocab = make_vocab()
    stoi = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    assert V == 27, f"Expected V=27, got V={V}"

    text = load_text8_chars(text8_path=args.text8_path, max_chars=int(args.max_chars))
    ids = text_to_ids(text, stoi)

    if ids.numel() < 1000:
        raise RuntimeError("Too few characters parsed from text8.")

    # sanity: must truly contain spaces
    space_id = stoi[" "]
    space_frac = float((ids == space_id).float().mean().item())
    print("[CHECK] space_frac =", space_frac)
    assert space_frac > 0.05, (
        "Space fraction too small. This usually means your text8 file has no spaces "
        "or you loaded the wrong file. Make sure --text8_path points to the official text8."
    )

    # unigram + bigram
    unig = torch.bincount(ids, minlength=V).to(torch.float32)
    unig = unig / unig.sum().clamp_min(1e-30)

    C = bigram_counts(ids, V)  # [V,V]
    P_mle = row_normalize(C + 1e-8)  # tiny floor avoids NaN rows

    # nu (unused when eps=0, but keep for compatibility)
    nu = torch.full((V,), 1.0 / V, dtype=torch.float32)

    # -------------------------
    # build sparse topk + dense P'
    # -------------------------
    nbr_idx, nbr_prob = topk_rows(P_mle, K_final)

    P_topk_dense = torch.zeros((V, V), dtype=torch.float32)
    P_topk_dense.scatter_(dim=1, index=nbr_idx, src=nbr_prob)

    # since eps=0, P' = P_topk
    P_prime = P_topk_dense

    # stationary pi under P'
    pi0 = unig.clone()
    pi_stationary = power_iteration_stationary(P_prime, pi0, iters=int(args.pi_iters))

    # sample sequences from P'
    samples = sample_markov(pi_stationary, P_prime, N=int(args.N), T=int(args.T), seed=int(args.seed))

    logP_safe = torch.log(P_prime.clamp_min(1e-30))
    itos = {i: c for i, c in enumerate(vocab)}

    # -------------------------
    # save artifact
    # -------------------------
    artifact = {
        "config": {
            "topk": int(K_final),
            "teleport_eps": float(eps),
            "teleport_eps_source": "fixed",
            "nu": nu_mode,
            "text8_path": os.path.abspath(args.text8_path),
            "max_chars": int(args.max_chars),
            "pi_iters": int(args.pi_iters),
            "seed": int(args.seed),
            "T": int(args.T),
            "N": int(args.N),
            "V": int(V),
            "space_frac": float(space_frac),
        },
        "vocab": vocab,
        "stoi": stoi,
        "itos": itos,
        "unigram": unig,  # [V]

        # sparse top-k
        "nbr_idx": nbr_idx,    # [V,K]
        "nbr_prob": nbr_prob,  # [V,K]

        # teleport fields (eps fixed 0)
        "nu": nu,              # [V]
        "eps": float(eps),     # scalar

        # stationary + samples
        "pi": pi_stationary,   # [V]
        "samples": samples,    # [N,T]

        # dense debug (small V)
        "P": P_prime,          # [V,V]
        "logP_safe": logP_safe,

        # compatibility aliases
        "V": int(V),
        "T": int(args.T),
        "N": int(args.N),
        "pi_stationary": pi_stationary,
        "teleport_eps": float(eps),
        "P_prime_dense": P_prime,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(artifact, args.out)

    print(f"[OK] saved GT to {args.out}")
    print(f"[GT] V={V} T={args.T} N={args.N} K={K_final} eps={eps:.1f} (fixed)")
    print(f"[GT] unigram(space)={float(unig[space_id]):.6f}  space_frac={space_frac:.4f}")


if __name__ == "__main__":
    main()

