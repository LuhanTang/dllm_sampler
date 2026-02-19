#!/usr/bin/env python3
# tokenizers/build_owt_bpe_gt.py
# ------------------------------------------------------------
# Build a GT Markov chain (bigram) from HF streaming OWT text,
# using a FIXED byte-level BPE tokenizer (tokenizer.json).
#
# Output: a .pt artifact compatible with your oracle + samplers.
#
# Optional vocab collapsing:
#   - keep top V_keep tokens by unigram frequency
#   - map all others -> single OTHER state
#   - IMPORTANT: if collapse_vocab=True, do NOT decode samples using tokenizer.decode
#
# Additions (2026-02, revised):
#   - Default N increased to 1000
#   - REMOVED "real_out" doc-window collection logic (cleaner MAUVE reference)
#   - NEW: optionally save GT-sampled sequences (from (pi, P')) to a *small* pt:
#       --gt_samples_out path.pt
#     containing:
#       gt_samples_ids: LongTensor [N,T]
#       gt_samples_text: list[str] (only if not collapsed)
#       meta: tokenizer_json, N, T, seed, eps, K, target_mass, nu, collapse info, etc.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from sampler.hf_text_stream import StreamConfig, stream_owt_text

# =========================================================
# Numerics
# =========================================================
NORM_CLAMP = 1e-30


def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt
    return plt


def _lazy_import_tokenizers():
    try:
        from tokenizers import Tokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'tokenizers' is required. Install with:\n"
            "  pip install tokenizers\n"
            f"Original error: {e}"
        )
    return Tokenizer


def row_normalize(mat: torch.Tensor, clamp: float = NORM_CLAMP) -> torch.Tensor:
    s = mat.sum(dim=1, keepdim=True).clamp_min(clamp)
    return mat / s


def topk_rows(P: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    V = P.shape[0]
    K = int(min(max(1, K), V))
    vals, idx = torch.topk(P, k=K, dim=1)
    vals = vals / vals.sum(dim=1, keepdim=True).clamp_min(NORM_CLAMP)
    return idx.to(torch.long), vals.to(torch.float32)


def power_iteration_stationary(P: torch.Tensor, pi0: torch.Tensor, iters: int = 500) -> torch.Tensor:
    pi = pi0.clone().to(torch.float32)
    pi = pi / pi.sum().clamp_min(NORM_CLAMP)
    for _ in range(iters):
        pi = pi @ P
        pi = pi / pi.sum().clamp_min(NORM_CLAMP)
    return pi


def sample_markov(pi: torch.Tensor, P: torch.Tensor, N: int, T: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.empty((N, T), dtype=torch.long)
    x[:, 0] = torch.multinomial(pi, num_samples=N, replacement=True, generator=g)
    for t in range(1, T):
        prev = x[:, t - 1]
        probs = P[prev]  # [N,V]
        x[:, t] = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).squeeze(1)
    return x


@torch.no_grad()
def compute_row_sorted_probs(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_probs, sorted_idx = torch.sort(P, dim=1, descending=True)
    return sorted_probs, sorted_idx


@torch.no_grad()
def compute_cum_mass(sorted_probs: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(sorted_probs, dim=1)


@torch.no_grad()
def pick_k_star(cum_mass: torch.Tensor, target_mass: float) -> torch.Tensor:
    V = cum_mass.shape[0]
    target = float(target_mass)
    ge = cum_mass >= target
    first = torch.argmax(ge.to(torch.int64), dim=1)  # 0-based
    last_mass = cum_mass[:, -1]
    bad = last_mass < target - 1e-6
    if bad.any():
        first[bad] = V - 1
    return first + 1  # 1-based


def choose_global_k(k_star: torch.Tensor, strategy: str) -> int:
    s = strategy.lower()
    if s == "median":
        return int(torch.median(k_star).item())
    if s == "mean":
        return int(torch.round(k_star.to(torch.float32).mean()).item())
    if s == "p90":
        return int(torch.quantile(k_star.to(torch.float32), 0.90).item())
    return int(k_star.max().item())


@torch.no_grad()
def estimate_eps_from_tail_mass(
    sorted_probs: torch.Tensor,
    sorted_idx: torch.Tensor,
    nu: torch.Tensor,
    K: int,
    nu_mode: str,
    eps_agg: str = "median",
    clamp_max: float = 0.999999,
) -> Tuple[float, torch.Tensor]:
    V = sorted_probs.shape[0]
    K = int(max(1, min(int(K), V)))
    nu = nu.to(torch.float32)
    nu = nu / nu.sum().clamp_min(NORM_CLAMP)

    head_mass = sorted_probs[:, :K].sum(dim=1)  # [V]
    tail_mass = (1.0 - head_mass).clamp_min(0.0)  # [V]

    if K == V:
        eps_i = torch.zeros((V,), dtype=torch.float32)
    else:
        if nu_mode == "uniform":
            eps_i = tail_mass * (float(V) / float(V - K))
        else:
            tail_idx = sorted_idx[:, K:]  # [V, V-K]
            nu_tail = nu[tail_idx].sum(dim=1).clamp_min(NORM_CLAMP)  # [V]
            eps_i = tail_mass / nu_tail

    eps_i = eps_i.clamp(min=0.0, max=clamp_max)

    agg = eps_agg.lower()
    if agg == "mean":
        eps = float(eps_i.mean().item())
    elif agg == "max":
        eps = float(eps_i.max().item())
    elif agg == "p90":
        eps = float(torch.quantile(eps_i, 0.90).item())
    else:
        eps = float(torch.median(eps_i).item())

    return eps, eps_i


# -------------------------
# plotting
# -------------------------
def plot_kstar_hist(k_star: torch.Tensor, out_path: str, title: str) -> None:
    plt = _lazy_import_matplotlib()
    ks = k_star.cpu().numpy()
    plt.figure()
    plt.hist(ks, bins=range(int(ks.min()), int(ks.max()) + 2), align="left", rwidth=0.9)
    plt.xlabel("K*_i (minimal K reaching target mass)")
    plt.ylabel("Count over tokens i")
    plt.title(title)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_eps_per_row(eps_i: torch.Tensor, out_path: str, title: str) -> None:
    plt = _lazy_import_matplotlib()
    vals = eps_i.cpu().numpy()
    plt.figure()
    plt.hist(vals, bins=60)
    plt.xlabel("eps_i (row-wise)")
    plt.ylabel("Count over tokens i")
    plt.title(title)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# streaming counts
# -------------------------
def build_unigram_streaming_full(
    *,
    tokenizer: "Tokenizer",
    text_stream,
    V: int,
    max_tokens_for_counts: int,
    log_every_docs: int = 2000,
    encode_batch_size: int = 128,
) -> Tuple[torch.Tensor, int, int]:
    unig = torch.zeros((V,), dtype=torch.float64)
    total_tokens = 0
    docs = 0
    batch: List[str] = []

    def _process_batch(batch_texts: List[str]) -> None:
        nonlocal total_tokens, docs, unig
        encs = tokenizer.encode_batch(batch_texts)
        for enc in encs:
            ids = enc.ids
            if not ids:
                continue
            remaining = max_tokens_for_counts - total_tokens
            if remaining <= 0:
                return
            if len(ids) > remaining:
                ids = ids[:remaining]
            t = torch.tensor(ids, dtype=torch.long)
            unig += torch.bincount(t, minlength=V).to(unig.dtype)
            total_tokens += int(t.numel())
            docs += 1
            if log_every_docs > 0 and docs % log_every_docs == 0:
                print(f"[UNI] docs={docs} total_tokens={total_tokens}/{max_tokens_for_counts}")
            if total_tokens >= max_tokens_for_counts:
                return

    for s in text_stream:
        batch.append(s)
        if len(batch) >= encode_batch_size:
            _process_batch(batch)
            batch = []
            if total_tokens >= max_tokens_for_counts:
                break

    if batch and total_tokens < max_tokens_for_counts:
        _process_batch(batch)

    return unig.to(torch.float32), docs, total_tokens


def build_counts_streaming_full(
    *,
    tokenizer: "Tokenizer",
    text_stream,
    V: int,
    max_tokens_for_counts: int,
    log_every_docs: int = 2000,
    encode_batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    unig = torch.zeros((V,), dtype=torch.float64)
    big = torch.zeros((V, V), dtype=torch.float64)
    total_tokens = 0
    docs = 0
    batch: List[str] = []

    def _process_batch(batch_texts: List[str]) -> None:
        nonlocal docs, unig, big, total_tokens
        encs = tokenizer.encode_batch(batch_texts)
        for enc in encs:
            ids = enc.ids
            if not ids:
                continue
            remaining = max_tokens_for_counts - total_tokens
            if remaining <= 0:
                return
            if len(ids) > remaining:
                ids = ids[:remaining]

            t = torch.tensor(ids, dtype=torch.long)
            unig += torch.bincount(t, minlength=V).to(unig.dtype)

            if t.numel() >= 2:
                prev = t[:-1]
                nxt = t[1:]
                idx = prev * V + nxt
                big += torch.bincount(idx, minlength=V * V).reshape(V, V).to(big.dtype)

            total_tokens += int(t.numel())
            docs += 1

            if log_every_docs > 0 and docs % log_every_docs == 0:
                print(f"[COUNT] docs={docs} total_tokens={total_tokens}/{max_tokens_for_counts}")

            if total_tokens >= max_tokens_for_counts:
                return

    for s in text_stream:
        batch.append(s)
        if len(batch) >= encode_batch_size:
            _process_batch(batch)
            batch = []
            if total_tokens >= max_tokens_for_counts:
                break

    if batch and total_tokens < max_tokens_for_counts:
        _process_batch(batch)

    return unig.to(torch.float32), big.to(torch.float32), docs, total_tokens


def build_counts_streaming_collapsed(
    *,
    tokenizer: "Tokenizer",
    text_stream,
    map_full_to_eff: torch.Tensor,  # [V_full] -> [V_eff]
    V_eff: int,
    max_tokens_for_counts: int,
    log_every_docs: int = 2000,
    encode_batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    unig = torch.zeros((V_eff,), dtype=torch.float64)
    big = torch.zeros((V_eff, V_eff), dtype=torch.float64)
    total_tokens = 0
    docs = 0
    batch: List[str] = []

    def _process_batch(batch_texts: List[str]) -> None:
        nonlocal docs, unig, big, total_tokens
        encs = tokenizer.encode_batch(batch_texts)
        for enc in encs:
            ids_full = enc.ids
            if not ids_full:
                continue
            remaining = max_tokens_for_counts - total_tokens
            if remaining <= 0:
                return
            if len(ids_full) > remaining:
                ids_full = ids_full[:remaining]

            t_full = torch.tensor(ids_full, dtype=torch.long)
            t_eff = map_full_to_eff[t_full]  # [L] in effective vocab

            unig += torch.bincount(t_eff, minlength=V_eff).to(unig.dtype)
            if t_eff.numel() >= 2:
                prev = t_eff[:-1]
                nxt = t_eff[1:]
                idx = prev * V_eff + nxt
                big += torch.bincount(idx, minlength=V_eff * V_eff).reshape(V_eff, V_eff).to(big.dtype)

            total_tokens += int(t_eff.numel())
            docs += 1

            if log_every_docs > 0 and docs % log_every_docs == 0:
                print(f"[COLLAPSE-COUNT] docs={docs} total_tokens={total_tokens}/{max_tokens_for_counts}")

            if total_tokens >= max_tokens_for_counts:
                return

    for s in text_stream:
        batch.append(s)
        if len(batch) >= encode_batch_size:
            _process_batch(batch)
            batch = []
            if total_tokens >= max_tokens_for_counts:
                break

    if batch and total_tokens < max_tokens_for_counts:
        _process_batch(batch)

    return unig.to(torch.float32), big.to(torch.float32), docs, total_tokens


def _build_full_itos_stoi(tokenizer) -> Tuple[List[str], Dict[str, int]]:
    vocab = tokenizer.get_vocab()  # token -> id
    V = int(tokenizer.get_vocab_size())
    itos = [""] * V
    for tok, idx in vocab.items():
        if 0 <= idx < V:
            itos[idx] = tok
    for i in range(V):
        if itos[i] == "":
            itos[i] = f"<unk_{i}>"
    return itos, vocab


def _build_eff_vocab_from_top_ids(
    *,
    itos_full: List[str],
    top_full_ids: torch.Tensor,  # [V_keep]
    other_token: str,
) -> Tuple[List[str], Dict[str, int]]:
    V_keep = int(top_full_ids.numel())
    V_eff = V_keep + 1
    OTHER_ID = V_keep

    itos_eff = [""] * V_eff
    for new_id, full_id in enumerate(top_full_ids.tolist()):
        itos_eff[new_id] = itos_full[int(full_id)]
    itos_eff[OTHER_ID] = other_token

    stoi_eff: Dict[str, int] = {tok: i for i, tok in enumerate(itos_eff)}
    return itos_eff, stoi_eff


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", type=str, required=True, help="output .pt path (full GT artifact)")
    ap.add_argument("--tokenizer_dir", type=str, required=True, help="directory containing tokenizer.json")
    ap.add_argument("--encode_batch_size", type=int, default=128)

    # OWT dataset
    ap.add_argument("--owt_name", type=str, default="stanford-cs336/owt-sample")
    ap.add_argument("--split", type=str, default="train")

    # streaming & budget
    ap.add_argument("--max_tokens_for_counts", type=int, default=50_000_000)
    ap.add_argument("--max_chars_per_doc", type=int, default=4096)
    ap.add_argument("--normalize_ws", action="store_true", help="compress whitespace (recommended for OWT)")
    ap.add_argument("--strip", action="store_true", help="strip per doc")
    ap.add_argument("--max_docs", type=int, default=0, help="0 means unlimited; budget stops first")
    ap.add_argument("--log_every_docs", type=int, default=2000)

    # samples from P'
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--N", type=int, default=1000)  # default 1000
    ap.add_argument("--seed", type=int, default=123)

    # NEW: save GT samples only (small pt) for MAUVE reference
    ap.add_argument(
        "--gt_samples_out",
        type=str,
        default="",
        help="If set, save GT-sampled sequences (from (pi,P')) ids+decoded text+meta to this .pt (small).",
    )

    # top-k selection
    ap.add_argument("--topk", type=int, default=128, help="final stored topk (ignored if --auto_topk)")
    ap.add_argument("--auto_topk", action="store_true")
    ap.add_argument("--target_mass", type=float, default=0.99)
    ap.add_argument("--k_strategy", type=str, default="p90", choices=["max", "median", "mean", "p90"])

    # teleport
    ap.add_argument("--nu", type=str, default="unigram", choices=["uniform", "unigram"])
    ap.add_argument("--teleport_eps", type=float, default=1e-4, help="ignored if --auto_eps")
    ap.add_argument("--auto_eps", action="store_true")
    ap.add_argument("--eps_agg", type=str, default="median", choices=["median", "mean", "p90", "max"])

    # stationary
    ap.add_argument("--pi_iters", type=int, default=500)

    # optional collapse (default OFF)
    ap.add_argument("--collapse_vocab", action="store_true")
    ap.add_argument("--v_keep", type=int, default=2048)
    ap.add_argument("--other_token", type=str, default="<other>")

    # plots
    ap.add_argument("--plot_dir", type=str, default="")

    args = ap.parse_args()

    # tokenizer
    tok_json = os.path.join(args.tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tok_json):
        raise FileNotFoundError(f"tokenizer.json not found at: {tok_json}")

    Tokenizer = _lazy_import_tokenizers()
    tokenizer = Tokenizer.from_file(tok_json)
    V_full = int(tokenizer.get_vocab_size())

    cfg = StreamConfig(
        streaming=True,
        split=args.split,
        max_docs=(None if args.max_docs <= 0 else int(args.max_docs)),
        max_chars_per_doc=int(args.max_chars_per_doc),
        normalize_ws=bool(args.normalize_ws),
        strip=bool(args.strip),
    )

    def _make_text_stream():
        return stream_owt_text(args.owt_name, cfg=cfg)

    print(f"[GT] dataset=owt name={args.owt_name} split={args.split}")
    print(f"[TOK] tokenizer={tok_json} V_full={V_full}")
    print(f"[CFG] max_tokens_for_counts={args.max_tokens_for_counts} max_chars_per_doc={args.max_chars_per_doc}")
    print(f"[SAMPLE] T={args.T} N={args.N} seed={args.seed}")
    if args.collapse_vocab:
        print(f"[COLLAPSE] enabled: v_keep={args.v_keep} => V_eff={args.v_keep + 1} (OTHER)")
        if str(args.gt_samples_out).strip():
            print("[GT_SAMPLES][WARN] collapse_vocab=True => gt_samples_text will NOT be decoded (effective ids only).")
    if str(args.gt_samples_out).strip():
        print(f"[GT_SAMPLES] will save GT samples (from (pi,P')) to: {args.gt_samples_out}")

    itos_full, stoi_full = _build_full_itos_stoi(tokenizer)

    # =========================================================
    # counts
    # =========================================================
    if not args.collapse_vocab:
        text_stream = _make_text_stream()
        unig_cnt, big_cnt, n_docs, total_tokens_counted = build_counts_streaming_full(
            tokenizer=tokenizer,
            text_stream=text_stream,
            V=V_full,
            max_tokens_for_counts=int(args.max_tokens_for_counts),
            log_every_docs=int(args.log_every_docs),
            encode_batch_size=int(args.encode_batch_size),
        )
        V_eff = V_full
        itos_eff = itos_full
        stoi_eff = stoi_full
        collapse_info: Dict[str, Any] = {"enabled": False}
    else:
        # Pass A: unigram full
        text_stream_A = _make_text_stream()
        unig_full_cnt, n_docs_A, total_tokens_A = build_unigram_streaming_full(
            tokenizer=tokenizer,
            text_stream=text_stream_A,
            V=V_full,
            max_tokens_for_counts=int(args.max_tokens_for_counts),
            log_every_docs=int(args.log_every_docs),
            encode_batch_size=int(args.encode_batch_size),
        )
        print(f"[UNI] done: docs={n_docs_A} tokens={total_tokens_A} V_full={V_full}")

        v_keep = int(min(max(1, args.v_keep), V_full))
        top_full_ids = torch.argsort(unig_full_cnt, descending=True)[:v_keep].to(torch.long)
        OTHER_ID = v_keep
        V_eff = v_keep + 1

        map_full_to_eff = torch.full((V_full,), OTHER_ID, dtype=torch.long)
        map_full_to_eff[top_full_ids] = torch.arange(v_keep, dtype=torch.long)

        # Pass B: counts collapsed
        text_stream_B = _make_text_stream()
        unig_cnt, big_cnt, n_docs, total_tokens_counted = build_counts_streaming_collapsed(
            tokenizer=tokenizer,
            text_stream=text_stream_B,
            map_full_to_eff=map_full_to_eff,
            V_eff=V_eff,
            max_tokens_for_counts=int(args.max_tokens_for_counts),
            log_every_docs=int(args.log_every_docs),
            encode_batch_size=int(args.encode_batch_size),
        )
        print(f"[COUNT] done (collapsed): docs={n_docs} tokens={total_tokens_counted} V_eff={V_eff}")

        itos_eff, stoi_eff = _build_eff_vocab_from_top_ids(
            itos_full=itos_full,
            top_full_ids=top_full_ids,
            other_token=args.other_token,
        )

        collapse_info = {
            "enabled": True,
            "V_full": int(V_full),
            "V_keep": int(v_keep),
            "V_eff": int(V_eff),
            "OTHER_ID": int(OTHER_ID),
            "other_token": str(args.other_token),
            "top_full_ids": top_full_ids.cpu(),
            "map_full_to_eff": map_full_to_eff.cpu(),
            "unig_full_cnt": unig_full_cnt.cpu(),
        }

        unig_eff = unig_cnt / unig_cnt.sum().clamp_min(NORM_CLAMP)
        print(f"[COLLAPSE] OTHER mass = {float(unig_eff[OTHER_ID].item()):.4f}")

    # =========================================================
    # build oracle from counts
    # =========================================================
    V = int(V_eff)
    unig = unig_cnt / unig_cnt.sum().clamp_min(NORM_CLAMP)

    C = big_cnt.clone()
    row_sums = C.sum(dim=1)
    zero_rows = row_sums <= 0
    if zero_rows.any():
        # fallback: if a row never appears, use unigram
        C[zero_rows] = unig.view(1, V).expand(int(zero_rows.sum().item()), V)

    P_mle = row_normalize(C + 1e-8)

    # nu
    if args.nu == "uniform":
        nu = torch.full((V,), 1.0 / V, dtype=torch.float32)
    else:
        nu = unig.clone()

    # auto K via cum-mass
    target_mass = float(max(0.0, min(0.999999, args.target_mass)))
    sorted_probs, sorted_idx = compute_row_sorted_probs(P_mle)
    cum_mass = compute_cum_mass(sorted_probs)
    k_star = pick_k_star(cum_mass, target_mass=target_mass)

    if args.auto_topk:
        K_final = choose_global_k(k_star, strategy=args.k_strategy)
    else:
        K_final = int(max(1, min(int(args.topk), V)))

    # eps
    eps_i = None
    if args.auto_eps:
        eps, eps_i = estimate_eps_from_tail_mass(
            sorted_probs=sorted_probs,
            sorted_idx=sorted_idx,
            nu=nu,
            K=K_final,
            nu_mode=args.nu,
            eps_agg=args.eps_agg,
        )
        eps_source = "auto"
    else:
        eps = float(args.teleport_eps)
        eps_source = "user"

    EPS_MIN = 1e-12
    EPS_MAX = 0.999999
    eps_raw = float(eps)
    eps = float(max(EPS_MIN, min(EPS_MAX, eps_raw)))
    if eps != eps_raw:
        print(f"[WARN] teleport_eps clamped: raw={eps_raw:.3e} -> eps={eps:.3e} (source={eps_source})")

    # sparse topk
    nbr_idx, nbr_prob = topk_rows(P_mle, K_final)

    # dense P' for stationary + sampling
    P_topk_dense = torch.zeros((V, V), dtype=torch.float32)
    P_topk_dense.scatter_(dim=1, index=nbr_idx, src=nbr_prob)
    P_prime = (1.0 - eps) * P_topk_dense + eps * nu.view(1, V)

    row_sum = P_prime.sum(dim=1)
    print(f"[SANITY] P' row_sum: min={row_sum.min().item():.6f} max={row_sum.max().item():.6f}")

    # stationary + samples
    pi0 = unig.clone()
    pi_stationary = power_iteration_stationary(P_prime, pi0, iters=int(args.pi_iters))
    samples = sample_markov(pi_stationary, P_prime, N=int(args.N), T=int(args.T), seed=int(args.seed))

    # optional plots
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        plot_kstar_hist(
            k_star=k_star,
            out_path=os.path.join(args.plot_dir, f"kstar_hist_owt_target{target_mass:.3f}.png"),
            title=f"owt: K*_i distribution (target={target_mass:.3f}); chosen K={K_final} via {args.k_strategy}",
        )
        if eps_i is not None:
            plot_eps_per_row(
                eps_i=eps_i,
                out_path=os.path.join(args.plot_dir, f"eps_i_owt_K{K_final}_{args.nu}_{args.eps_agg}.png"),
                title=f"owt: eps_i (K={K_final}, nu={args.nu}) agg={args.eps_agg}; eps={eps:.2e}",
            )

    logP_safe = torch.log(P_prime.clamp_min(NORM_CLAMP))

    # =========================================================
    # NEW: optionally save GT samples only (small pt) for MAUVE reference
    # =========================================================
    if str(args.gt_samples_out).strip():
        gt_text: List[str] = []
        if (not args.collapse_vocab) and samples.numel() > 0:
            for row in samples.tolist():
                try:
                    gt_text.append(tokenizer.decode(row))
                except Exception:
                    gt_text.append("")
        else:
            # collapse_vocab=True: decoding is not meaningful
            gt_text = []

        gt_pack: Dict[str, Any] = {
            "gt_samples_ids": samples,          # LongTensor [N,T] in modeling space (eff vocab)
            "gt_samples_text": gt_text,         # List[str], only if not collapsed
            "meta": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "dataset": "owt",
                "dataset_name": str(args.owt_name),
                "split": str(args.split),
                "tokenizer_dir": os.path.abspath(args.tokenizer_dir),
                "tokenizer_json": os.path.abspath(tok_json),
                "collapse_vocab": bool(args.collapse_vocab),
                "V_full": int(V_full),
                "V_eff": int(V),
                "V_keep": (int(args.v_keep) if args.collapse_vocab else int(V_full)),
                "other_token": (str(args.other_token) if args.collapse_vocab else ""),
                "T": int(args.T),
                "N": int(args.N),
                "seed": int(args.seed),
                "topk": int(K_final),
                "auto_topk": bool(args.auto_topk),
                "target_mass": float(target_mass),
                "k_strategy": str(args.k_strategy),
                "nu": str(args.nu),
                "teleport_eps": float(eps),
                "teleport_eps_source": str(eps_source),
                "eps_agg": (str(args.eps_agg) if args.auto_eps else ""),
                "pi_iters": int(args.pi_iters),
                "note": (
                    "GT MAUVE reference: samples are drawn from (pi_stationary, P_prime) "
                    "and decoded with the SAME tokenizer as sampler outputs. "
                    "If collapse_vocab=True, text decoding is omitted."
                ),
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.gt_samples_out)) or ".", exist_ok=True)
        torch.save(gt_pack, args.gt_samples_out)
        print(f"[GT_SAMPLES] saved to {os.path.abspath(args.gt_samples_out)} "
              f"(N={samples.shape[0]} T={samples.shape[1]})")

    # =========================================================
    # save GT artifact (full)
    # =========================================================
    artifact: Dict[str, Any] = {
        "config": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dataset": "owt",
            "dataset_name": args.owt_name,
            "split": args.split,
            "tokenizer_dir": os.path.abspath(args.tokenizer_dir),
            "tokenizer_json": os.path.abspath(tok_json),
            "V_full": int(V_full),
            "V_eff": int(V),
            "collapse_vocab": bool(args.collapse_vocab),
            "V_keep": (int(args.v_keep) if args.collapse_vocab else int(V_full)),
            "other_token": (str(args.other_token) if args.collapse_vocab else ""),
            "T": int(args.T),
            "N": int(args.N),
            "seed": int(args.seed),
            "max_tokens_for_counts": int(args.max_tokens_for_counts),
            "tokens_counted": int(total_tokens_counted),
            "max_chars_per_doc": int(args.max_chars_per_doc),
            "max_docs": int(args.max_docs),
            "normalize_ws": bool(args.normalize_ws),
            "strip": bool(args.strip),
            "topk": int(K_final),
            "auto_topk": bool(args.auto_topk),
            "target_mass": float(target_mass),
            "k_strategy": args.k_strategy,
            "nu": args.nu,
            "teleport_eps": float(eps),
            "teleport_eps_source": str(eps_source),
            "eps_agg": (args.eps_agg if args.auto_eps else ""),
            "pi_iters": int(args.pi_iters),
            "docs_counted": int(n_docs),
            "gt_samples_out": (os.path.abspath(args.gt_samples_out) if str(args.gt_samples_out).strip() else ""),
        },
        "collapse": collapse_info,

        # effective vocab space
        "vocab": itos_eff,  # id -> token (list[str]) in effective space
        "stoi": stoi_eff,   # token -> id (dict) in effective space
        "itos": {i: itos_eff[i] for i in range(V)},  # compatibility

        # oracle tensors (effective)
        "unigram": unig,        # [V]
        "nbr_idx": nbr_idx,     # [V,K]
        "nbr_prob": nbr_prob,   # [V,K]
        "nu": nu,               # [V]
        "eps": float(eps),      # scalar
        "pi": pi_stationary,    # [V]

        "samples": samples,     # [N,T] token ids in effective space

        # dense debug (heavy)
        "P": P_prime,           # [V,V]
        "logP_safe": logP_safe, # [V,V]

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

    print(f"[OK] saved GT to {os.path.abspath(args.out)}")
    print(f"[GT] V_full={V_full} V_eff={V} T={args.T} N={args.N} K={K_final} target_mass={target_mass:.3f} nu={args.nu}")
    print(f"[K*] per-row: min={int(k_star.min().item())} median={int(torch.median(k_star).item())} max={int(k_star.max().item())}")
    if args.auto_eps:
        print(f"[eps] auto ({args.eps_agg}) = {eps:.3e}")
    else:
        print(f"[eps] user = {eps:.3e}")
    if args.collapse_vocab:
        print(f"[COLLAPSE] V_keep={args.v_keep} => V_eff={V} (OTHER)")
    if args.plot_dir:
        print(f"[plots] saved to {os.path.abspath(args.plot_dir)}")


if __name__ == "__main__":
    main()

