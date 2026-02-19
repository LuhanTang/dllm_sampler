# sampler/metrics_text8_full.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# =========================================================
# Numerics
# =========================================================
NORM_CLAMP = 1e-30

def _ensure_distribution(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(0.0)
    return p / p.sum().clamp_min(NORM_CLAMP)

# =========================================================
# Sparse teleport prior (rank-1 teleport)
# =========================================================
@dataclass
class SparseTeleportPrior:
    """
    P'(i->j) = (1-eps) * P_topk(i->j) + eps * nu(j)
    Stored as outgoing adjacency list for P_topk:
      nbr_idx:  [V,K] long
      nbr_prob: [V,K] float, row-stochastic over K neighbors
    plus teleport:
      nu: [V] distribution
      eps: scalar
    """
    nbr_idx: torch.Tensor
    nbr_prob: torch.Tensor
    nu: torch.Tensor
    eps: float

    @property
    def V(self) -> int:
        return int(self.nbr_idx.shape[0])

    @property
    def K(self) -> int:
        return int(self.nbr_idx.shape[1])


# =========================================================
# Metrics: n-gram + dup + unigram
# =========================================================
def _as_numpy_int64(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").contiguous().numpy().astype(np.int64, copy=False)


def unigram_l1(x: torch.Tensor, pi: torch.Tensor, V: int) -> float:
    counts = torch.bincount(x.reshape(-1), minlength=V).float()
    p = counts / counts.sum().clamp_min(1.0)
    pi = _ensure_distribution(pi.to(p.device))
    return float(torch.sum(torch.abs(p - pi)).item())


def _hash_ngrams_uint64(seq: np.ndarray, n: int) -> Tuple[int, int]:
    T = int(seq.shape[0])
    if T < n:
        return 0, 0
    seen = set()
    total = 0
    for t in range(T - n + 1):
        hh = np.uint64(0x9e3779b97f4a7c15)
        for k in range(n):
            hh ^= np.uint64(seq[t + k] + 1) + np.uint64(0x9e3779b97f4a7c15) + (hh << np.uint64(6)) + (hh >> np.uint64(2))
            hh *= np.uint64(0xbf58476d1ce4e5b9)
        seen.add(int(hh))
        total += 1
    return len(seen), total


def unique_ngram_ratio(x: torch.Tensor, n: int) -> float:
    x_np = _as_numpy_int64(x)
    uniq = 0
    tot = 0
    for s in x_np:
        u, t = _hash_ngrams_uint64(s, n=n)
        uniq += u
        tot += t
    return float(uniq / max(1, tot))


def dup_rate(x: torch.Tensor) -> float:
    x_np = _as_numpy_int64(x)
    N = x_np.shape[0]
    seen = set()
    for r in range(N):
        h = np.uint64(0x9e3779b97f4a7c15)
        row = x_np[r]
        for v in row:
            h ^= np.uint64(v + 1) + np.uint64(0x9e3779b97f4a7c15) + (h << np.uint64(6)) + (h >> np.uint64(2))
            h *= np.uint64(0xbf58476d1ce4e5b9)
        seen.add(int(h))
    return float(1.0 - (len(seen) / max(1, N)))


def top_unigrams_bigrams_print(x: torch.Tensor, V: int, k: int = 15, vocab: Optional[List[str]] = None) -> None:
    x_cpu = x.detach().to("cpu")
    N, T = x_cpu.shape

    uc = torch.bincount(x_cpu.reshape(-1), minlength=V).float()
    topv, topi = torch.topk(uc, k=min(k, V))
    print("\n[Sanity] Top unigrams:")
    for c, i in zip(topv.tolist(), topi.tolist()):
        tok = vocab[i] if (vocab is not None and i < len(vocab)) else str(i)
        print(f"  {tok:>6s} : {c / uc.sum().item():.4f}")

    cnt: Dict[Tuple[int, int], int] = {}
    for r in range(N):
        row = x_cpu[r].tolist()
        for t in range(T - 1):
            key = (row[t], row[t + 1])
            cnt[key] = cnt.get(key, 0) + 1
    items = sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:k]
    print("\n[Sanity] Top bigrams:")
    denom = N * (T - 1)
    for (i, j), c in items:
        ti = vocab[i] if (vocab is not None and i < len(vocab)) else str(i)
        tj = vocab[j] if (vocab is not None and j < len(vocab)) else str(j)
        print(f"  ({ti},{tj}) : {c / denom:.4f}")


# =========================================================
# Edge-table acceleration (NO interface changes)
# =========================================================
# Cache: key -> (edge_key_sorted, edge_prob_sorted, edge_pos_sorted)
#   edge_key = i*V + j  for each topK edge (i -> nbr_idx[i,k])
_EDGE_CACHE: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

def _edge_cache_key(prior: SparseTeleportPrior, device: torch.device) -> Tuple:
    # Using data_ptr to invalidate cache when tensors change.
    # Include V,K and device to avoid accidental reuse.
    nbr_idx = prior.nbr_idx
    nbr_prob = prior.nbr_prob
    return (
        str(device),
        int(prior.V),
        int(prior.K),
        int(nbr_idx.data_ptr()),
        int(nbr_prob.data_ptr()),
        str(nbr_idx.dtype),
        str(nbr_prob.dtype),
    )

@torch.no_grad()
def _get_edge_table(prior: SparseTeleportPrior, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build or fetch cached sorted edge table on the given device.

    Returns:
      edge_key_sorted:  [V*K] int64
      edge_prob_sorted: [V*K] float32
      edge_pos_sorted:  [V*K] int64   (the neighbor slot k in [0,K-1])
    """
    key = _edge_cache_key(prior, device)
    cached = _EDGE_CACHE.get(key, None)
    if cached is not None:
        return cached

    V, K = prior.V, prior.K
    nbr_idx = prior.nbr_idx.to(device=device, dtype=torch.long)
    nbr_prob = prior.nbr_prob.to(device=device, dtype=torch.float32)

    # Flatten edges in row-major order: (i,k) for i=0..V-1, k=0..K-1
    i_flat = torch.arange(V, device=device, dtype=torch.int64).repeat_interleave(K)  # [V*K]
    k_flat = torch.arange(K, device=device, dtype=torch.int64).repeat(V)             # [V*K]
    j_flat = nbr_idx.reshape(-1).to(torch.int64)                                     # [V*K]
    p_flat = nbr_prob.reshape(-1).to(torch.float32)                                  # [V*K]

    edge_key = i_flat * int(V) + j_flat                                              # [V*K]
    order = torch.argsort(edge_key)
    edge_key_sorted = edge_key[order].contiguous()
    edge_prob_sorted = p_flat[order].contiguous()
    edge_pos_sorted = k_flat[order].contiguous()

    _EDGE_CACHE[key] = (edge_key_sorted, edge_prob_sorted, edge_pos_sorted)
    return edge_key_sorted, edge_prob_sorted, edge_pos_sorted

@torch.no_grad()
def _edge_lookup(
    edge_key_sorted: torch.Tensor,
    edge_prob_sorted: torch.Tensor,
    edge_pos_sorted: torch.Tensor,
    key_obs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Lookup observed edge keys in sorted edge table.

    Args:
      key_obs: [M] int64 keys (prev*V + nxt)

    Returns:
      hit:   [M] bool
      prob:  [M] float32  (topK prob if hit else 0)
      pos:   [M] int64    (neighbor slot if hit else 0)
    """
    # searchsorted returns insertion indices in [0, E]
    idx = torch.searchsorted(edge_key_sorted, key_obs)
    E = edge_key_sorted.numel()

    # Clamp idx to valid range for gather; mark hits safely
    idx_safe = idx.clamp(0, max(0, E - 1))
    hit = (idx < E) & (edge_key_sorted[idx_safe] == key_obs)

    prob = torch.zeros_like(key_obs, dtype=torch.float32)
    pos = torch.zeros_like(key_obs, dtype=torch.int64)
    if hit.any():
        hit_idx = idx_safe[hit]
        prob[hit] = edge_prob_sorted[hit_idx]
        pos[hit] = edge_pos_sorted[hit_idx]
    return hit, prob, pos


# =========================================================
# NLL/token under P' without dense P  (fast: O(M log(VK)))
# =========================================================
@torch.no_grad()
def nll_transition_sparse_teleport(
    x: torch.Tensor,
    prior: SparseTeleportPrior,
) -> float:
    device = x.device
    V, K = prior.V, prior.K  # keep for interface/consistency
    _ = K  # silence unused in some linters

    nu = _ensure_distribution(prior.nu).to(device=device, dtype=torch.float32)
    eps = float(prior.eps)

    prev = x[:, :-1].reshape(-1).to(torch.int64)  # [M]
    nxt = x[:, 1:].reshape(-1).to(torch.int64)    # [M]
    key_obs = prev * int(V) + nxt                 # [M] int64

    edge_key_sorted, edge_prob_sorted, edge_pos_sorted = _get_edge_table(prior, device=device)
    hit, local_prob, _pos = _edge_lookup(edge_key_sorted, edge_prob_sorted, edge_pos_sorted, key_obs)

    # Mixture prob: q = eps*nu(nxt) + (1-eps)*P_topk(nxt|prev) if hit else eps*nu(nxt)
    tp = (eps * nu[nxt]).clamp_min(NORM_CLAMP)  # [M]
    q = (tp + (1.0 - eps) * local_prob).clamp_min(NORM_CLAMP)
    nll = -torch.log(q).mean()
    return float(nll.item())


# =========================================================
# Restricted-bin metrics (fast, no [M,K] expand)
# =========================================================
@torch.no_grad()
def restricted_transition_metrics_sparse_teleport(
    x: torch.Tensor,
    prior: SparseTeleportPrior,
) -> Dict[str, float]:
    device = x.device
    N, T = x.shape
    V, K = prior.V, prior.K
    M = N * (T - 1)

    nbr_idx = prior.nbr_idx.to(device=device, dtype=torch.long)
    nbr_prob = prior.nbr_prob.to(device=device, dtype=torch.float32)
    nu = _ensure_distribution(prior.nu).to(device=device, dtype=torch.float32)
    eps = float(prior.eps)

    prev = x[:, :-1].reshape(-1).to(torch.int64)  # [M]
    nxt = x[:, 1:].reshape(-1).to(torch.int64)    # [M]
    key_obs = prev * int(V) + nxt                 # [M]

    edge_key_sorted, edge_prob_sorted, edge_pos_sorted = _get_edge_table(prior, device=device)
    hit, _local_prob, pos = _edge_lookup(edge_key_sorted, edge_prob_sorted, edge_pos_sorted, key_obs)

    matched = hit
    other_mass_rate = float((~matched).float().mean().item())

    counts_nbr = torch.zeros((V, K), device=device, dtype=torch.int64)
    counts_other = torch.zeros((V,), device=device, dtype=torch.int64)

    if matched.any():
        prev_m = prev[matched].to(torch.int64)
        pos_m = pos[matched].to(torch.int64)  # 0..K-1
        flat_idx = prev_m * int(K) + pos_m
        counts_flat = counts_nbr.view(-1)
        ones = torch.ones_like(flat_idx, dtype=torch.int64)
        counts_flat.scatter_add_(0, flat_idx, ones)

    if (~matched).any():
        prev_o = prev[~matched].to(torch.int64)
        ones = torch.ones_like(prev_o, dtype=torch.int64)
        counts_other.scatter_add_(0, prev_o, ones)

    row_tot = counts_nbr.sum(dim=1) + counts_other
    mask_rows = row_tot > 0
    if not mask_rows.any():
        return {
            "restricted_kl_rate": 0.0,
            "restricted_tv_rate": 0.0,
            "restricted_entropy_rate": 0.0,
            "other_mass_rate": other_mass_rate,
        }

    row_tot_f = row_tot.clamp_min(1).to(torch.float32)
    p_nbr = counts_nbr.to(torch.float32) / row_tot_f.unsqueeze(1)
    p_other = counts_other.to(torch.float32) / row_tot_f

    # GT on bins
    q_nbr = (1.0 - eps) * nbr_prob + eps * nu[nbr_idx]
    nu_on_nbr = nu[nbr_idx].sum(dim=1).clamp(0.0, 1.0)
    q_other = (eps * (1.0 - nu_on_nbr)).clamp_min(0.0)

    q_sum = (q_nbr.sum(dim=1) + q_other).clamp_min(NORM_CLAMP)
    q_nbr = q_nbr / q_sum.unsqueeze(1)
    q_other = q_other / q_sum

    w = (row_tot.to(torch.float32) / float(M))
    w = torch.where(mask_rows, w, torch.zeros_like(w))

    kl_nbr = torch.where(
        p_nbr > 0,
        p_nbr * (torch.log(p_nbr.clamp_min(NORM_CLAMP)) - torch.log(q_nbr.clamp_min(NORM_CLAMP))),
        torch.zeros_like(p_nbr),
    ).sum(dim=1)
    kl_other = torch.where(
        p_other > 0,
        p_other * (torch.log(p_other.clamp_min(NORM_CLAMP)) - torch.log(q_other.clamp_min(NORM_CLAMP))),
        torch.zeros_like(p_other),
    )
    restricted_kl_rate = float(((kl_nbr + kl_other) * w).sum().item())

    tv_nbr = torch.abs(p_nbr - q_nbr).sum(dim=1)
    tv_other = torch.abs(p_other - q_other)
    restricted_tv_rate = float((0.5 * (tv_nbr + tv_other) * w).sum().item())

    ent_nbr = torch.where(
        p_nbr > 0,
        -p_nbr * torch.log(p_nbr.clamp_min(NORM_CLAMP)),
        torch.zeros_like(p_nbr),
    ).sum(dim=1)
    ent_other = torch.where(
        p_other > 0,
        -p_other * torch.log(p_other.clamp_min(NORM_CLAMP)),
        torch.zeros_like(p_other),
    )
    restricted_entropy_rate = float(((ent_nbr + ent_other) * w).sum().item())

    return {
        "restricted_kl_rate": restricted_kl_rate,
        "restricted_tv_rate": restricted_tv_rate,
        "restricted_entropy_rate": restricted_entropy_rate,
        "other_mass_rate": other_mass_rate,
    }


# =========================================================
# Full metrics (fast: no [M,K] expand, no chunk loop)
# =========================================================
@torch.no_grad()
def full_kl_rate_sparse_teleport(
    x: torch.Tensor,
    prior: SparseTeleportPrior,
    *,
    chunk_edges: int = 200_000,  # kept for backward compatibility; no longer needed
) -> Dict[str, float]:
    """
    Exact FULL:
      KLrate = sum_i w_i KL(p_hat(.|i) || q(.|i))
      TVrate = sum_i w_i TV(p_hat(.|i), q(.|i))
    where q(.|i) = (1-eps) P_topk(.|i) + eps nu(.), and P_topk has support on K neighbors.

    Returns:
      full_kl_rate, full_tv_rate, full_entropy_rate, other_mass_rate, support_frac
    """
    device = x.device
    N, T = x.shape
    V, K = prior.V, prior.K
    M = N * (T - 1)
    _ = K
    _ = chunk_edges  # keep signature stable

    nu = _ensure_distribution(prior.nu.to(device=device, dtype=torch.float32))
    eps = float(prior.eps)

    prev = x[:, :-1].reshape(-1).to(torch.int64)  # [M]
    nxt = x[:, 1:].reshape(-1).to(torch.int64)    # [M]
    key_obs = prev * int(V) + nxt                 # [M]

    edge_key_sorted, edge_prob_sorted, edge_pos_sorted = _get_edge_table(prior, device=device)
    hit, _local_prob_all, _pos_all = _edge_lookup(edge_key_sorted, edge_prob_sorted, edge_pos_sorted, key_obs)
    other_mass_rate = float((~hit).float().mean().item())

    # Unique observed edges: key = i*V + j
    uniq_key, counts = torch.unique(key_obs, return_counts=True)  # [E], [E]
    counts = counts.to(torch.float32)
    E = int(uniq_key.numel())
    support_frac = float(E / max(1, M))

    i = (uniq_key // int(V)).to(torch.long)  # [E]
    j = (uniq_key % int(V)).to(torch.long)   # [E]

    # Row totals and weights
    row_tot = torch.zeros((V,), device=device, dtype=torch.float32)
    row_tot.scatter_add_(0, i, counts)
    w_row = row_tot / float(M)  # [V]

    # p_hat(j|i) on observed edges
    p_cond = (counts / row_tot[i].clamp_min(1.0)).clamp_min(NORM_CLAMP)  # [E]

    # teleport contribution for edges
    tp = (eps * nu[j]).clamp_min(NORM_CLAMP)  # [E]

    # local prob for uniq edges via edge lookup
    hit_u, local_u, _pos_u = _edge_lookup(edge_key_sorted, edge_prob_sorted, edge_pos_sorted, uniq_key.to(torch.int64))
    # local_u already 0 for miss
    q = (tp + (1.0 - eps) * local_u).clamp_min(NORM_CLAMP)  # [E]

    # ---- FULL KL-rate (exact) ----
    kl_edge = p_cond * (torch.log(p_cond) - torch.log(q))
    full_kl_rate = float((w_row[i] * kl_edge).sum().item())

    # ---- FULL entropy-rate of p_hat(.|i) ----
    ent_edge = -p_cond * torch.log(p_cond)
    full_entropy_rate = float((w_row[i] * ent_edge).sum().item())

    # ---- FULL TV-rate (exact) ----
    # For each row i:
    # L1_i = sum_{j in S_i} |p_hat - q| + (1 - sum_{j in S_i} q)
    abs_edge = torch.abs(p_cond - q)
    sum_abs = torch.zeros((V,), device=device, dtype=torch.float32)
    sum_qsup = torch.zeros((V,), device=device, dtype=torch.float32)
    sum_abs.scatter_add_(0, i, abs_edge)
    sum_qsup.scatter_add_(0, i, q)

    l1_row = sum_abs + (1.0 - sum_qsup).clamp_min(0.0)
    tv_row = 0.5 * l1_row
    full_tv_rate = float((w_row * tv_row).sum().item())

    return {
        "full_kl_rate": full_kl_rate,
        "full_entropy_rate": full_entropy_rate,
        "full_tv_rate": full_tv_rate,
        "other_mass_rate": other_mass_rate,
        "support_frac": support_frac,
    }

# =========================================================
# Prompt-aware metrics (NEW; no breaking changes)
#   - prompt_len = L
#   - distribution metrics evaluate suffix x[:, L:]
#   - transition metrics evaluate conditional continuation:
#       include boundary transition x_L -> x_{L+1}  by using x[:, L-1:]
# =========================================================

def _slice_suffix(x: torch.Tensor, prompt_len: int) -> torch.Tensor:
    L = int(prompt_len)
    if L <= 0:
        return x
    if L >= x.shape[1]:
        # empty suffix; return empty with correct shape
        return x[:, :0]
    return x[:, L:]


def _slice_transitions_for_conditional(x: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """
    For conditional continuation given a prompt of length L:
      - We want transitions starting at t=L (1-indexed), i.e. x_L -> x_{L+1}.
      - In 0-indexed tensor x[:, 0..T-1], that boundary transition is between
        x[:, L-1] and x[:, L].
    So we evaluate transitions on x[:, L-1:].
    """
    L = int(prompt_len)
    if L <= 0:
        return x
    if L == 0:
        return x
    if L >= x.shape[1]:
        return x[:, :0]
    return x[:, (L - 1):]


@torch.no_grad()
def unigram_l1_suffix(
    x: torch.Tensor,
    pi: torch.Tensor,
    V: int,
    *,
    prompt_len: int = 0,
) -> float:
    xs = _slice_suffix(x, prompt_len)
    if xs.numel() == 0:
        return float("nan")
    return unigram_l1(xs, pi=pi, V=V)


@torch.no_grad()
def unique_ngram_ratio_suffix(
    x: torch.Tensor,
    n: int,
    *,
    prompt_len: int = 0,
) -> float:
    xs = _slice_suffix(x, prompt_len)
    # If too short, define ratio as 0 (or nan). I suggest 0 for stability.
    if xs.shape[1] < n:
        return 0.0
    return unique_ngram_ratio(xs, n=n)


@torch.no_grad()
def dup_rate_suffix(
    x: torch.Tensor,
    *,
    prompt_len: int = 0,
) -> float:
    xs = _slice_suffix(x, prompt_len)
    if xs.numel() == 0:
        return float("nan")
    return dup_rate(xs)


@torch.no_grad()
def nll_transition_sparse_teleport_conditional(
    x: torch.Tensor,
    prior: SparseTeleportPrior,
    *,
    prompt_len: int = 0,
) -> float:
    xt = _slice_transitions_for_conditional(x, prompt_len)
    if xt.shape[1] < 2:
        return float("nan")
    return nll_transition_sparse_teleport(xt, prior)


@torch.no_grad()
def full_kl_rate_sparse_teleport_conditional(
    x: torch.Tensor,
    prior: SparseTeleportPrior,
    *,
    prompt_len: int = 0,
    chunk_edges: int = 200_000,
) -> Dict[str, float]:
    xt = _slice_transitions_for_conditional(x, prompt_len)
    if xt.shape[1] < 2:
        return {
            "full_kl_rate": float("nan"),
            "full_entropy_rate": float("nan"),
            "full_tv_rate": float("nan"),
            "other_mass_rate": float("nan"),
            "support_frac": float("nan"),
        }
    return full_kl_rate_sparse_teleport(xt, prior, chunk_edges=chunk_edges)


@torch.no_grad()
def restricted_transition_metrics_sparse_teleport_conditional(
    x: torch.Tensor,
    prior: SparseTeleportPrior,
    *,
    prompt_len: int = 0,
) -> Dict[str, float]:
    xt = _slice_transitions_for_conditional(x, prompt_len)
    if xt.shape[1] < 2:
        return {
            "restricted_kl_rate": float("nan"),
            "restricted_tv_rate": float("nan"),
            "restricted_entropy_rate": float("nan"),
            "other_mass_rate": float("nan"),
        }
    return restricted_transition_metrics_sparse_teleport(xt, prior)
