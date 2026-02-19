# sampler/gt_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

NORM_CLAMP = 1e-30
DEFAULT_TELEPORT_EPS = 1e-4
DEFAULT_STATIONARY_ITERS = 4000
DEFAULT_STATIONARY_TOL = 1e-12

NINF = -1e30  # fp16-safe "minus inf"


# =========================================================
# Data containers
# =========================================================
@dataclass
class Text8CharGT:
    path: str
    config: Dict[str, Any]
    vocab: Optional[List[str]]
    stoi: Optional[Dict[str, int]]
    itos: Optional[Dict[int, str]]

    # ---- Sparse teleport target (preferred) ----
    # P'(i->j) = (1-eps)*P_topk(i->j) + eps*nu(j)
    nbr_idx: torch.Tensor   # [V,K] long
    nbr_prob: torch.Tensor  # [V,K] float32, row-stochastic over K
    nu: torch.Tensor        # [V] float32, sums to 1
    eps: float              # scalar
    pi: torch.Tensor        # [V] float32, stationary of P'

    # NEW: explicit empirical unigram distribution (from corpus)
    unigram: Optional[torch.Tensor] = None  # [V] or None

    # optional cached samples from (pi, P')
    samples: Optional[torch.Tensor] = None  # [N,T] or None

    # ---- Optional dense stuff (only for small V / debugging) ----
    P: Optional[torch.Tensor] = None         # [V,V] if available
    logP_safe: Optional[torch.Tensor] = None # [V,V] if available

    @property
    def V(self) -> int:
        return int(self.nbr_idx.shape[0])

    @property
    def K(self) -> int:
        return int(self.nbr_idx.shape[1])

    @property
    def N(self) -> int:
        return 0 if self.samples is None else int(self.samples.shape[0])

    @property
    def T(self) -> int:
        return 0 if self.samples is None else int(self.samples.shape[1])


# =========================================================
# Helpers
# =========================================================
@torch.no_grad()
def _ensure_distribution(p: torch.Tensor) -> torch.Tensor:
    p = p.to(torch.float32).clamp_min(0.0)
    return p / p.sum().clamp_min(NORM_CLAMP)


@torch.no_grad()
def _row_normalize_dense(P: torch.Tensor) -> torch.Tensor:
    P = P.to(torch.float32).clamp_min(0.0)
    s = P.sum(dim=1, keepdim=True)
    bad = s < 1e-12
    if bad.any():
        V = P.shape[0]
        P = torch.where(bad, torch.full_like(P, 1.0 / V), P)
        s = P.sum(dim=1, keepdim=True)
    return P / s.clamp_min(NORM_CLAMP)


@torch.no_grad()
def _build_topk_from_dense(P: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    V = P.shape[0]
    K = int(max(1, min(int(K), V)))
    vals, idx = torch.topk(P, k=K, dim=1)
    vals = vals.clamp_min(0.0)
    vals = vals / vals.sum(dim=1, keepdim=True).clamp_min(NORM_CLAMP)
    return idx.to(torch.long), vals.to(torch.float32)


@torch.no_grad()
def _power_iteration_stationary_sparse(
    nbr_idx: torch.Tensor,
    nbr_prob: torch.Tensor,
    nu: torch.Tensor,
    eps: float,
    iters: int,
    tol: float,
) -> torch.Tensor:
    device = nbr_idx.device
    V, K = nbr_idx.shape
    eps = float(eps)
    nu = _ensure_distribution(nu.to(device))

    pi = torch.full((V,), 1.0 / V, device=device, dtype=torch.float64)
    nbr_prob64 = nbr_prob.to(device=device, dtype=torch.float64)
    nu64 = nu.to(torch.float64)

    for _ in range(int(iters)):
        contrib = (pi[:, None] * nbr_prob64)  # [V,K]
        pi_next = torch.zeros((V,), device=device, dtype=torch.float64)
        pi_next.scatter_add_(0, nbr_idx.reshape(-1), contrib.reshape(-1))

        pi_next = (1.0 - eps) * pi_next + eps * nu64
        pi_next = pi_next / pi_next.sum().clamp_min(NORM_CLAMP)

        if torch.max(torch.abs(pi_next - pi)).item() < tol:
            pi = pi_next
            break
        pi = pi_next

    return pi.to(torch.float32)


@torch.no_grad()
def _sample_ar_sparse_teleport(
    pi: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_prob: torch.Tensor,
    nu: torch.Tensor,
    eps: float,
    N: int,
    T: int,
    seed: int,
) -> torch.Tensor:
    device = nbr_idx.device
    gen_dev = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_dev)
    g.manual_seed(int(seed))


    V, K = nbr_idx.shape
    pi = _ensure_distribution(pi.to(device))
    nu = _ensure_distribution(nu.to(device))
    eps = float(eps)

    x = torch.empty((N, T), dtype=torch.long, device=device)
    cur = torch.multinomial(pi, num_samples=N, replacement=True, generator=g)
    x[:, 0] = cur

    for t in range(1, T):
        u = torch.rand((N,), device=device, generator=g)
        do_tp = (u < eps)
        nxt = torch.empty((N,), device=device, dtype=torch.long)

        if do_tp.any():
            nxt_tp = torch.multinomial(nu, num_samples=int(do_tp.sum().item()), replacement=True, generator=g)
            nxt[do_tp] = nxt_tp

        if (~do_tp).any():
            cur2 = cur[~do_tp]
            probs2 = nbr_prob[cur2]  # [n2,K]
            k_idx = torch.multinomial(probs2, num_samples=1, replacement=True, generator=g).squeeze(1)
            nxt2 = nbr_idx[cur2, k_idx]
            nxt[~do_tp] = nxt2

        x[:, t] = nxt
        cur = nxt

    return x


def _maybe_read_topk(config: Dict[str, Any], V: int) -> int:
    k = config.get("topk", None)
    if k is None:
        return V
    try:
        k = int(k)
    except Exception:
        return V
    return int(max(1, min(k, V)))


def _maybe_read_nu(config: Dict[str, Any], V: int, device: torch.device) -> torch.Tensor:
    for key in ["teleport_nu", "nu"]:
        if key in config:
            try:
                nu = torch.as_tensor(config[key], device=device, dtype=torch.float32)
                if nu.numel() == V:
                    return _ensure_distribution(nu)
            except Exception:
                pass
    return torch.full((V,), 1.0 / V, device=device, dtype=torch.float32)


def _maybe_read_unigram(obj: Dict[str, Any], dev: torch.device, V: int) -> Optional[torch.Tensor]:
    """
    Prefer artifact["unigram"] (explicit). If absent, try to recover from:
      - if config says nu == "unigram" and artifact["nu"] exists, use nu as unigram proxy.
    Otherwise return None.
    """
    uni = obj.get("unigram", None)
    if isinstance(uni, torch.Tensor):
        uni = uni.to(dev).to(torch.float32)
        if uni.numel() == V:
            return _ensure_distribution(uni)

    # fallback proxy
    cfg = obj.get("config", {})
    try:
        nu_mode = (cfg.get("nu", "") if isinstance(cfg, dict) else "")
    except Exception:
        nu_mode = ""

    nu = obj.get("nu", None)
    if nu_mode == "unigram" and isinstance(nu, torch.Tensor):
        nu = nu.to(dev).to(torch.float32)
        if nu.numel() == V:
            return _ensure_distribution(nu)

    return None


# =========================================================
# Main loader
# =========================================================
def load_gt(
    path: str,
    device: str = "cpu",
    *,
    teleport_eps: float = DEFAULT_TELEPORT_EPS,
    teleport_nu: Optional[torch.Tensor] = None,
    topk: Optional[int] = None,
    stationary_iters: int = DEFAULT_STATIONARY_ITERS,
    stationary_tol: float = DEFAULT_STATIONARY_TOL,
    resample_seed: int = 0,
    cache_write: bool = True,
    write_sparse_cache_when_possible: bool = True,
) -> Text8CharGT:
    obj: Dict[str, Any] = torch.load(path, map_location="cpu")
    required_meta = ["config"]
    for k in required_meta:
        if k not in obj:
            raise KeyError(f"GT artifact missing key: {k}")
    vocab = obj.get("vocab", None)
    stoi  = obj.get("stoi", None)
    itos  = obj.get("itos", None)

    dev = torch.device(device)
    config: Dict[str, Any] = obj["config"] if isinstance(obj["config"], dict) else {}

    # ---------------------------------------------------------
    # 1) Fast path: artifact already has sparse teleport target
    # ---------------------------------------------------------
    if all(k in obj for k in ["nbr_idx", "nbr_prob", "nu", "eps", "pi"]):
        nbr_idx = obj["nbr_idx"].to(dev).to(torch.long)
        nbr_prob = obj["nbr_prob"].to(dev).to(torch.float32)
        nu = obj["nu"].to(dev).to(torch.float32)
        eps = float(obj["eps"])
        pi = obj["pi"].to(dev).to(torch.float32)

        V, K = nbr_idx.shape
        if nbr_prob.shape != (V, K):
            raise ValueError(f"nbr_prob shape mismatch: got {tuple(nbr_prob.shape)} vs {(V,K)}")

        nu = _ensure_distribution(nu)
        pi = _ensure_distribution(pi)

        nbr_prob = nbr_prob.clamp_min(0.0)
        nbr_prob = nbr_prob / nbr_prob.sum(dim=1, keepdim=True).clamp_min(NORM_CLAMP)

        unigram = _maybe_read_unigram(obj, dev=dev, V=V)

        samples = obj.get("samples", None)
        if isinstance(samples, torch.Tensor):
            samples = samples.to(dev).to(torch.long)

        P = obj.get("P", None)
        logP_safe = obj.get("logP_safe", None)
        if isinstance(P, torch.Tensor):
            P = P.to(dev).to(torch.float32)
        if isinstance(logP_safe, torch.Tensor):
            logP_safe = logP_safe.to(dev).to(torch.float32)

        return Text8CharGT(
            path=path,
            config=config,
            vocab=obj["vocab"],
            stoi=obj["stoi"],
            itos=obj["itos"],
            nbr_idx=nbr_idx,
            nbr_prob=nbr_prob,
            nu=nu,
            eps=eps,
            pi=pi,
            unigram=unigram,
            samples=samples,
            P=P,
            logP_safe=logP_safe,
        )

    # ---------------------------------------------------------
    # 2) Fallback: build sparse teleport from dense P (small V)
    # ---------------------------------------------------------
    if "P" not in obj:
        raise KeyError("GT artifact has no sparse fields and no dense P. Need either (nbr_idx/nbr_prob/nu/eps/pi) or P.")

    P_raw = obj["P"].to(dev).to(torch.float32)
    if P_raw.dim() != 2 or P_raw.shape[0] != P_raw.shape[1]:
        raise ValueError(f"P must be square [V,V], got {tuple(P_raw.shape)}")
    V = int(P_raw.shape[0])

    unigram = _maybe_read_unigram(obj, dev=dev, V=V)

    K = _maybe_read_topk(config, V) if topk is None else int(max(1, min(int(topk), V)))
    if teleport_nu is None:
        nu = _maybe_read_nu(config, V=V, device=dev)
    else:
        nu = _ensure_distribution(teleport_nu.to(dev))
        if nu.numel() != V:
            raise ValueError(f"teleport_nu must have shape [V]={V}, got {tuple(nu.shape)}")

    eps = float(max(0.0, min(0.999999, float(teleport_eps))))

    P_base = _row_normalize_dense(P_raw)
    nbr_idx, nbr_prob = _build_topk_from_dense(P_base, K=K)

    pi = _power_iteration_stationary_sparse(
        nbr_idx=nbr_idx.to(dev),
        nbr_prob=nbr_prob.to(dev),
        nu=nu.to(dev),
        eps=eps,
        iters=stationary_iters,
        tol=stationary_tol,
    )

    if "samples" in obj and isinstance(obj["samples"], torch.Tensor) and obj["samples"].dim() == 2:
        N, T = int(obj["samples"].shape[0]), int(obj["samples"].shape[1])
    else:
        raise KeyError("Dense-only artifact must contain samples [N,T] so loader knows N,T (or store N,T in config).")

    samples = _sample_ar_sparse_teleport(
        pi=pi,
        nbr_idx=nbr_idx.to(dev),
        nbr_prob=nbr_prob.to(dev),
        nu=nu.to(dev),
        eps=eps,
        N=N,
        T=T,
        seed=resample_seed,
    )

    P_prime_dense = None
    logP_safe_dense = None

    if cache_write and write_sparse_cache_when_possible:
        obj["nbr_idx"] = nbr_idx.detach().cpu()
        obj["nbr_prob"] = nbr_prob.detach().cpu()
        obj["nu"] = nu.detach().cpu()
        obj["eps"] = float(eps)
        obj["pi"] = pi.detach().cpu()
        obj["samples"] = samples.detach().cpu()
        if unigram is not None:
            obj["unigram"] = unigram.detach().cpu()
        try:
            torch.save(obj, path)
        except Exception:
            pass

    return Text8CharGT(
        path=path,
        config=config,
        vocab=obj["vocab"],
        stoi=obj["stoi"],
        itos=obj["itos"],
        nbr_idx=nbr_idx.to(dev),
        nbr_prob=nbr_prob.to(dev),
        nu=nu.to(dev),
        eps=eps,
        pi=pi.to(dev),
        unigram=unigram.to(dev) if isinstance(unigram, torch.Tensor) else None,
        samples=samples.to(dev),
        P=P_prime_dense,
        logP_safe=logP_safe_dense,
    )
