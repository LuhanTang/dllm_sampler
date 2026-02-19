# sampler/ar_baseline_sparse_temp.py
from __future__ import annotations

import math
import torch

# Keep consistent with your run.py / metrics numerics
NORM_CLAMP = 1e-30


def _ensure_distribution(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(0.0)
    return p / p.sum().clamp_min(NORM_CLAMP)


@torch.no_grad()
def sample_ar_sparse_teleport_temp(
    *,
    pi: torch.Tensor,                  # [V]
    prior,                             # SparseTeleportPrior (duck-typed)
    N: int,
    T: int,
    seed: int,
    device: torch.device,
    tau: float = 1.0,                  # temperature on LOCAL top-k transitions only
    eps_teleport_override: float | None = None,  # optional: override prior.eps
) -> torch.Tensor:
    """
    Sample x_0 ~ pi, then x_{t+1} ~ P'(x_t -> ·).

    P'(i->j) = (1-eps) * P_topk(i->j) + eps * nu(j)

    Temperature is applied ONLY to the local top-k kernel:
        p_tau(j|i) ∝ p(j|i)^(1/tau)  (equivalently logp / tau then softmax)

    - tau = 1.0: original sampling
    - tau < 1.0: sharper / lower temperature (more greedy / concentrated)
    - tau > 1.0: flatter

    Teleport component nu is left unchanged (recommended to avoid confounding).

    Complexity: O(N*T*K)
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    pi = _ensure_distribution(pi).to(device=device, dtype=torch.float32)
    nu = _ensure_distribution(prior.nu).to(device=device, dtype=torch.float32)

    nbr_idx = prior.nbr_idx.to(device=device, dtype=torch.long)          # [V,K]
    nbr_prob = prior.nbr_prob.to(device=device, dtype=torch.float32)     # [V,K]

    V, K = nbr_idx.shape
    x = torch.empty((N, T), device=device, dtype=torch.long)

    # x0
    x[:, 0] = torch.multinomial(pi, num_samples=N, replacement=True)

    eps = float(prior.eps) if eps_teleport_override is None else float(eps_teleport_override)
    eps = max(0.0, min(1.0, eps))

    # Precompute a safe flag: if tau==1, we can skip extra softmax work
    use_temp = (abs(tau - 1.0) > 1e-12)

    for t in range(T - 1):
        cur = x[:, t]  # [N]
        u = torch.rand((N,), device=device)
        do_tp = (u < eps)

        nxt = torch.empty((N,), device=device, dtype=torch.long)

        # teleport
        if do_tp.any():
            n_tp = int(do_tp.sum().item())
            nxt_tp = torch.multinomial(nu, num_samples=n_tp, replacement=True)
            nxt[do_tp] = nxt_tp

        # local top-k
        if (~do_tp).any():
            cur2 = cur[~do_tp]               # [n2]
            probs2 = nbr_prob[cur2]          # [n2,K] row-normalized over K

            if use_temp:
                # temperature reweight on LOCAL top-k distribution only:
                # p_tau ∝ p^(1/tau)  (stable: log -> /tau -> softmax)
                logp = torch.log(probs2.clamp_min(NORM_CLAMP))     # [n2,K]
                logits = logp / tau                                # [n2,K]
                probs2 = torch.softmax(logits, dim=-1)             # [n2,K]

            k_idx = torch.multinomial(probs2, num_samples=1, replacement=True).squeeze(1)  # [n2]
            nxt2 = nbr_idx[cur2, k_idx]  # [n2]
            nxt[~do_tp] = nxt2

        x[:, t + 1] = nxt

    return x
