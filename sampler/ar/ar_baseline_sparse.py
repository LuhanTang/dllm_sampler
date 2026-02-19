# sampler/ar_baseline_sparse.py
from __future__ import annotations

import torch

# Keep consistent with your run.py / metrics numerics
NORM_CLAMP = 1e-30


def _ensure_distribution(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(0.0)
    return p / p.sum().clamp_min(NORM_CLAMP)


@torch.no_grad()
def sample_ar_sparse_teleport(
    *,
    pi: torch.Tensor,                  # [V]
    prior,                             # SparseTeleportPrior (duck-typed)
    N: int,
    T: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample x_0 ~ pi, then x_{t+1} ~ P'(x_t -> ·).

    P'(i->j) = (1-eps) * P_topk(i->j) + eps * nu(j)

    Sampling scheme per step:
      - with prob eps: sample from nu
      - else: sample from categorical over nbr_prob[row] (over K neighbors only)

    Complexity: O(N*T*K) (via multinomial over K only).
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    pi = _ensure_distribution(pi).to(device=device, dtype=torch.float32)
    nu = _ensure_distribution(prior.nu).to(device=device, dtype=torch.float32)
    nbr_idx = prior.nbr_idx.to(device=device, dtype=torch.long)
    nbr_prob = prior.nbr_prob.to(device=device, dtype=torch.float32)

    x = torch.empty((N, T), device=device, dtype=torch.long)

    # x0
    x0 = torch.multinomial(pi, num_samples=N, replacement=True)
    x[:, 0] = x0

    eps = float(prior.eps)
    for t in range(T - 1):
        cur = x[:, t]  # [N]
        u = torch.rand((N,), device=device)
        do_tp = (u < eps)

        nxt = torch.empty((N,), device=device, dtype=torch.long)

        # teleport
        if do_tp.any():
            nxt_tp = torch.multinomial(nu, num_samples=int(do_tp.sum().item()), replacement=True)
            nxt[do_tp] = nxt_tp

        # local top-k
        if (~do_tp).any():
            cur2 = cur[~do_tp]  # [n2]
            probs2 = nbr_prob[cur2]  # [n2,K]
            k_idx = torch.multinomial(probs2, num_samples=1, replacement=True).squeeze(1)  # [n2]
            nxt2 = nbr_idx[cur2, k_idx]  # [n2]
            nxt[~do_tp] = nxt2

        x[:, t + 1] = nxt

    return x
