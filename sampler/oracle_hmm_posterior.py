# sampler/oracle_hmm_posterior.py
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

NORM_CLAMP = 1e-30
NINF = -1e30  # fp16-safe -inf


def lse(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim)


def _ensure_distribution(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(0.0)
    return p / p.sum().clamp_min(NORM_CLAMP)


@dataclass
class SparseTeleportPrior:
    nbr_idx: torch.Tensor   # [V,K] long
    nbr_prob: torch.Tensor  # [V,K] float, row-normalized over K neighbors
    nu: torch.Tensor        # [V]
    eps: float

    @property
    def V(self) -> int:
        return int(self.nbr_idx.shape[0])

    @property
    def K(self) -> int:
        return int(self.nbr_idx.shape[1])


class OracleHMMPosterior_LogRank1Teleport(torch.nn.Module):
    """
    Log-domain forward-backward with exact rank-1 teleport mixture:
      P' = (1-eps) P_topk + eps * 1 nu^T

    Hard evidence:
      - if z_t == mask_id: phi_t(i)=1 for all i
      - else: phi_t(z_t)=1, others 0

    Output: p_x0 [B,T,V]
      - unmasked positions: one-hot(z_t)
      - masked positions: gamma_t from FB

    IMPORTANT:
      - eps may be 0 (no teleport). In that case we use P' = P_topk exactly.
    """
    def __init__(
        self,
        prior: SparseTeleportPrior,
        pi0: torch.Tensor,
        mask_id: int,
        store_dtype: torch.dtype = torch.float16,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.mask_id = int(mask_id)
        self.store_dtype = store_dtype
        self.compute_dtype = compute_dtype

        # ---- cache prior on buffers (module.to(device) will move them) ----
        nbr_idx = prior.nbr_idx.to(torch.long)
        self.register_buffer("nbr_idx", nbr_idx)  # [V,K]

        # IMPORTANT: row-normalize over K (not global normalize!)
        nbr_prob = prior.nbr_prob.to(torch.float32).clamp_min(0.0)
        nbr_prob = nbr_prob / nbr_prob.sum(dim=1, keepdim=True).clamp_min(NORM_CLAMP)
        self.register_buffer("nbr_prob", nbr_prob)  # [V,K]

        logP_topk = torch.log(nbr_prob.clamp_min(NORM_CLAMP)).to(torch.float32)
        self.register_buffer("logP_topk", logP_topk)  # [V,K]

        # Pre-flatten neighbor indices for scatter_add (avoid expand/reshape each step)
        self.register_buffer("nbr_idx_flat", nbr_idx.reshape(-1))  # [V*K]

        pi0 = _ensure_distribution(pi0).to(dtype=compute_dtype)
        self.register_buffer("pi0", pi0)

        nu = _ensure_distribution(prior.nu).to(dtype=compute_dtype)
        self.register_buffer("nu", nu)
        self.register_buffer("log_nu", torch.log(nu.clamp_min(NORM_CLAMP)))

        # ------------------------------------------------------------
        # Teleport handling (allow eps==0)
        # ------------------------------------------------------------
        eps = float(prior.eps)

        # clamp only for numeric sanity on the upper side;
        # keep exact 0.0 if provided (we want "no teleport" semantics)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.999999:
            eps = 0.999999

        self.eps = eps
        self.use_teleport = (eps > 0.0)

        # store as python floats (fast; broadcast works fine)
        if not self.use_teleport:
            # P' = P (no teleport)
            self.log_1m_eps = 0.0            # log(1)
            self.log_eps = float("-inf")     # never used in mixture
        else:
            self.log_1m_eps = math.log(1.0 - eps)
            self.log_eps = math.log(eps)

    @torch.no_grad()
    def _hard_evidence_logphi(self, obs_t: torch.Tensor, V: int) -> torch.Tensor:
        # returns [B,V]
        B = obs_t.shape[0]
        logphi = torch.zeros((B, V), device=obs_t.device, dtype=self.compute_dtype)
        not_mask = (obs_t != self.mask_id)
        if not_mask.any():
            logphi[not_mask] = NINF
            logphi[not_mask, obs_t[not_mask]] = 0.0
        return logphi

    @torch.no_grad()
    def _forward_sparse_s(self, alpha_prob: torch.Tensor) -> torch.Tensor:
        """
        s(j)=sum_i alpha(i) P_topk(i->j) using outgoing edges + scatter_add.  O(B*V*K)
        """
        B, V = alpha_prob.shape

        nbr_prob = self.nbr_prob.to(dtype=self.compute_dtype)  # [V,K]
        contrib = alpha_prob[:, :, None] * nbr_prob[None, :, :]  # [B,V,K]

        s = torch.zeros((B, V), device=alpha_prob.device, dtype=self.compute_dtype)

        idx = self.nbr_idx_flat[None, :].expand(B, -1)  # [B,V*K]
        s.scatter_add_(dim=1, index=idx, src=contrib.reshape(B, -1))
        return s

    @torch.no_grad()
    def _backward_sparse_log_a(self, log_w: torch.Tensor) -> torch.Tensor:
        """
        log a(i) = LSE_k ( logP(i,k) + log_w(nbr[i,k]) ).  O(B*V*K)
        """
        nbr_idx = self.nbr_idx  # [V,K]
        logP = self.logP_topk.to(dtype=self.compute_dtype)  # [V,K]

        logw_nbr = log_w[:, nbr_idx]  # [B,V,K]
        terms = logP[None, :, :] + logw_nbr
        return torch.logsumexp(terms, dim=2)  # [B,V]

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        B, T = z.shape
        V = int(self.nbr_idx.shape[0])
        assert self.mask_id == V, f"mask_id should be V={V} for absorbing mask"

        # ---------- forward: store log_alpha_t ----------
        log_alpha_all = torch.empty((B, T, V), device=device, dtype=self.store_dtype)

        logphi0 = self._hard_evidence_logphi(z[:, 0], V)
        logpi = torch.log(self.pi0.clamp_min(NORM_CLAMP))[None, :].expand(B, -1)
        loga0 = logpi + logphi0
        loga0 = loga0 - lse(loga0, dim=1).unsqueeze(1)
        log_alpha_all[:, 0] = loga0.to(self.store_dtype)

        for t in range(1, T):
            alpha_prev = torch.exp(log_alpha_all[:, t - 1].to(self.compute_dtype))  # [B,V]
            s = self._forward_sparse_s(alpha_prev)  # [B,V]
            log_s = torch.log(s.clamp_min(NORM_CLAMP))  # [B,V]

            # ---- P' = (1-eps) P + eps nu ; eps=0 => P' = P ----
            if not self.use_teleport:
                log_tilde = log_s
            else:
                log_tilde = torch.logsumexp(
                    torch.stack(
                        [
                            log_s + self.log_1m_eps,
                            (self.log_nu[None, :] + self.log_eps).expand(B, -1),
                        ],
                        dim=0,
                    ),
                    dim=0,
                )  # [B,V]

            logphi = self._hard_evidence_logphi(z[:, t], V)
            loga = logphi + log_tilde
            loga = loga - lse(loga, dim=1).unsqueeze(1)
            log_alpha_all[:, t] = loga.to(self.store_dtype)

        # ---------- backward + posterior ----------
        log_beta = torch.zeros((B, V), device=device, dtype=self.compute_dtype)

        is_mask = (z == self.mask_id)
        p_x0 = torch.zeros((B, T, V), device=device, dtype=self.compute_dtype)

        # unmasked -> onehot
        if (~is_mask).any():
            z_clamped = z.clamp_max(V - 1)
            onehot = F.one_hot(z_clamped, num_classes=V).to(self.compute_dtype)
            p_x0 = torch.where((~is_mask).unsqueeze(-1), onehot, p_x0)

        def write_masked(t: int, log_beta_t: torch.Tensor) -> None:
            m = is_mask[:, t]
            if not m.any():
                return
            log_alpha_t = log_alpha_all[:, t].to(self.compute_dtype)
            log_post = log_alpha_t + log_beta_t
            log_post = log_post - lse(log_post, dim=1).unsqueeze(1)
            gamma = torch.exp(log_post)
            p_x0[:, t, :] = torch.where(m.unsqueeze(-1), gamma, p_x0[:, t, :])

        write_masked(T - 1, log_beta)

        for t in range(T - 2, -1, -1):
            logphi_next = self._hard_evidence_logphi(z[:, t + 1], V)
            log_w = logphi_next + log_beta  # [B,V]

            log_a = self._backward_sparse_log_a(log_w)  # [B,V]

            # ---- eps=0 => beta update is just log_a (no teleport) ----
            if not self.use_teleport:
                log_beta_new = log_a
            else:
                log_c = lse(self.log_nu[None, :] + log_w, dim=1)  # [B]
                log_beta_new = torch.logaddexp(
                    log_a + self.log_1m_eps,
                    (log_c[:, None] + self.log_eps).expand(B, V),
                )

            # stabilize
            log_beta_new = log_beta_new - lse(log_beta_new, dim=1).unsqueeze(1)

            log_beta = log_beta_new
            write_masked(t, log_beta)

        p_x0 = p_x0.clamp_min(0.0)
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(NORM_CLAMP)
        return p_x0
