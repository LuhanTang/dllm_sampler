# sampler/remdm/updates_from_diffusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

# --- copied behavior from diffusion.py
def _sample_categorical(categorical_probs: Tensor) -> Tensor:
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@dataclass
class ReMDMSamplerConfig:
    """
    Minimal config fields used by diffusion.py branches.
    """
    sampler: str                 # 'remdm-rescale' | 'remdm-conf' | 'remdm-loop' | ...
    eta: float = 1.0             # used by remdm-rescale / remdm-loop / remdm-cap
    # remdm-loop params:
    t_on: float = 0.8
    t_off: float = 0.2
    alpha_on: float = 0.5


@torch.no_grad()
def ddpm_caching_update_from_diffusion(
    *,
    x: Tensor,                   # [B,L] int in {0..V} where V is mask_index
    t: Tensor,                   # [B,1] float in (0,1]
    dt: float,
    p_x0: Tensor,                # [B,L,V+1] probs (already includes mask dim)
    mask_index: int,             # V
    cfg: ReMDMSamplerConfig,
    conf: Optional[Tensor] = None,  # [B,L] used only by remdm-conf/loop
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    A minimal pure-function version of Diffusion._ddpm_caching_update
    that exactly matches the formulas in your diffusion.py for the
    relevant sampler branches.

    Returns:
      xs: [B,L]
      conf: updated conf (if used), else unchanged
    """
    assert t.ndim == 2 and t.shape[1] == 1
    B, L = x.shape
    Vp1 = p_x0.shape[-1]
    assert Vp1 == mask_index + 1, (Vp1, mask_index)

    # move_chance_t = t, move_chance_s = t - dt  (as in diffusion.py)
    move_chance_t = t[:, None, None]            # [B,1,1] then broadcast
    move_chance_s = (t - dt)[:, None, None]

    # NOTE: diffusion.py uses alpha_t = (1 - move_chance_t)[0].item() etc (scalar per step)
    alpha_t = (1 - move_chance_t)[0].item()
    alpha_s = (1 - move_chance_s)[0].item()

    # ---- branches ----
    if cfg.sampler == "remdm-rescale":
        if alpha_t > 0:
            sigma_max = min(1.0, (1.0 - alpha_s) / alpha_t)
        else:
            sigma_max = 1.0
        sigma = cfg.eta * sigma_max

        q_xs = p_x0 * (1 - sigma)
        q_xs[..., mask_index] = sigma

        q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
        q_xs_2[..., mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)

        copy_flag = (x != mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q_xs)

        return xs, conf

    if cfg.sampler == "remdm-conf":
        assert conf is not None, "remdm-conf needs conf tensor [B,L]"
        if alpha_t > 0:
            sigma_max = min(1.0, (1.0 - alpha_s) / alpha_t)
        else:
            sigma_max = 1.0

        eta = conf.softmax(dim=-1)              # [B,L]
        masked_flag = (x == mask_index).to(torch.bool)
        eta[masked_flag] = 0
        sigma = eta * sigma_max                 # [B,L]

        q_xs = p_x0 * (1 - sigma[:, :, None])
        q_xs[..., mask_index] = sigma

        q_xs_2 = p_x0 * ((alpha_s - (1 - sigma[:, :, None]) * alpha_t) / (1 - alpha_t))
        q_xs_2[..., mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)

        copy_flag = (x != mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q_xs)

        # ---- update conf (exactly as diffusion.py) ----
        # unmask_mask: previously masked, now unmasked
        unmask_mask = (x == mask_index) & (xs != mask_index)
        batch_indices = torch.arange(xs.shape[0], device=xs.device)[:, None]
        feature_indices = torch.arange(xs.shape[1], device=xs.device)[None, :]

        # conf_values = - p_x0[batch, pos, xs]
        xs_clamped = xs.clamp(0, mask_index)  # includes mask idx
        conf_values = -p_x0[batch_indices, feature_indices, xs_clamped]
        conf[unmask_mask] = conf_values[unmask_mask]

        remask_mask = (x != mask_index) & (xs == mask_index)
        conf[remask_mask] = -torch.inf

        return xs, conf

    if cfg.sampler == "remdm-loop":
        # This branch mixes MDLM-like updates at early/late times, and ReMDM in the middle.
        time = t[0].item()

        # compute adjusted move_chance_t and move_chance_s
        if time > cfg.t_on:
            move_chance_t2 = (1 - (1 - t) * cfg.alpha_on / (1 - cfg.t_on))[:, None, None]
            move_chance_s2 = (1 - (1 - t + dt) * cfg.alpha_on / (1 - cfg.t_on))[:, None, None]
        elif time <= cfg.t_off:
            move_chance_t2 = (t * (1 - cfg.alpha_on) / cfg.t_off)[:, None, None]
            move_chance_s2 = ((t - dt) * (1 - cfg.alpha_on) / cfg.t_off)[:, None, None]
        else:
            move_chance_t2, move_chance_s2 = None, None

        # use MDLM region
        if time > cfg.t_on or time <= cfg.t_off:
            q_xs = p_x0 * (move_chance_t2 - move_chance_s2)
            q_xs[:, :, mask_index] = move_chance_s2[:, :, 0]
            _x = _sample_categorical(q_xs)
            copy_flag = (x != mask_index).to(x.dtype)
            xs = copy_flag * x + (1 - copy_flag) * _x
            return xs, conf

        # use ReMDM region (middle)
        sigma = cfg.eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., mask_index] = sigma

        q_xs_2 = p_x0 * ((cfg.alpha_on - (1 - sigma) * cfg.alpha_on) / (1 - cfg.alpha_on))
        q_xs_2[..., mask_index] = (1 - cfg.alpha_on - cfg.alpha_on * sigma) / (1 - cfg.alpha_on)

        copy_flag = (x != mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q_xs)
        return xs, conf

    raise ValueError(f"Unsupported sampler: {cfg.sampler}")
