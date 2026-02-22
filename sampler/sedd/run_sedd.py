from __future__ import annotations

import argparse
import os
import json
import csv
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Local modules
from sampler.gt_io import load_gt

# --- AR baseline sparse
from sampler.ar_baseline_sparse import sample_ar_sparse_teleport

# --- SEDD bits
from sampler.sedd.graph_lib import Absorbing
from sampler.sedd.noise_lib import LogLinearNoise
from sampler.sedd.sampling import get_pc_sampler

# --- Metrics
from sampler.metrics_full import (
    SparseTeleportPrior,
    unigram_l1,
    unique_ngram_ratio,
    dup_rate,
    top_unigrams_bigrams_print,
    nll_transition_sparse_teleport,
    full_kl_rate_sparse_teleport,
)

# --- utils_io
from sampler.utils_io import parse_steps, _fmt_float_tag, _sanitize

# =========================================================
# Optional decoder (HF tokenizers)
# =========================================================
try:
    from tokenizers import Tokenizer as HFTokenizer
except Exception:
    HFTokenizer = None

# =========================================================
# Numerics
# =========================================================
NORM_CLAMP = 1e-30
MASKED_SCORE_MODE = "ratio"  # "ratio" or "posterior"
NINF = -1e30  # practical -inf (fp16-safe)


# =========================================================
# Utility: dataset tag
# =========================================================
def make_ds_tag(meta: dict, V_eff: int) -> str:
    """
    Build a clean dataset tag for plots/tables.

    Distinguish:
      - tokV: tokenizer vocab size stored in GT config (often 4096 for BPE)
      - Veff: oracle effective state space size used in evaluation (gt.V)

    Examples:
      Stack-Python (BPE, tokV=4096, Veff=4096)
      OWT (BPE, tokV=4096, Veff=2049)
      text8-char (char, Veff=27)
    """
    if not isinstance(meta, dict):
        return f"Veff={int(V_eff)}"

    ds = str(meta.get("dataset", "")).strip().lower()
    tok = str(meta.get("tokenizer", "")).strip().lower()
    tokV = meta.get("V", None)

    tokV_str = ""
    if tokV is not None:
        try:
            tokV_str = str(int(tokV))
        except Exception:
            tokV_str = str(tokV)

    # dataset name
    if ds in ("text8", "text8_char", "text8-char", "text8char"):
        name = "text8-char"
        tok = "char" if not tok else tok
    elif ds == "owt":
        name = "OWT"
        tok = "bpe" if not tok else tok
    elif ds in ["stack_py", "stack", "the_stack", "the-stack", "stack-python", "stack_python"]:
        name = "Stack-Python"
        tok = "bpe" if not tok else tok
    elif ds:
        name = ds
    else:
        name = "dataset"

    if tok == "char":
        return f"{name} (char, Veff={int(V_eff)})"

    if tokV_str:
        return f"{name} (BPE, tokV={tokV_str}, Veff={int(V_eff)})"
    return f"{name} (BPE, Veff={int(V_eff)})"


def lse(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim)


def _ensure_distribution(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(0.0)
    return p / p.sum().clamp_min(NORM_CLAMP)


# =========================================================
# Decoding helpers
# =========================================================
def _load_hf_tokenizer(tokenizer_json: str):
    if not tokenizer_json:
        return None
    if HFTokenizer is None:
        raise RuntimeError("`tokenizers` is not installed. Please `pip install tokenizers` to decode.")
    if not os.path.exists(tokenizer_json):
        raise FileNotFoundError(f"tokenizer_json not found: {tokenizer_json}")
    return HFTokenizer.from_file(tokenizer_json)


def _decode_batch(tok, x: torch.Tensor, max_seqs: int) -> List[str]:
    """
    tok: HF tokenizers.Tokenizer
    x: [N,T] LongTensor on any device
    returns list[str] length <= max_seqs
    """
    x_cpu = x.detach().cpu().tolist()
    out: List[str] = []
    for seq in x_cpu[:max_seqs]:
        out.append(tok.decode([int(t) for t in seq], skip_special_tokens=False))
    return out


# =========================================================
# Oracle HMM score model: log-domain FB + rank-1 teleport (O(T*(V*K+V)))
# =========================================================
class OracleSEDDHMM_LogRank1Teleport(torch.nn.Module):
    """
    Implements your LaTeX: log-domain forward/backward with exact rank-1 teleport.
    Hard evidence:
      - MASK => phi_t(i)=1 for all i  (logphi=0)
      - observed token z => phi_t(z)=1, phi_t(i!=z)=0  (logphi=0/-inf)
    """

    def __init__(
        self,
        prior: SparseTeleportPrior,
        pi0: torch.Tensor,
        mask_id: int,
        store_dtype: torch.dtype = torch.float16,
        compute_dtype: torch.dtype = torch.float32,
        truncate_eps: float = 0.0,
        topm_trunc: int = 0,
        posterior_quant: str = "none",
        masked_score_mode: str = MASKED_SCORE_MODE,
    ):
        super().__init__()
        self.prior = prior
        self.mask_id = int(mask_id)
        self.store_dtype = store_dtype
        self.compute_dtype = compute_dtype
        self.masked_score_mode = masked_score_mode

        V = prior.V
        pi0 = _ensure_distribution(pi0).to(dtype=compute_dtype)
        self.register_buffer("pi0", pi0)

        nu = _ensure_distribution(prior.nu).to(dtype=compute_dtype)
        self.register_buffer("nu", nu)
        self.register_buffer("log_nu", torch.log(nu.clamp_min(NORM_CLAMP)))

        eps = float(prior.eps)
        self.log_1m_eps = float(torch.log(torch.tensor(1.0 - eps)).item())
        self.log_eps = float(torch.log(torch.tensor(eps)).item())

        # cache sparse transition on buffers
        self.register_buffer("nbr_idx", prior.nbr_idx.to(torch.long))  # [V,K]
        self.register_buffer("nbr_prob", prior.nbr_prob.to(torch.float32))  # [V,K]
        self.register_buffer("logP_topk", torch.log(self.nbr_prob.clamp_min(NORM_CLAMP)))  # [V,K]

        # knobs
        self.truncate_eps = float(truncate_eps)
        self.topm_trunc = int(topm_trunc)
        self.posterior_quant = str(posterior_quant).lower().strip()
        if self.posterior_quant not in ["none", "fp16", "bf16"]:
            raise ValueError(f"--posterior_quant must be one of none/fp16/bf16, got {posterior_quant}")

    def _quantize(self, p: torch.Tensor) -> torch.Tensor:
        if self.posterior_quant == "none":
            return p
        if self.posterior_quant == "fp16":
            return p.to(torch.float16).to(p.dtype)
        if self.posterior_quant == "bf16":
            return p.to(torch.bfloat16).to(p.dtype)
        return p

    def _row_renorm_with_fallback(self, p: torch.Tensor) -> torch.Tensor:
        B, V = p.shape
        row_sum = p.sum(dim=-1, keepdim=True)
        bad = row_sum < 1e-12
        if bad.any():
            p = torch.where(bad, torch.full_like(p, 1.0 / V), p)
            row_sum = p.sum(dim=-1, keepdim=True)
        return p / row_sum.clamp_min(NORM_CLAMP)

    def _distort_probs(self, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(0.0)
        p = self._row_renorm_with_fallback(p)

        p = self._quantize(p)
        p = p.clamp_min(0.0)
        p = self._row_renorm_with_fallback(p)

        if self.truncate_eps > 0.0:
            p = torch.where(p >= self.truncate_eps, p, torch.zeros_like(p))
            p = self._row_renorm_with_fallback(p)

        if self.topm_trunc and 0 < self.topm_trunc < p.shape[-1]:
            vals, idx = torch.topk(p, k=self.topm_trunc, dim=-1)
            p2 = torch.zeros_like(p)
            p2.scatter_(dim=-1, index=idx, src=vals)
            p = self._row_renorm_with_fallback(p2)

        return p

    @torch.no_grad()
    def _hard_evidence_logphi(self, obs_t: torch.Tensor, V: int, device) -> torch.Tensor:
        B = obs_t.shape[0]
        logphi = torch.zeros((B, V), device=device, dtype=self.compute_dtype)
        not_mask = (obs_t != self.mask_id)
        if not_mask.any():
            logphi[not_mask] = NINF
            logphi[not_mask, obs_t[not_mask]] = 0.0
        return logphi

    @torch.no_grad()
    def _forward_sparse_s(self, alpha_prob: torch.Tensor) -> torch.Tensor:
        B, V = alpha_prob.shape
        nbr_idx = self.nbr_idx  # [V,K] long, on device
        nbr_prob = self.nbr_prob.to(dtype=self.compute_dtype)  # [V,K]

        contrib = alpha_prob[:, :, None] * nbr_prob[None, :, :]  # [B,V,K]
        s = torch.zeros((B, V), device=alpha_prob.device, dtype=self.compute_dtype)
        s.scatter_add_(
            dim=1,
            index=nbr_idx[None, :, :].expand(B, -1, -1).reshape(B, -1),
            src=contrib.reshape(B, -1),
        )
        return s

    @torch.no_grad()
    def _backward_sparse_log_a(self, log_w: torch.Tensor) -> torch.Tensor:
        B, V = log_w.shape
        nbr_idx = self.nbr_idx  # [V,K]
        logP = self.logP_topk.to(dtype=self.compute_dtype)  # [V,K]
        logw_nbr = log_w[:, nbr_idx]  # [B,V,K]
        terms = logP[None, :, :] + logw_nbr
        return torch.logsumexp(terms, dim=2)  # [B,V]

    @torch.no_grad()
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        device = x.device
        assert self.nbr_idx.device == device, f"nbr_idx on {self.nbr_idx.device}, x on {device}"
        B, T = x.shape
        V, K = self.nbr_idx.shape
        assert self.mask_id == V, f"mask_id should be V={V} for absorbing graph"

        # sigma -> ratio a/b
        sigma = sigma.view(B, 1).to(self.compute_dtype)
        a = torch.exp(-sigma)
        b = (1.0 - torch.exp(-sigma)).clamp_min(1e-12)
        ratio = (a / b)  # [B,1]

        # ---------- forward: store log_alpha_t ----------
        log_alpha_all = torch.empty((B, T, V), device=device, dtype=self.store_dtype)

        logphi0 = self._hard_evidence_logphi(x[:, 0], V, device)  # [B,V]
        logpi = torch.log(self.pi0.clamp_min(NORM_CLAMP))[None, :].expand(B, -1)
        loga0 = logpi + logphi0
        loga0 = loga0 - lse(loga0, dim=1).unsqueeze(1)
        log_alpha_all[:, 0] = loga0.to(self.store_dtype)

        for t in range(1, T):
            alpha_prev = torch.exp(log_alpha_all[:, t - 1].to(self.compute_dtype))  # [B,V]
            s = self._forward_sparse_s(alpha_prev)  # [B,V]
            log_s = torch.log(s.clamp_min(NORM_CLAMP))  # [B,V]

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

            logphi = self._hard_evidence_logphi(x[:, t], V, device)  # [B,V]
            loga = logphi + log_tilde
            loga = loga - lse(loga, dim=1).unsqueeze(1)
            log_alpha_all[:, t] = loga.to(self.store_dtype)

        # ---------- backward + output score ----------
        score = torch.zeros((B, T, V + 1), device=device, dtype=self.compute_dtype)
        is_mask = (x == self.mask_id)
        not_mask = ~is_mask

        if not_mask.any():
            onehot = F.one_hot(x.clamp_max(V - 1), num_classes=V).to(self.compute_dtype)
            score[..., :V] = torch.where(not_mask.unsqueeze(-1), onehot, score[..., :V])
        score[..., V] = 1.0

        log_beta = torch.zeros((B, V), device=device, dtype=self.compute_dtype)  # log(1)=0

        def write_masked(t: int, log_beta_t: torch.Tensor):
            if not is_mask[:, t].any():
                return
            log_alpha_t = log_alpha_all[:, t].to(self.compute_dtype)
            log_post = log_alpha_t + log_beta_t
            log_post = log_post - lse(log_post, dim=1).unsqueeze(1)
            gamma = torch.exp(log_post)

            gamma = self._distort_probs(gamma)

            if self.masked_score_mode == "ratio":
                score_tokens = gamma * ratio
            elif self.masked_score_mode == "posterior":
                score_tokens = gamma
            else:
                raise ValueError(f"Unknown masked_score_mode={self.masked_score_mode}")

            score[:, t, :V] = torch.where(is_mask[:, t].unsqueeze(-1), score_tokens, score[:, t, :V])

        write_masked(T - 1, log_beta)

        for t in range(T - 2, -1, -1):
            logphi_next = self._hard_evidence_logphi(x[:, t + 1], V, device)  # [B,V]
            log_w = logphi_next + log_beta  # [B,V]

            log_a = self._backward_sparse_log_a(log_w)  # [B,V]
            log_c = lse(self.log_nu[None, :] + log_w, dim=1)  # [B]

            log_beta_new = torch.logaddexp(
                log_a + self.log_1m_eps,
                (log_c[:, None] + self.log_eps).expand(B, V),
            )

            log_beta_new = log_beta_new - lse(log_beta_new, dim=1).unsqueeze(1)

            log_beta = log_beta_new
            write_masked(t, log_beta)

        return torch.log(score.clamp_min(1e-30))


# =========================================================
# Temperature wrapper: sampler-side only
# =========================================================
class TempWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, beta: float):
        super().__init__()
        self.base = base
        self.beta = float(beta)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        logits = self.base(x, sigma)
        if self.beta != 1.0:
            logits = logits * self.beta
        return logits


# =========================================================
# SEDD runner
# =========================================================
@torch.no_grad()
def run_sedd_sampler(
    *,
    model: torch.nn.Module,
    V: int,
    x_init: torch.Tensor,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    N, T = x_init.shape
    graph = Absorbing(V)
    noise = LogLinearNoise(eps=1e-3).to(device)
    sampler = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(N, T),
        predictor="analytic",
        steps=int(steps),
        denoise=True,
        eps=1e-3,
        device=device,
    )
    return sampler(model)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True, help="path to gt_*.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--N_eval",
        type=int,
        default=128,
        help="Number of sequences used for evaluation (override gt.N)."
    )

    parser.add_argument("--out_dir", type=str, default="sampler_output")
    parser.add_argument("--run_name", type=str, default="")

    # --- NEW: accurate/inaccurate switch ---
    parser.add_argument(
        "--accuracy",
        type=str,
        default="accurate",
        choices=["accurate", "inaccurate"],
        help="Toggle oracle accuracy. accurate=clean posterior, inaccurate=apply distortion knobs.",
    )

    # --- inaccurate knobs (used only when --accuracy inaccurate) ---
    parser.add_argument(
        "--temp_beta", type=float, default=1.0,
        help="sampler temperature via logits scaling: logits *= beta. >1 sharper, <1 flatter"
    )
    parser.add_argument(
        "--truncate_eps", type=float, default=0.0,
        help="hard truncate posterior probs < eps to 0, then renormalize"
    )
    parser.add_argument(
        "--topm_trunc", type=int, default=0,
        help="keep only top-m posterior probs (0 disables)"
    )
    parser.add_argument(
        "--posterior_quant", type=str, default="none",
        choices=["none", "fp16", "bf16"],
        help="simulate low-precision posterior by casting gamma to fp16/bf16 then back"
    )

    # --- NEW: save sequences + optional decode ---
    parser.add_argument(
        "--save_seqs",
        action="store_true",
        help="Save generated sequences (ids + decoded text if tokenizer_json provided) to disk.",
    )
    parser.add_argument(
        "--tokenizer_json",
        type=str,
        default="",
        help="Path to tokenizer.json for decoding (e.g., tokenizers/owt_bytebpe_v4096/tokenizer.json).",
    )
    parser.add_argument(
        "--save_max_seqs",
        type=int,
        default=64,
        help="Max number of sequences to decode+save as text (ids are always saved for all N).",
    )

    # --- diagnostics ---
    parser.add_argument("--sanity_print", action="store_true", help="print top unigrams/bigrams per step (debug)")
    parser.add_argument("--sanity_k", type=int, default=15, help="top-k for sanity print")

    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    # ------------------------------------------------------------
    # Apply accuracy switch (FORCE accurate knobs if requested)
    # ------------------------------------------------------------
    if args.accuracy == "accurate":
        args.temp_beta = 1.0
        args.truncate_eps = 0.0
        args.topm_trunc = 0
        args.posterior_quant = "none"

    if args.truncate_eps > 0.0 and args.topm_trunc > 0:
        print("[WARN] Both truncate_eps and topm_trunc are enabled; truncation may be very aggressive.")

    # --------------------
    # Load sparse GT
    # --------------------
    gt = load_gt(args.gt, device=str(device))

    V = int(gt.V)
    N_gt = int(gt.N)      # for logging only
    N = int(args.N_eval)  # USE THIS for sampling/eval
    T = int(gt.T)

    nbr_idx = gt.nbr_idx.to(device=device, dtype=torch.long)      # [V,K]
    nbr_prob = gt.nbr_prob.to(device=device, dtype=torch.float32) # [V,K]
    nu = gt.nu.to(device=device, dtype=torch.float32)             # [V]
    eps = float(gt.eps)
    pi0 = gt.pi.to(device=device, dtype=torch.float32)            # [V]

    vocab = gt.vocab if hasattr(gt, "vocab") and isinstance(gt.vocab, list) else None
    meta: Dict[str, Any] = gt.config if hasattr(gt, "config") and isinstance(gt.config, dict) else {}

    ds_tag = make_ds_tag(meta, V_eff=V)
    K = int(nbr_idx.shape[1])

    print(f"[DS] {ds_tag}")
    print(f"[GT] path={args.gt}")
    print(f"[GT] V={V}, T={T}, gt.N={N_gt}, eval.N={N}, K={K}, eps={eps:g}")

    print(f"[CFG] accuracy={args.accuracy}")
    print(
        f"[CFG] knobs: temp_beta={args.temp_beta}, truncate_eps={args.truncate_eps}, "
        f"topm_trunc={args.topm_trunc}, posterior_quant={args.posterior_quant}"
    )
    if args.save_seqs:
        print(f"[CFG] save_seqs=True | tokenizer_json='{args.tokenizer_json}' | save_max_seqs={args.save_max_seqs}")
        if args.tokenizer_json and HFTokenizer is None:
            print("[WARN] tokenizers not installed; will save ids only (no decoded text).")

    prior = SparseTeleportPrior(nbr_idx=nbr_idx, nbr_prob=nbr_prob, nu=nu, eps=eps)

    # =========================================================
    # Output dirs (per-run folder)
    # =========================================================
    gt_base = os.path.splitext(os.path.basename(args.gt))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    knobs_tag = _sanitize(
        f"sedd_{args.accuracy}"
        f"_beta{_fmt_float_tag(args.temp_beta)}"
        f"_tr{_fmt_float_tag(args.truncate_eps)}"
        f"_topm{int(args.topm_trunc)}"
        f"_q{args.posterior_quant}"
        f"_K{K}_eps{_fmt_float_tag(eps)}"
    )

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{gt_base}_{knobs_tag}_seed{args.seed}_{timestamp}"

    OUTPUT_ROOT = os.path.join(args.out_dir, "sedd")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[OUT] run_dir={os.path.abspath(run_dir)}")

    metrics_json_path = os.path.join(run_dir, "metrics.json")
    metrics_jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")

    # --- NEW: sequences output dir
    seq_dir = os.path.join(run_dir, "sequences")
    if args.save_seqs:
        os.makedirs(seq_dir, exist_ok=True)
        print(f"[SEQ] seq_dir={os.path.abspath(seq_dir)}")

    PLOTS_ROOT = os.path.join("sampler_plots", "sedd")
    os.makedirs(PLOTS_ROOT, exist_ok=True)
    PLOT_DIR = os.path.join(PLOTS_ROOT, run_name)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"[PLOT] plot_dir={os.path.abspath(PLOT_DIR)}")
    # =========================================================

    # --- load tokenizer (optional) once
    tok = None
    if args.save_seqs and args.tokenizer_json:
        try:
            tok = _load_hf_tokenizer(args.tokenizer_json)
            print(f"[SEQ] loaded tokenizer from {args.tokenizer_json}")
        except Exception as e:
            tok = None
            print(f"[WARN] failed to load tokenizer_json='{args.tokenizer_json}': {e}")
            print("       Will save ids only (no decoded text).")

    # --------------------
    # AR baseline
    # --------------------
    x_ar = sample_ar_sparse_teleport(
        pi=pi0,
        prior=prior,
        N=N,
        T=T,
        seed=args.seed + 777,
        device=device,
    )

    # --- NEW: save AR sequences
    if args.save_seqs:
        torch.save({"x": x_ar.detach().cpu()}, os.path.join(seq_dir, "ar_ids.pt"))
        if tok is not None:
            try:
                texts = _decode_batch(tok, x_ar, max_seqs=int(args.save_max_seqs))
                with open(os.path.join(seq_dir, "ar_text.txt"), "w") as f:
                    for t in texts:
                        f.write(t + "\n")
            except Exception as e:
                print(f"[WARN] AR decode failed: {e}")

    ar_nll = nll_transition_sparse_teleport(x_ar, prior)
    ar_rt = full_kl_rate_sparse_teleport(x_ar, prior)
    ar_uni = unigram_l1(x_ar, pi=pi0, V=V)
    ar_u2 = unique_ngram_ratio(x_ar, n=2)
    ar_u3 = unique_ngram_ratio(x_ar, n=3)
    ar_dup = dup_rate(x_ar)

    ar_rec = {
        "type": "baseline_ar",
        "steps": 0,
        "seed": int(args.seed + 777),
        "accuracy": "baseline_ar",
        "dataset_tag": ds_tag,
        "tokV": int(meta.get("V", V)) if ("V" in meta) else None,
        "V_eff": int(V),

        "nll_token": float(ar_nll),
        "full_kl_rate": float(ar_rt["full_kl_rate"]),
        "full_tv_rate": float(ar_rt["full_tv_rate"]),
        "full_entropy_rate": float(ar_rt["full_entropy_rate"]),
        "unigram_L1": float(ar_uni),
        "unique_2gram_ratio": float(ar_u2),
        "unique_3gram_ratio": float(ar_u3),
        "dup_rate": float(ar_dup),
        "other_mass_rate": float(ar_rt["other_mass_rate"]),
        "support_frac": float(ar_rt["support_frac"]),
    }

    # --------------------
    # Write header
    # --------------------
    header = {
        "type": "header",
        "gt_path": args.gt,
        "device": str(device),
        "seed": int(args.seed),
        "V": int(V),
        "T": int(T),
        "K": int(K),
        "tokV": int(meta.get("V", V)) if ("V" in meta) else None,
        "V_eff": int(V),
        "dataset_tag": ds_tag,
        "gt_N": int(N_gt),
        "N_eval": int(N),

        "eps": float(eps),
        "masked_score_mode": MASKED_SCORE_MODE,

        "accuracy_mode": str(args.accuracy),

        "inaccurate_knobs_effective": {
            "temp_beta": float(args.temp_beta),
            "truncate_eps": float(args.truncate_eps),
            "topm_trunc": int(args.topm_trunc),
            "posterior_quant": str(args.posterior_quant),
        },

        # NEW: seq saving cfg
        "save_seqs": bool(args.save_seqs),
        "tokenizer_json": str(args.tokenizer_json),
        "save_max_seqs": int(args.save_max_seqs),

        "gt_meta": meta,
        "ar_baseline": ar_rec,
        "notes": "Sparse GT: stores nbr_idx/nbr_prob + rank-1 teleport (eps, nu). No dense VxV needed.",
    }

    with open(metrics_jsonl_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        f.write(json.dumps(ar_rec) + "\n")

    print("\n[AR baseline]")
    print(
        f"  AR | NLL/token={ar_nll:.6f} | fKL={ar_rt['full_kl_rate']:.3e} "
        f"| fTV={ar_rt['full_tv_rate']:.3e} | fH={ar_rt['full_entropy_rate']:.3f} "
        f"| uniL1={ar_uni:.3e} | u2={ar_u2:.4f} u3={ar_u3:.4f} | dup={ar_dup:.4f} "
        f"| other={ar_rt['other_mass_rate']:.4f} | supp={ar_rt['support_frac']:.4f}"
    )

    # --------------------
    # Build Oracle HMM model
    # --------------------
    mask_id = V
    x_init = torch.full((N, T), mask_id, dtype=torch.long, device=device)

    base_model = OracleSEDDHMM_LogRank1Teleport(
        prior=prior,
        pi0=pi0,
        mask_id=mask_id,
        store_dtype=torch.float16,
        compute_dtype=torch.float32,
        truncate_eps=args.truncate_eps,
        topm_trunc=args.topm_trunc,
        posterior_quant=args.posterior_quant,
        masked_score_mode=MASKED_SCORE_MODE,
    ).to(device).eval()

    model = TempWrapper(base_model, beta=args.temp_beta).to(device).eval()

    # --------------------
    # steps sweep
    # --------------------
    steps_list = parse_steps(args.steps)
    if not steps_list:
        raise ValueError("Empty --steps")
    print(f"\n[SEDD] steps sweep: {steps_list}")

    rows = []

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    for s in steps_list:
        x_sedd = run_sedd_sampler(
            model=model,
            V=V,
            x_init=x_init,
            steps=int(s),
            device=device,
        )

        assert (x_sedd != mask_id).all(), "SEDD output still contains mask"
        assert (x_sedd >= 0).all() and (x_sedd < V).all()

        # --- NEW: save seqs per step
        if args.save_seqs:
            step_tag = f"s{int(s):04d}"
            torch.save({"x": x_sedd.detach().cpu()}, os.path.join(seq_dir, f"sedd_{args.accuracy}_{step_tag}_ids.pt"))
            if tok is not None:
                try:
                    texts = _decode_batch(tok, x_sedd, max_seqs=int(args.save_max_seqs))
                    with open(os.path.join(seq_dir, f"sedd_{args.accuracy}_{step_tag}_text.txt"), "w") as f:
                        for t in texts:
                            f.write(t + "\n")
                except Exception as e:
                    print(f"[WARN] decode failed at step={s}: {e}")

        nll_tok = nll_transition_sparse_teleport(x_sedd, prior)
        rt = full_kl_rate_sparse_teleport(x_sedd, prior)
        uni_l1 = unigram_l1(x_sedd, pi=pi0, V=V)
        u2 = unique_ngram_ratio(x_sedd, n=2)
        u3 = unique_ngram_ratio(x_sedd, n=3)
        dr = dup_rate(x_sedd)

        rec = {
            "type": "step",
            "steps": int(s),
            "seed": int(args.seed),
            "accuracy": str(args.accuracy),
            "dataset_tag": ds_tag,
            "tokV": int(meta.get("V", V)) if ("V" in meta) else None,
            "V_eff": int(V),

            "nll_token": float(nll_tok),
            "full_kl_rate": float(rt["full_kl_rate"]),
            "full_tv_rate": float(rt["full_tv_rate"]),
            "full_entropy_rate": float(rt["full_entropy_rate"]),
            "unigram_L1": float(uni_l1),
            "unique_2gram_ratio": float(u2),
            "unique_3gram_ratio": float(u3),
            "dup_rate": float(dr),
            "other_mass_rate": float(rt["other_mass_rate"]),
            "support_frac": float(rt["support_frac"]),
        }
        rows.append(rec)
        with open(metrics_jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        print(
            f"  step={int(s):4d} | NLL/token={nll_tok:.6f} | fKL={rt['full_kl_rate']:.3e} "
            f"| fTV={rt['full_tv_rate']:.3e} | fH={rt['full_entropy_rate']:.3f} "
            f"| uniL1={uni_l1:.3e} | u2={u2:.4f} u3={u3:.4f} | dup={dr:.4f} "
            f"| other={rt['other_mass_rate']:.4f} | supp={rt['support_frac']:.4f}"
        )

        if args.sanity_print:
            top_unigrams_bigrams_print(x_sedd, V=V, k=args.sanity_k, vocab=vocab)

    # --------------------
    # Save summary + CSV
    # --------------------
    summary = {**header, "results": rows}
    with open(metrics_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n[OK] Saved metrics:\n  - {os.path.abspath(metrics_json_path)}\n  - {os.path.abspath(metrics_jsonl_path)}\n  - {os.path.abspath(metrics_csv_path)}"
    )

    if args.save_seqs:
        print(f"[OK] Saved sequences under: {os.path.abspath(seq_dir)}")

    # --------------------
    # Plots
    # --------------------
    xs = [r["steps"] for r in rows]

    def _plot(ykey: str, title: str, ylog: bool = False, ar_value: Optional[float] = None):
        ys = [r[ykey] for r in rows]
        path = os.path.join(PLOT_DIR, f"{ykey}_vs_steps_{knobs_tag}.png")

        plt.figure(figsize=(9.2, 4.9))
        plt.plot(xs, ys, marker="o", linewidth=2.3, markersize=6, label=f"SEDD ({args.accuracy})")
        if ar_value is not None:
            plt.axhline(ar_value, linestyle="--", linewidth=2.0, label="AR baseline")

        plt.xscale("log")
        if ylog:
            plt.yscale("log")

        plt.xlabel("steps")
        plt.ylabel(ykey)
        plt.title(title)
        plt.grid(True, which="both", ls="--", alpha=0.45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        print(f"[OK] Saved plot: {os.path.abspath(path)}")

    title_prefix = f"{ds_tag} | {args.accuracy.upper()} | T={T} N={N} K={K}"

    _plot("nll_token",         f"{title_prefix} | NLL/token under P'", ylog=False, ar_value=ar_rec["nll_token"])
    _plot("full_kl_rate",      f"{title_prefix} | FULL KL-rate",       ylog=True,  ar_value=ar_rec["full_kl_rate"])
    _plot("full_tv_rate",      f"{title_prefix} | FULL TV-rate",       ylog=False, ar_value=ar_rec["full_tv_rate"])
    _plot("full_entropy_rate", f"{title_prefix} | FULL entropy-rate",  ylog=False, ar_value=ar_rec["full_entropy_rate"])
    _plot("support_frac",      f"{title_prefix} | support fraction",   ylog=True,  ar_value=ar_rec["support_frac"])

    _plot("unigram_L1",        f"{title_prefix} | unigram L1 vs pi",   ylog=True,  ar_value=ar_rec["unigram_L1"])
    _plot("unique_2gram_ratio", f"{title_prefix} | unique 2-gram ratio", ylog=False, ar_value=ar_rec["unique_2gram_ratio"])
    _plot("unique_3gram_ratio", f"{title_prefix} | unique 3-gram ratio", ylog=False, ar_value=ar_rec["unique_3gram_ratio"])
    _plot("dup_rate",          f"{title_prefix} | duplicate sequence rate", ylog=False, ar_value=ar_rec["dup_rate"])
    _plot("other_mass_rate",   f"{title_prefix} | OTHER-mass rate",    ylog=False, ar_value=ar_rec["other_mass_rate"])


if __name__ == "__main__":
    main()
