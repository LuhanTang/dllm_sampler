#!/usr/bin/env python3
# tokenizers/inspect_gt_samples.py
# ------------------------------------------------------------
# Inspect an OWT GT artifact:
#  - print config + key stats
#  - sanity-check K under "p90" logic (approx, using stored dense P' = artifact["P"])
#  - decode and print a few samples (only when collapse_vocab=False)
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import torch

NORM_CLAMP = 1e-30


def _lazy_import_tokenizers():
    try:
        from tokenizers import Tokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'tokenizers' is required for decoding.\n"
            "Install with: pip install tokenizers\n"
            f"Original error: {e}"
        )
    return Tokenizer


@torch.no_grad()
def compute_row_sorted_probs(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # P: [V,V]
    sorted_probs, sorted_idx = torch.sort(P, dim=1, descending=True)
    return sorted_probs, sorted_idx


@torch.no_grad()
def pick_k_star_from_sorted(sorted_probs: torch.Tensor, target_mass: float) -> torch.Tensor:
    # sorted_probs: [V,V] each row sorted desc, sums to 1
    cum_mass = torch.cumsum(sorted_probs, dim=1)
    target = float(target_mass)
    ge = cum_mass >= target
    # argmax over bool-as-int gives first True position; if all False -> 0, so handle
    first = torch.argmax(ge.to(torch.int64), dim=1)  # 0-based
    last_mass = cum_mass[:, -1]
    bad = last_mass < target - 1e-6
    if bad.any():
        first[bad] = sorted_probs.shape[1] - 1
    return first + 1  # 1-based


def summarize_kstar(k_star: torch.Tensor) -> Dict[str, Any]:
    ks = k_star.to(torch.float32)
    out = {
        "min": int(k_star.min().item()),
        "median": int(torch.median(k_star).item()),
        "p90": int(torch.quantile(ks, 0.90).item()),
        "p95": int(torch.quantile(ks, 0.95).item()),
        "max": int(k_star.max().item()),
        "mean": float(ks.mean().item()),
    }
    return out


def decode_samples(
    *,
    samples: torch.Tensor,           # [N,T] long
    tokenizer_json: str,
    n_show: int,
) -> List[str]:
    Tokenizer = _lazy_import_tokenizers()
    tok = Tokenizer.from_file(tokenizer_json)

    outs: List[str] = []
    n = min(int(n_show), int(samples.shape[0]))
    for i in range(n):
        ids = samples[i].tolist()
        # tokenizers.Tokenizer.decode expects List[int]
        txt = tok.decode(ids, skip_special_tokens=False)
        outs.append(txt)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True, help="path to GT .pt artifact")
    ap.add_argument("--n_show", type=int, default=5, help="how many sequences to print")
    ap.add_argument("--out_txt", type=str, default="", help="optional path to save decoded samples")
    ap.add_argument("--target_mass", type=float, default=0.99, help="target mass for K* check (on P' approx)")
    ap.add_argument("--k_limit", type=int, default=300, help="print whether p90 K* <= k_limit")
    args = ap.parse_args()

    art: Dict[str, Any] = torch.load(args.gt, map_location="cpu")
    cfg = art.get("config", {})
    collapse = art.get("collapse", {})
    collapse_enabled = bool(collapse.get("enabled", cfg.get("collapse_vocab", False)))

    print("============================================================")
    print("[GT] ", os.path.abspath(args.gt))
    print("------------------------------------------------------------")
    # Print key config fields (robust to missing keys)
    keys = [
        "dataset", "dataset_name", "split",
        "V_full", "V_eff", "T", "N",
        "topk", "auto_topk", "target_mass", "k_strategy",
        "nu", "teleport_eps", "teleport_eps_source",
        "max_tokens_for_counts", "tokens_counted", "docs_counted",
        "normalize_ws", "strip",
        "tokenizer_json", "tokenizer_dir",
        "collapse_vocab", "V_keep", "other_token",
    ]
    for k in keys:
        if k in cfg:
            print(f"{k:>20s}: {cfg[k]}")
    print(f"{'collapse_enabled':>20s}: {collapse_enabled}")
    if collapse_enabled:
        print(f"{'OTHER_ID':>20s}: {collapse.get('OTHER_ID', 'NA')}")
    print("============================================================")

    # ------------------------------------------------------------
    # K* check (approx) using stored dense P' (after topk+teleport)
    # ------------------------------------------------------------
    P = art.get("P", None)
    if P is None:
        print("[WARN] artifact has no dense P. Skip K* check.")
    else:
        P = P.to(torch.float32)
        V = int(P.shape[0])
        row_sum = P.sum(dim=1)
        print(f"[SANITY] P row_sum: min={row_sum.min().item():.6f} max={row_sum.max().item():.6f} V={V}")

        sorted_probs, _ = compute_row_sorted_probs(P)
        k_star = pick_k_star_from_sorted(sorted_probs, target_mass=float(args.target_mass))
        stat = summarize_kstar(k_star)

        print("------------------------------------------------------------")
        print(f"[K* approx on P'] target_mass={args.target_mass:.3f}")
        for kk, vv in stat.items():
            if isinstance(vv, float):
                print(f"  {kk:>8s}: {vv:.3f}")
            else:
                print(f"  {kk:>8s}: {vv}")
        ok = stat["p90"] <= int(args.k_limit)
        print(f"[CHECK] p90 K* <= {args.k_limit}?  =>  {ok}   (p90={stat['p90']})")
        print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # decode samples
    # ------------------------------------------------------------
    samples = art.get("samples", None)
    if samples is None:
        print("[WARN] artifact has no 'samples'. Done.")
        return

    samples = samples.to(torch.long)

    if collapse_enabled:
        print("[NOTE] collapse_vocab=True. Decoding with tokenizer is NOT reliable (effective ids include OTHER).")
        print("       If you want readable samples, rebuild GT with --collapse_vocab disabled.")
        return

    tok_json = cfg.get("tokenizer_json", "")
    if not tok_json or not os.path.exists(tok_json):
        print("[WARN] tokenizer_json missing or not found in config. Cannot decode.")
        print("       Try rebuilding GT so config stores tokenizer_json, or edit script to pass correct path.")
        return

    decoded = decode_samples(samples=samples, tokenizer_json=tok_json, n_show=int(args.n_show))

    print("[SAMPLES]")
    for i, txt in enumerate(decoded):
        print("------------------------------------------------------------")
        print(f"[{i}]")
        print(txt)

    if args.out_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_txt)) or ".", exist_ok=True)
        with open(args.out_txt, "w", encoding="utf-8") as f:
            for i, txt in enumerate(decoded):
                f.write(f"[{i}]\n{txt}\n\n")
        print(f"[OK] saved decoded samples to: {args.out_txt}")


if __name__ == "__main__":
    main()
