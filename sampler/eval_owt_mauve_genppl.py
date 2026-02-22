#!/usr/bin/env python3
# eval_owt_mauve_genppl.py

import os, csv, argparse
from datetime import datetime
import torch
from tokenizers import Tokenizer

# =============================
# Ground-truth / AR
# =============================
GT_PT = "sampler_gt/gt_samples_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt"
GT_KEY = "gt_samples_ids"

AR_PT = "results/sedd/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_sedd_accurate_beta1_tr0_topm0_qnone_K206_eps0p0001_seed123_20260208_110115/sequences/ar_ids.pt"

# =============================
# SEDD runs
# =============================
SEDD_ACC_DIR = "results/sedd/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_sedd_accurate_beta1_tr0_topm0_qnone_K206_eps0p0001_seed123_20260208_110115/sequences"
SEDD_ACC_PREFIX = "sedd_accurate"

INACC = [
    (3,  "results/sedd/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_sedd_inaccurate_beta3_tr0_topm0_qnone_K206_eps0p0001_seed123_20260208_110331/sequences"),
    (5,  "results/sedd/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_sedd_inaccurate_beta5_tr0_topm0_qnone_K206_eps0p0001_seed123_20260208_110501/sequences"),
    (10, "results/sedd/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_sedd_inaccurate_beta10_tr0_topm0_qnone_K206_eps0p0001_seed123_20260208_110630/sequences"),
    (20, "results/sedd/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_sedd_inaccurate_beta20_tr0_topm0_qnone_K206_eps0p0001_seed123_20260209_111510/sequences"),
    (30, "results/sedd/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_sedd_inaccurate_beta30_tr0_topm0_qnone_K206_eps0p0001_seed123_20260210_091302/sequences"),
]
SEDD_INACC_PREFIX = "sedd_inaccurate"

# =============================
# ReMDM runs
# NOTE: each run_dir contains samples_step{step}.pt
# =============================
REMDM_RUNS = [
    ("ReMDM", "conf_p1",   "results/remdm/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_remdm_conf_p1_eta0p9_ton1_toff0_aon0_nr1_Ne512_K206_eps0p0001_seed123_20260209_104146"),
    ("ReMDM", "conf_p0p9", "results/remdm/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_remdm_conf_p0p9_eta0p9_ton1_toff0_aon0_nr1_Ne512_K206_eps0p0001_seed123_20260210_093458"),
    ("ReMDM", "loop_p1",   "results/remdm/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_remdm_loop_p1_eta0p02_ton0p55_toff0p05_aon0p9_nr1_Ne512_K206_eps0p0001_seed123_20260209_105003"),
    ("ReMDM", "loop_p0p9", "results/remdm/owt/json/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123_remdm_loop_p0p9_eta0p02_ton0p55_toff0p05_aon0p9_nr1_Ne512_K206_eps0p0001_seed123_20260209_152133"),
]

# =============================
# Shared knobs
# =============================
STEPS = [8, 64, 128, 256, 512, 1024]

TOK_PATH = "tokenizers/owt_bytebpe_v4096/tokenizer.json"

N_USE = 512
MAUVE_MAX_LEN = 1024
DEVICE_ID = 0  # MAUVE device_id; set -1 for CPU

OUT_CSV = "sampler_output/metrics/owt/mauve_genppl_summary_sedd_remdm.csv"


# -----------------------------
# IO helpers (robust)
# -----------------------------
def load_ids(pt_path: str, key: str | None = None) -> torch.Tensor:
    obj = torch.load(pt_path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        if obj.ndim != 2:
            raise ValueError(f"{pt_path}: expected 2D tensor, got shape={tuple(obj.shape)}")
        return obj.cpu()

    if isinstance(obj, dict):
        if key is not None:
            if key not in obj:
                raise KeyError(f"{pt_path}: key '{key}' not found. keys={sorted(obj.keys())}")
            x = obj[key]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{pt_path}: obj['{key}'] is not a Tensor (got {type(x)})")
            if x.ndim != 2:
                raise ValueError(f"{pt_path}: obj['{key}'] expected 2D tensor, got shape={tuple(x.shape)}")
            return x.cpu()

        candidates = [
            "ids", "seqs", "samples", "x", "token_ids", "input_ids",
            "gt_samples_ids", "ar_ids", "samples_ids"
        ]
        for k in candidates:
            if k in obj and isinstance(obj[k], torch.Tensor) and obj[k].ndim == 2:
                return obj[k].cpu()

        two_d = [(k, v) for k, v in obj.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
        if len(two_d) > 0:
            two_d.sort(key=lambda kv: (kv[1].shape[1], kv[1].shape[0]), reverse=True)
            _, v = two_d[0]
            return v.cpu()

        print(f"[load_ids] Could not find a 2D ids tensor in dict: {pt_path}")
        print("[load_ids] keys:", sorted(obj.keys()))
        for k, v in obj.items():
            print(" ", k, type(v), getattr(v, "shape", None))
        raise KeyError(f"{pt_path}: cannot auto-detect ids tensor")

    raise TypeError(f"{pt_path}: unsupported object type {type(obj)}")


def decode_batch(tok: Tokenizer, ids_2d: torch.Tensor, n_use: int) -> list[str]:
    ids_2d = ids_2d[:n_use]
    ids_list = ids_2d.tolist() if isinstance(ids_2d, torch.Tensor) else ids_2d
    return [tok.decode(seq) for seq in ids_list]


def sedd_path(run_dir: str, prefix: str, step: int) -> str:
    return os.path.join(run_dir, f"{prefix}_s{step:04d}_ids.pt")


def remdm_path(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, f"samples_step{step}.pt")


# -----------------------------
# Metrics
# -----------------------------
def compute_mauve_score(real_texts: list[str], gen_texts: list[str]) -> float:
    import mauve
    out = mauve.compute_mauve(
        p_text=real_texts,
        q_text=gen_texts,
        device_id=DEVICE_ID,
        max_text_length=MAUVE_MAX_LEN,
        num_buckets=25,
    )
    return float(out.mauve)


def init_gpt2large():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)
    model.eval()
    return tok, model, device


def compute_genppl_gpt2large(texts: list[str], gpt_tok, gpt_model, device) -> float:
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for t in texts:
            enc = gpt_tok(t, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(device)
            out = gpt_model(input_ids, labels=input_ids)
            loss = out.loss
            n_tok = int(input_ids.numel())
            total_nll += float(loss) * n_tok
            total_tokens += n_tok

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(avg_nll)))
    return float(ppl)


def add_row(rows: list[dict], *, model: str, setting: str, step, mauve_val, genppl_val):
    rows.append({
        "model": model,
        "setting": setting,
        "step": step,
        "mauve": "" if mauve_val is None else mauve_val,
        "genppl_gpt2large": "" if genppl_val is None else genppl_val,
        "n_use": N_USE,
        "mauve_max_len": MAUVE_MAX_LEN,
    })


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-mauve", action="store_true", help="Compute MAUVE only (skip GenPPL).")
    ap.add_argument("--only-genppl", action="store_true", help="Compute GenPPL only (skip MAUVE).")
    ap.add_argument("--skip-sedd", action="store_true", help="Skip all SEDD evals.")
    ap.add_argument("--skip-remdm", action="store_true", help="Skip all ReMDM evals.")
    ap.add_argument("--tag", type=str, default="", help="Tag for output filename (used by bash).")
    args = ap.parse_args()

    if args.only_mauve and args.only_genppl:
        raise SystemExit("Error: choose at most one of --only-mauve / --only-genppl.")

    do_mauve = args.only_mauve or (not args.only_genppl)
    do_genppl = args.only_genppl or (not args.only_mauve)

    tok = Tokenizer.from_file(TOK_PATH)

    # real texts (only for mauve)
    real_texts = None
    if do_mauve:
        gt_ids = load_ids(GT_PT, GT_KEY)
        real_texts = decode_batch(tok, gt_ids, N_USE)

    # init gpt2-large once (only for genppl)
    gpt_tok = gpt_model = device = None
    if do_genppl:
        gpt_tok, gpt_model, device = init_gpt2large()

    rows: list[dict] = []

    def eval_one(ids_pt: str, label_model: str, label_setting: str, label_step):
        if not os.path.exists(ids_pt):
            print(f"[WARN] missing: {ids_pt} (skip)")
            return

        ids = load_ids(ids_pt, key=None)
        texts = decode_batch(tok, ids, N_USE)

        mauve_val = compute_mauve_score(real_texts, texts) if do_mauve else None
        genppl_val = compute_genppl_gpt2large(texts, gpt_tok, gpt_model, device) if do_genppl else None

        add_row(
            rows,
            model=label_model,
            setting=label_setting,
            step=label_step,
            mauve_val=mauve_val,
            genppl_val=genppl_val,
        )

        print(f"[OK] {label_model:6s} | {label_setting:16s} | step={str(label_step):>4s} | {os.path.basename(ids_pt)}")

    # AR baseline
    eval_one(AR_PT, "AR", "baseline", "")

    # SEDD
    if not args.skip_sedd:
        for step in STEPS:
            pt = sedd_path(SEDD_ACC_DIR, SEDD_ACC_PREFIX, step)
            eval_one(pt, "SEDD", "accurate_beta1", step)

        for beta, run_dir in INACC:
            for step in STEPS:
                pt = sedd_path(run_dir, SEDD_INACC_PREFIX, step)
                eval_one(pt, "SEDD", f"inaccurate_beta{beta}", step)

    # ReMDM
    if not args.skip_remdm:
        for model, setting, run_dir in REMDM_RUNS:
            for step in STEPS:
                pt = remdm_path(run_dir, step)
                eval_one(pt, model, setting, step)

    # -----------------------------
    # Write CSV: add tag + timestamp
    # -----------------------------
    base_dir = os.path.dirname(OUT_CSV)
    base_name = os.path.splitext(os.path.basename(OUT_CSV))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip().replace(" ", "_")
    suffix = f"__{tag}" if tag else ""
    out_csv = os.path.join(base_dir, f"{base_name}{suffix}__{ts}.csv")

    os.makedirs(base_dir, exist_ok=True)
    fieldnames = ["model", "setting", "step", "mauve", "genppl_gpt2large", "n_use", "mauve_max_len"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_csv)
    print(f"Computed: mauve={do_mauve}, genppl={do_genppl}")


if __name__ == "__main__":
    main()
