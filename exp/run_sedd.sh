#!/bin/bash
# ============================================================
# SEDD Experiments (text8 / OpenWebText)
# ============================================================
# Usage:
#   bash exp/run_sedd.sh <config> <gpu_id>
#
# Notes:
# - OWT GT: tokenizer V=4096, but oracle effective V may be keep-K + OTHER (depends on GT).
# - We explicitly set --seed 123 and --N_eval (default 128) for every run.
#
# IMPORTANT MOD (Feb 2026):
# - "inaccurate" runs only enable TEMPERATURE via --temp_beta.
#   All other distortion knobs (truncate/topm/quant) are disabled (not passed).
#
# NEW MOD (Samples):
# - Optional dumping of generated sequences:
#     SAVE_SEQS=1 ...  -> adds --save_seqs
# - Optional decoding:
#     OWT uses tokenizers/owt_bytebpe_v4096/tokenizer.json via --tokenizer_json
# - Limit how many sequences to decode/save to text:
#     SAVE_MAX_SEQS=50 (default)
# ============================================================

# QUICK START / EXAMPLES
# ------------------------------------------------------------
#
# 1) text8 (small, T=128), accurate oracle
#    - No evaluator
#    - Save metrics + plots only
#
#   bash exp/run_sedd.sh text8_small_accurate 0
#
#
# 2) text8 (small, T=128), inaccurate oracle (temperature only)
#
#   bash exp/run_sedd.sh text8_small_inaccurate 0
#
#
# 3) text8 (large, T=1024), accurate oracle
#    - Uses GenPPL evaluator (GPT-2)
#    - Optionally save samples
#
#   SAVE_SEQS=1 bash exp/run_sedd.sh text8_large_accurate 0
#
#
# 4) text8 (large, T=1024), inaccurate oracle
#    - Sweep over beta
#    - Demonstrates GenPPL manipulability
#
#   SAVE_SEQS=1 bash exp/run_sedd.sh text8_large_inaccurate 0
#
#
# 5) OpenWebText (OWT), accurate oracle
#    - NO decoding
#    - NO evaluator
#    - Saves metrics + ids only
#
#   SAVE_SEQS=1 bash exp/run_sedd.sh owt_large_accurate 0
#
#
# 6) OpenWebText (OWT), inaccurate oracle (temperature only)
#
#   SAVE_SEQS=1 bash exp/run_sedd.sh owt_large_inaccurate 0
#
#
# 7) Override evaluation size (all configs)
#
#   N_EVAL=64 bash exp/run_sedd.sh owt_large_accurate 0
#
#
# 8) Save a small subset of samples (for inspection only)
#     - Saves ids for all runs
#     - text8 additionally saves decoded text (first 50 seqs)
#
#   SAVE_SEQS=1 SAVE_MAX_SEQS=50 bash exp/run_sedd.sh text8_large_accurate 0
#
#
# NOTES
# ------------------------------------------------------------
# - For OWT:
#     SAVE_SEQS=1 saves ONLY token ids (.pt), no decoding.
# - For text8:
#     SAVE_SEQS=1 additionally saves decoded text (.txt).
# - All evaluator-based metrics (GenPPL, MAUVE) are computed
#   only in dedicated scripts or the *_genppl runner.
# - Sampler outputs (ids) are always saved BEFORE any decoding
#   or evaluator is applied.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-text8_small_accurate}"
GPU="${2:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

# ------------------------------------------------------------
# Single runner (no GenPPL)
# ------------------------------------------------------------
PYMOD_DEFAULT="sampler.sedd.run_sedd"

# -------------------------
# Shared settings
# -------------------------
SEED=123
N_EVAL_DEFAULT=512

STEPS_TEXT8_SMALL="8,16,32,64,128,256"
STEPS_LARGE="8,16,32,64,128,256,512,1024"

# -------------------------
# GT paths
# -------------------------
GT_TEXT8_T128="gt_text8_char_T128_N1000_topk27_lam1e-4.pt"

GT_TEXT8_T1024="sampler_gt/gt_text8_char_withSpace_T1024_N128_topk27_eps0.pt"
GT_OWT="sampler_gt/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt"

# -------------------------
# Tokenizer paths for decoding
# -------------------------
TOK_OWT_JSON="tokenizers/owt_bytebpe_v4096/tokenizer.json"

# -------------------------
# Temperature (ONLY inaccurate knob)
# -------------------------
BETA_TEXT8_SMALL=1.2
BETA_TEXT8_LARGE=10
BETA_OWT_LARGE=30

# -------------------------
# Samples dump settings (env override)
# -------------------------
SAVE_SEQS="${SAVE_SEQS:-0}"          # set SAVE_SEQS=1 to dump sequences
SAVE_MAX_SEQS="${SAVE_MAX_SEQS:-50}" # limit how many sequences to decode/write to text

# optional: allow overriding N_eval via env var, e.g. N_EVAL=64 bash exp/run_sedd.sh ...
N_EVAL="${N_EVAL:-$N_EVAL_DEFAULT}"

# -------------------------
# Helper: check file exists
# -------------------------
must_exist () {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] File not found: $p"
    exit 1
  fi
}

# -------------------------
# Helper: append sample dumping flags to CMD array
# -------------------------
maybe_add_save_seqs () {
  # usage: maybe_add_save_seqs CMD_ARRAY_NAME
  local -n _cmd=$1
  if [[ "$SAVE_SEQS" == "1" ]]; then
    _cmd+=(--save_seqs --save_max_seqs "$SAVE_MAX_SEQS")
  fi
}

# -------------------------
# Helper: append OWT tokenizer json for decoding if SAVE_SEQS enabled
# -------------------------
maybe_add_owt_tokenizer () {
  local -n _cmd=$1
  if [[ "$SAVE_SEQS" == "1" ]]; then
    must_exist "$TOK_OWT_JSON"
    _cmd+=(--tokenizer_json "$TOK_OWT_JSON")
  fi
}

case "$CONFIG" in

# ============================================================
# text8 (small)
# ============================================================
text8_small_accurate)
  echo "[SEDD][text8] T=128 | accurate | seed=$SEED | N_eval=$N_EVAL | save_seqs=$SAVE_SEQS"
  must_exist "$GT_TEXT8_T128"

  CMD=(python -m "$PYMOD_DEFAULT"
    --gt "$GT_TEXT8_T128"
    --device cuda:0
    --steps "$STEPS_TEXT8_SMALL"
    --seed "$SEED"
    --N_eval "$N_EVAL"
    --accuracy accurate
  )
  maybe_add_save_seqs CMD
  "${CMD[@]}"
  ;;

text8_small_inaccurate)
  echo "[SEDD][text8] T=128 | inaccurate(beta=$BETA_TEXT8_SMALL) | seed=$SEED | N_eval=$N_EVAL | save_seqs=$SAVE_SEQS"
  must_exist "$GT_TEXT8_T128"

  CMD=(python -m "$PYMOD_DEFAULT"
    --gt "$GT_TEXT8_T128"
    --device cuda:0
    --steps "$STEPS_TEXT8_SMALL"
    --seed "$SEED"
    --N_eval "$N_EVAL"
    --accuracy inaccurate
    --temp_beta "$BETA_TEXT8_SMALL"
  )
  maybe_add_save_seqs CMD
  "${CMD[@]}"
  ;;

# ============================================================
# text8 (large)
# ============================================================
text8_large_accurate)
  echo "[SEDD][text8] T=1024 | accurate | seed=$SEED | N_eval=$N_EVAL | save_seqs=$SAVE_SEQS"
  must_exist "$GT_TEXT8_T1024"

  CMD=(python -m "$PYMOD_DEFAULT"
    --gt "$GT_TEXT8_T1024"
    --device cuda:0
    --steps "$STEPS_LARGE"
    --seed "$SEED"
    --N_eval "$N_EVAL"
    --accuracy accurate
  )
  maybe_add_save_seqs CMD
  "${CMD[@]}"
  ;;

text8_large_inaccurate)
  must_exist "$GT_TEXT8_T1024"

  # You can sweep betas by editing this list or overriding in-place.
  BETAS_TEXT8_LARGE=(10)

  echo "[SEDD][text8] T=1024 | inaccurate | betas=${BETAS_TEXT8_LARGE[*]} | seed=$SEED | N_eval=$N_EVAL | save_seqs=$SAVE_SEQS"

  for BETA in "${BETAS_TEXT8_LARGE[@]}"; do
    echo ""
    echo "---- beta=${BETA} ----"

    CMD=(python -m "$PYMOD_DEFAULT"
      --gt "$GT_TEXT8_T1024"
      --device cuda:0
      --steps "$STEPS_LARGE"
      --seed "$SEED"
      --N_eval "$N_EVAL"
      --accuracy inaccurate
      --temp_beta "$BETA"
      --run_name "text8_large_inaccurate_beta${BETA}"
    )
    maybe_add_save_seqs CMD
    "${CMD[@]}"
  done
  ;;

# ============================================================
# OWT (large) — decode via byteBPE tokenizer.json if SAVE_SEQS=1
# ============================================================
owt_large_accurate)
  echo "[SEDD][OWT] T=1024 | accurate | seed=$SEED | N_eval=$N_EVAL | save_seqs=$SAVE_SEQS"
  must_exist "$GT_OWT"

  CMD=(python -m "$PYMOD_DEFAULT"
    --gt "$GT_OWT"
    --device cuda:0
    --steps "$STEPS_LARGE"
    --seed "$SEED"
    --N_eval "$N_EVAL"
    --accuracy accurate
  )
  maybe_add_save_seqs CMD
  maybe_add_owt_tokenizer CMD
  "${CMD[@]}"
  ;;

owt_large_inaccurate)
  echo "[SEDD][OWT] T=1024 | inaccurate(beta=$BETA_OWT_LARGE) | seed=$SEED | N_eval=$N_EVAL | save_seqs=$SAVE_SEQS"
  must_exist "$GT_OWT"

  CMD=(python -m "$PYMOD_DEFAULT"
    --gt "$GT_OWT"
    --device cuda:0
    --steps "$STEPS_LARGE"
    --seed "$SEED"
    --N_eval "$N_EVAL"
    --accuracy inaccurate
    --temp_beta "$BETA_OWT_LARGE"
  )
  maybe_add_save_seqs CMD
  maybe_add_owt_tokenizer CMD
  "${CMD[@]}"
  ;;

# ============================================================
# Batch
# ============================================================
all_text8)
  echo "Running ALL text8 configs on GPU=$GPU (seed=$SEED, N_eval=$N_EVAL, save_seqs=$SAVE_SEQS)"
  for c in text8_large_accurate text8_large_inaccurate; do
    echo ""
    echo "====== $c ======"
    bash "$0" "$c" "$GPU"
  done
  ;;

all_owt)
  echo "Running ALL OWT configs on GPU=$GPU (seed=$SEED, N_eval=$N_EVAL, save_seqs=$SAVE_SEQS)"
  for c in owt_large_accurate owt_large_inaccurate; do
    echo ""
    echo "====== $c ======"
    bash "$0" "$c" "$GPU"
  done
  ;;

all_large)
  echo "Running ALL large-scale configs (text8 + OWT) on GPU=$GPU (seed=$SEED, N_eval=$N_EVAL, save_seqs=$SAVE_SEQS)"
  for c in text8_large_accurate text8_large_inaccurate \
           owt_large_accurate owt_large_inaccurate; do
    echo ""
    echo "====== $c ======"
    bash "$0" "$c" "$GPU"
  done
  ;;

all)
  bash "$0" all_text8 "$GPU"
  bash "$0" all_owt "$GPU"
  ;;

*)
  echo "Usage: bash exp/run_sedd.sh [config] [gpu_id]"
  echo ""
  echo "Configs:"
  echo "  text8_small_accurate, text8_small_inaccurate"
  echo "  text8_large_accurate, text8_large_inaccurate"
  echo "  owt_large_accurate,   owt_large_inaccurate"
  echo ""
  echo "Batch:"
  echo "  all_text8, all_owt, all_large, all"
  echo ""
  echo "Env overrides examples:"
  echo "  N_EVAL=64 SAVE_SEQS=1 SAVE_MAX_SEQS=30 bash exp/run_sedd.sh owt_large_accurate 0"
  echo "  SAVE_SEQS=1 bash exp/run_sedd.sh text8_large_inaccurate 0"
  exit 1
  ;;
esac

echo "✅ Done"
