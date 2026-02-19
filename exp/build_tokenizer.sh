#!/bin/bash
# ============================================================
# Build OWT tokenizer + OWT GT (current tokenizers/ scripts)
# ============================================================
# Usage:
#   bash exp/build_tokenizer.sh <mode>
#
# mode:
#   tokenizer   train OWT byte-level BPE tokenizer only
#   gt          build OWT GT only (requires existing tokenizer)
#   all         train tokenizer, then build GT
#   inspect     inspect/decode GT samples
#
# Examples:
#   bash exp/build_tokenizer.sh tokenizer
#   bash exp/build_tokenizer.sh gt
#   bash exp/build_tokenizer.sh all
#   bash exp/build_tokenizer.sh inspect
#
# Env overrides (optional):
#   TOKENIZER_DIR=tokenizers/owt_bytebpe_v4096
#   OWT_GT_OUT=sampler_gt/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt
#   OWT_GT_SAMPLES_OUT=sampler_gt/gt_owt_samples_T1024_N1000_seed123.pt
#   OWT_NAME=stanford-cs336/owt-sample
#   MAX_TEXT_BYTES=200000000
#   MAX_TOKENS_FOR_COUNTS=50000000
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-all}"

# -------------------------
# Paths
# -------------------------
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizers/owt_bytebpe_v4096}"
TOKENIZER_JSON="${TOKENIZER_DIR}/tokenizer.json"

OWT_GT_OUT="${OWT_GT_OUT:-sampler_gt/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt}"
OWT_GT_SAMPLES_OUT="${OWT_GT_SAMPLES_OUT:-sampler_gt/gt_owt_samples_T1024_N1000_seed123.pt}"

# -------------------------
# Shared config
# -------------------------
SEED="${SEED:-123}"
OWT_NAME="${OWT_NAME:-stanford-cs336/owt-sample}"
SPLIT="${SPLIT:-train}"

# Tokenizer train config
VOCAB_SIZE="${VOCAB_SIZE:-4096}"
MAX_TEXT_BYTES="${MAX_TEXT_BYTES:-200000000}"
MAX_DOCS_TOK="${MAX_DOCS_TOK:-0}"
MAX_CHARS_PER_DOC_TOK="${MAX_CHARS_PER_DOC_TOK:-4096}"
NORMALIZE_WS_TOK="${NORMALIZE_WS_TOK:-0}"
STRIP_TOK="${STRIP_TOK:-1}"

# GT build config
MAX_TOKENS_FOR_COUNTS="${MAX_TOKENS_FOR_COUNTS:-50000000}"
MAX_CHARS_PER_DOC_GT="${MAX_CHARS_PER_DOC_GT:-4096}"
MAX_DOCS_GT="${MAX_DOCS_GT:-0}"
LOG_EVERY_DOCS="${LOG_EVERY_DOCS:-2000}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-128}"

T="${T:-1024}"
N="${N:-1000}"
TOPK="${TOPK:-128}"
TARGET_MASS="${TARGET_MASS:-0.99}"
K_STRATEGY="${K_STRATEGY:-p90}"
NU="${NU:-unigram}"
TELEPORT_EPS="${TELEPORT_EPS:-1e-4}"
PI_ITERS="${PI_ITERS:-500}"

AUTO_TOPK="${AUTO_TOPK:-1}"
AUTO_EPS="${AUTO_EPS:-0}"
EPS_AGG="${EPS_AGG:-median}"
COLLAPSE_VOCAB="${COLLAPSE_VOCAB:-0}"
V_KEEP="${V_KEEP:-2048}"
OTHER_TOKEN="${OTHER_TOKEN:-<other>}"
PLOT_DIR="${PLOT_DIR:-sampler_plots/gt_owt}"

must_exist() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Missing file: $p"
    exit 1
  fi
}

build_tokenizer() {
  echo "[STEP] Train OWT ByteBPE tokenizer"
  echo "[OUT ] ${TOKENIZER_DIR}"

  CMD=(python tokenizers/train_owt_bytebpe_tokenizer.py
    --out_dir "${TOKENIZER_DIR}"
    --vocab_size "${VOCAB_SIZE}"
    --max_text_bytes "${MAX_TEXT_BYTES}"
    --seed "${SEED}"
    --owt_name "${OWT_NAME}"
    --split "${SPLIT}"
    --max_docs "${MAX_DOCS_TOK}"
    --max_chars_per_doc "${MAX_CHARS_PER_DOC_TOK}"
  )

  if [[ "${NORMALIZE_WS_TOK}" == "1" ]]; then
    CMD+=(--normalize_ws)
  fi
  if [[ "${STRIP_TOK}" == "1" ]]; then
    CMD+=(--strip)
  fi

  echo "[CMD ] ${CMD[*]}"
  "${CMD[@]}"
}

build_gt() {
  must_exist "${TOKENIZER_JSON}"
  mkdir -p "$(dirname "${OWT_GT_OUT}")"
  mkdir -p "$(dirname "${OWT_GT_SAMPLES_OUT}")"

  echo "[STEP] Build OWT GT from tokenizer"
  echo "[TOK ] ${TOKENIZER_JSON}"
  echo "[OUT ] ${OWT_GT_OUT}"

  CMD=(python tokenizers/build_owt_bpe_gt.py
    --out "${OWT_GT_OUT}"
    --tokenizer_dir "${TOKENIZER_DIR}"
    --encode_batch_size "${ENCODE_BATCH_SIZE}"
    --owt_name "${OWT_NAME}"
    --split "${SPLIT}"
    --max_tokens_for_counts "${MAX_TOKENS_FOR_COUNTS}"
    --max_chars_per_doc "${MAX_CHARS_PER_DOC_GT}"
    --max_docs "${MAX_DOCS_GT}"
    --log_every_docs "${LOG_EVERY_DOCS}"
    --T "${T}"
    --N "${N}"
    --seed "${SEED}"
    --gt_samples_out "${OWT_GT_SAMPLES_OUT}"
    --topk "${TOPK}"
    --target_mass "${TARGET_MASS}"
    --k_strategy "${K_STRATEGY}"
    --nu "${NU}"
    --teleport_eps "${TELEPORT_EPS}"
    --eps_agg "${EPS_AGG}"
    --pi_iters "${PI_ITERS}"
    --v_keep "${V_KEEP}"
    --other_token "${OTHER_TOKEN}"
    --plot_dir "${PLOT_DIR}"
  )

  if [[ "${AUTO_TOPK}" == "1" ]]; then
    CMD+=(--auto_topk)
  fi
  if [[ "${AUTO_EPS}" == "1" ]]; then
    CMD+=(--auto_eps)
  fi
  if [[ "${COLLAPSE_VOCAB}" == "1" ]]; then
    CMD+=(--collapse_vocab)
  fi

  # Keep text cleanup behavior explicit for GT stream
  CMD+=(--normalize_ws --strip)

  echo "[CMD ] ${CMD[*]}"
  "${CMD[@]}"
}

inspect_gt() {
  must_exist "${OWT_GT_OUT}"

  echo "[STEP] Inspect GT artifact"
  CMD=(python tokenizers/inspect_gt_samples.py
    --gt "${OWT_GT_OUT}"
    --n_show "${N_SHOW:-5}"
    --target_mass "${TARGET_MASS}"
    --k_limit "${K_LIMIT:-300}"
  )
  echo "[CMD ] ${CMD[*]}"
  "${CMD[@]}"
}

case "${MODE}" in
  tokenizer)
    build_tokenizer
    ;;
  gt)
    build_gt
    ;;
  all)
    build_tokenizer
    build_gt
    ;;
  inspect)
    inspect_gt
    ;;
  *)
    echo "Usage: bash exp/build_tokenizer.sh <mode>"
    echo ""
    echo "mode: tokenizer | gt | all | inspect"
    exit 1
    ;;
esac

echo "✅ Done"
