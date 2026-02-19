#!/bin/bash
# ============================================================
# Unified batch runner for current experiment scripts.
# ============================================================
# Usage:
#   bash exp/run_all.sh <target> <gpu_id> [N_eval]
#
# target:
#   text8 | owt | large | all
#
# Notes:
# - LLaDA/MDLM script is exp/run_mdlm_llada.sh
#   (low_confidence = LLaDA, random = MDLM)
# - ReMDM accepts optional 3rd arg N_eval.
# - SEDD / LLaDA-MDLM use their own env-based N eval settings.
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-all}"
GPU="${2:-0}"
N_EVAL="${3:-512}"

run_llada_block() {
  local mode="$1"
  echo ""
  echo "======== LLaDA/MDLM (${mode}) ========"
  bash ./run_mdlm_llada.sh "$mode" "$GPU"
}

run_sedd_block() {
  local mode="$1"
  echo ""
  echo "======== SEDD (${mode}) ========"
  bash ./run_sedd.sh "$mode" "$GPU"
}

run_remdm_block() {
  local mode="$1"
  echo ""
  echo "======== ReMDM (${mode}) ========"
  bash ./run_remdm.sh "$mode" "$GPU" "$N_EVAL"
}

echo "========================================"
echo "  Running Sampler Batch"
echo "  Target: $TARGET"
echo "  GPU: $GPU"
echo "  N_eval (ReMDM): $N_EVAL"
echo "========================================"

case "$TARGET" in
  text8)
    run_llada_block all_text8
    run_sedd_block all_text8
    run_remdm_block all_text8
    ;;
  owt)
    run_llada_block all_owt
    run_sedd_block all_owt
    run_remdm_block all_owt
    ;;
  large)
    run_llada_block all_large
    run_sedd_block all_large
    run_remdm_block all_large
    ;;
  all)
    run_llada_block all
    run_sedd_block all
    run_remdm_block all
    ;;
  *)
    echo "Usage: bash exp/run_all.sh <target> <gpu_id> [N_eval]"
    echo ""
    echo "target:"
    echo "  text8 | owt | large | all"
    exit 1
    ;;
esac

echo ""
echo "========================================"
echo "  Batch completed."
echo "========================================"
