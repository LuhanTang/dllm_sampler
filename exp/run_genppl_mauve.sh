##!/bin/bash
# ============================================================
# Run MAUVE / GenPPL evaluation on OWT (SEDD + ReMDM)
#
# Python script:
#   sampler/eval_owt_mauve_genppl.py
#
# ------------------------------------------------------------
# Usage:
#   bash exp/run_genppl_mauve.sh [FLAGS] [GPU_ID]
#
# If last argument is a number, it is treated as GPU_ID.
# Otherwise GPU 0 is used by default.
#
# ------------------------------------------------------------
# FLAGS (passed through to python):
#
#   --only-mauve        Compute MAUVE only (skip GenPPL)
#   --only-genppl       Compute GenPPL only (skip MAUVE)
#   --skip-sedd         Skip SEDD evaluation
#   --skip-remdm        Skip ReMDM evaluation
#   --tag XXX           Manually specify output tag
#
# ------------------------------------------------------------
# Output behavior:
#
#   If --tag is NOT provided, the script auto-generates a tag:
#
#     both models run     -> tag = sedd_remdm
#     only SEDD           -> tag = sedd
#     only ReMDM          -> tag = remdm
#
#   Output CSV will look like:
#
#     mauve_genppl_summary__TAG__YYYYMMDD_HHMMSS.csv
#
# ------------------------------------------------------------
# Examples:
#
#   # Run MAUVE + GenPPL for both SEDD and ReMDM on GPU0
#   bash exp/run_genppl_mauve.sh 0
#
#   # MAUVE only (both models)
#   bash exp/run_genppl_mauve.sh --only-mauve 0
#
#   # GenPPL only on GPU1
#   bash exp/run_genppl_mauve.sh --only-genppl 1
#
#   # Only ReMDM
#   bash exp/run_genppl_mauve.sh --skip-sedd 0
#
#   # Only SEDD
#   bash exp/run_genppl_mauve.sh --skip-remdm 0
#
#   # MAUVE only + only ReMDM
#   bash exp/run_genppl_mauve.sh --only-mauve --skip-sedd 0
#
#   # Manually specify tag
#   bash exp/run_genppl_mauve.sh --tag experimentA 0
# ============================================================

set -euo pipefail

# --------- parse GPU id (last arg if numeric) ----------
GPU_ID="0"
ARGS=("$@")

if [[ "${#ARGS[@]}" -gt 0 && "${ARGS[-1]}" =~ ^[0-9]+$ ]]; then
  GPU_ID="${ARGS[-1]}"
  unset 'ARGS[-1]'
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Python args: ${ARGS[*]:-(none)}"

# -----------------------------------------
# Detect flags
# -----------------------------------------
SKIP_SEDD=0
SKIP_REMDM=0
HAS_TAG=0

for ((i=0; i<${#ARGS[@]}; i++)); do
  arg="${ARGS[$i]}"
  if [[ "$arg" == "--skip-sedd" ]]; then
    SKIP_SEDD=1
  elif [[ "$arg" == "--skip-remdm" ]]; then
    SKIP_REMDM=1
  elif [[ "$arg" == "--tag" ]]; then
    HAS_TAG=1
  elif [[ "$arg" == --tag=* ]]; then
    HAS_TAG=1
  fi
done

if [[ "$SKIP_SEDD" -eq 1 && "$SKIP_REMDM" -eq 1 ]]; then
  echo "[ERROR] Both --skip-sedd and --skip-remdm are set."
  exit 1
fi

# -----------------------------------------
# Auto tag (if user didn't provide one)
# -----------------------------------------
AUTO_TAG=""
if [[ "$HAS_TAG" -eq 0 ]]; then
  if [[ "$SKIP_SEDD" -eq 0 && "$SKIP_REMDM" -eq 0 ]]; then
    AUTO_TAG="sedd_remdm"
  elif [[ "$SKIP_SEDD" -eq 1 ]]; then
    AUTO_TAG="remdm"
  elif [[ "$SKIP_REMDM" -eq 1 ]]; then
    AUTO_TAG="sedd"
  fi
  echo "[INFO] Auto tag: ${AUTO_TAG}"
fi

# --------- run python ----------
if [[ "$HAS_TAG" -eq 0 ]]; then
  python sampler/eval_owt_mauve_genppl.py --tag "${AUTO_TAG}" "${ARGS[@]}"
else
  python sampler/eval_owt_mauve_genppl.py "${ARGS[@]}"
fi

echo "[DONE]"
