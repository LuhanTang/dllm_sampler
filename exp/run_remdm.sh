#!/bin/bash
# ============================================================
# ReMDM Experiments (UPDATED for run_remdm.py, chunked N)
#   - Supports --N_chunk to reduce VRAM peak while keeping total N_eval fixed.
#   - Keeps seed convention:
#       ReMDM seed = 123 (global, once)
#       AR baseline seed = 123 + 777 (inside python)
# ============================================================
#
# Usage:
#   bash exp/run_remdm.sh <config_name> <gpu_id> [N_eval]
#
# Examples:
#   # Default N_chunk=128, N_eval=512
#   bash exp/run_remdm.sh owt_large_conf 0 512
#
#   # Smaller chunk if still OOM
#   N_CHUNK=64 bash exp/run_remdm.sh owt_large_conf 0 512
#
#   # Dump ALL N samples (ids) for ALL steps on OWT
#   DUMP_SAMPLES=1 DUMP_STEPS="" DUMP_LIMIT=0 \
#     bash exp/run_remdm.sh owt_large_conf 0 512
# ============================================================

set -e
cd "$(dirname "$0")/.."

CONFIG="${1:-text8_small_loop}"
GPU="${2:-0}"
N_EVAL="${3:-512}"

export CUDA_VISIBLE_DEVICES=$GPU

# allocator anti-fragmentation (optional but recommended)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ----------------------------
# NEW: chunk size for sampling
#   default: 128 (as you requested)
# ----------------------------
N_CHUNK="${N_CHUNK:-128}"

# ----------------------------
# GT paths
# ----------------------------
GT_TEXT8_LARGE="sampler_gt/gt_text8_char_withSpace_T1024_N128_topk27_eps0.pt"
GT_OWT_LARGE="sampler_gt/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt"
GT_TEXT8_SMALL="gt_text8_char_T128_N1000_topk27_lam1e-4.pt"

# ----------------------------
# Global optional knobs
# ----------------------------
NUCLEUS_P="${NUCLEUS_P:-0.9}"

# ----------------------------
# Author-like ReMDM-loop parameters
# ----------------------------
ETA="${ETA:-0.02}"
T_ON="${T_ON:-0.55}"
T_OFF="${T_OFF:-0.05}"
ALPHA_ON="${ALPHA_ON:-0.9}"

# ----------------------------
# Optional debug toggle
# ----------------------------
DEBUG_STATS="${DEBUG_STATS:-0}"
DBG_FLAG=""
if [ "$DEBUG_STATS" -eq 1 ]; then
  DBG_FLAG="--debug_stats"
fi

# ----------------------------
# Optional dumping: token IDs only (recommended for OWT)
# ----------------------------
DUMP_SAMPLES="${DUMP_SAMPLES:-0}"     # 1 => enable dump ids
DUMP_STEPS="${DUMP_STEPS:-}"          # e.g. "32,128,512"; empty => dump all steps in --steps
DUMP_LIMIT="${DUMP_LIMIT:-0}"         # 0 => dump all N; else dump first K

DUMP_FLAGS=""
if [ "$DUMP_SAMPLES" -eq 1 ]; then
  DUMP_FLAGS="--dump_ids"
  if [ -n "$DUMP_STEPS" ]; then
    DUMP_FLAGS="${DUMP_FLAGS} --dump_steps ${DUMP_STEPS}"
  fi
  if [ "$DUMP_LIMIT" -ne 0 ]; then
    DUMP_FLAGS="${DUMP_FLAGS} --dump_limit ${DUMP_LIMIT}"
  fi
fi

# ----------------------------
# Common flags
# ----------------------------
COMMON_FLAGS="--device cuda:0 --seed 123 --N_eval ${N_EVAL} --N_chunk ${N_CHUNK} ${DBG_FLAG}"

case "$CONFIG" in

# ------------------------------------------------------------
# TEXT8
# ------------------------------------------------------------
text8_small_loop)
    echo "[ReMDM][text8] T=128 | sampler=remdm_loop | nucleus_p=${NUCLEUS_P} | N_eval=${N_EVAL} | N_chunk=${N_CHUNK}"
    python -m sampler.remdm.run_remdm \
        --gt "${GT_TEXT8_SMALL}" \
        --steps "8,16,32,64,128,256" \
        --sampler remdm_loop \
        --eta "${ETA}" --t_on "${T_ON}" --t_off "${T_OFF}" --alpha_on "${ALPHA_ON}" \
        --nucleus_p "${NUCLEUS_P}" \
        --noise_removal \
        ${COMMON_FLAGS}
    ;;

text8_small_conf)
    echo "[ReMDM][text8] T=128 | sampler=remdm_conf | nucleus_p=${NUCLEUS_P} | N_eval=${N_EVAL} | N_chunk=${N_CHUNK}"
    python -m sampler.remdm.run_remdm \
        --gt "${GT_TEXT8_SMALL}" \
        --steps "8,16,32,64,128,256" \
        --sampler remdm_conf \
        --nucleus_p "${NUCLEUS_P}" \
        --noise_removal \
        ${COMMON_FLAGS}
    ;;

text8_large_loop)
    echo "[ReMDM][text8] T=1024 | sampler=remdm_loop | nucleus_p=${NUCLEUS_P} | N_eval=${N_EVAL} | N_chunk=${N_CHUNK}"
    python -m sampler.remdm.run_remdm \
        --gt "${GT_TEXT8_LARGE}" \
        --steps "8,16,32,64,128,256,512,1024" \
        --sampler remdm_loop \
        --eta "${ETA}" --t_on "${T_ON}" --t_off "${T_OFF}" --alpha_on "${ALPHA_ON}" \
        --nucleus_p "${NUCLEUS_P}" \
        --noise_removal \
        ${COMMON_FLAGS}
    ;;

text8_large_conf)
    echo "[ReMDM][text8] T=1024 | sampler=remdm_conf | nucleus_p=${NUCLEUS_P} | N_eval=${N_EVAL} | N_chunk=${N_CHUNK}"
    python -m sampler.remdm.run_remdm \
        --gt "${GT_TEXT8_LARGE}" \
        --steps "8,16,32,64,128,256,512,1024" \
        --sampler remdm_conf \
        --nucleus_p "${NUCLEUS_P}" \
        --noise_removal \
        ${COMMON_FLAGS}
    ;;

# ------------------------------------------------------------
# OPENWEBTEXT (OWT)
# ------------------------------------------------------------
owt_large_loop)
    echo "[ReMDM][OWT] T=1024 | sampler=remdm_loop | nucleus_p=${NUCLEUS_P} | N_eval=${N_EVAL} | N_chunk=${N_CHUNK} | dump_ids=${DUMP_SAMPLES}"
    python -m sampler.remdm.run_remdm \
        --gt "${GT_OWT_LARGE}" \
        --steps "8,16,32,64,128,256,512,1024" \
        --sampler remdm_loop \
        --eta "${ETA}" --t_on "${T_ON}" --t_off "${T_OFF}" --alpha_on "${ALPHA_ON}" \
        --nucleus_p "${NUCLEUS_P}" \
        --noise_removal \
        ${COMMON_FLAGS} \
        ${DUMP_FLAGS}
    ;;

owt_large_conf)
    echo "[ReMDM][OWT] T=1024 | sampler=remdm_conf | nucleus_p=${NUCLEUS_P} | N_eval=${N_EVAL} | N_chunk=${N_CHUNK} | dump_ids=${DUMP_SAMPLES}"
    python -m sampler.remdm.run_remdm \
        --gt "${GT_OWT_LARGE}" \
        --steps "8,16,32,64,128,256,512,1024" \
        --sampler remdm_conf \
        --nucleus_p "${NUCLEUS_P}" \
        --noise_removal \
        ${COMMON_FLAGS} \
        ${DUMP_FLAGS}
    ;;

# ------------------------------------------------------------
# Batch runners
# ------------------------------------------------------------
all_text8)
    for cfg in text8_large_loop text8_large_conf; do
        bash "$0" "$cfg" "$GPU" "$N_EVAL"
    done
    ;;

all_owt)
    for cfg in owt_large_loop owt_large_conf; do
        bash "$0" "$cfg" "$GPU" "$N_EVAL"
    done
    ;;

all_large)
    for cfg in text8_large_loop text8_large_conf owt_large_loop owt_large_conf; do
        bash "$0" "$cfg" "$GPU" "$N_EVAL"
    done
    ;;

all)
    bash "$0" all_text8 "$GPU" "$N_EVAL"
    bash "$0" all_owt "$GPU" "$N_EVAL"
    ;;

*)
    echo "Usage: bash exp/run_remdm.sh <config> <gpu_id> [N_eval]"
    echo ""
    echo "Configs:"
    echo "  text8_small_loop,  text8_small_conf"
    echo "  text8_large_loop,  text8_large_conf"
    echo "  owt_large_loop,    owt_large_conf"
    echo ""
    echo "Batch:"
    echo "  all_text8, all_owt, all_large, all"
    echo ""
    echo "Args:"
    echo "  N_eval (optional, default=512): number of sequences used for sampling/eval"
    echo ""
    echo "Env (optional):"
    echo "  N_CHUNK=128       (chunk size for sampling; smaller => lower VRAM peak)"
    echo "  DEBUG_STATS=1     (enable --debug_stats)"
    echo "  NUCLEUS_P=0.9     (top-p on oracle p*(x0|xt), 1.0 disables)"
    echo "  ETA=0.02 T_ON=0.55 T_OFF=0.05 ALPHA_ON=0.9 (override loop params)"
    echo "  DUMP_SAMPLES=1    (OWT: dump token ids as .pt files)"
    echo "  DUMP_STEPS=\"\"     (empty => dump all steps in --steps; or e.g. \"32,128,512\")"
    echo "  DUMP_LIMIT=0      (0 => dump all N; else dump first K)"
    echo ""
    echo "Allocator:"
    echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (default set here; override if desired)"
    exit 1
    ;;
esac

echo "✅ Done!"
