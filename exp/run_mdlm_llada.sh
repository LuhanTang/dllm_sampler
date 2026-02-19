#!/bin/bash
# ============================================================
# LLaDA Experiments (NO-PROMPT, +text dumps +ids dumps)
# ============================================================
#
# Usage:
#   bash exp/run_mdlm_llada.sh <config_name> <gpu_id>
#
# Policy:
#   - Prompt is ALWAYS OFF (we do NOT call run_llada_prompt at all).
#   - Main runner is sampler.llada.run_llada (your updated runner).
#   - BOTH remasking modes dump:
#       * dumps/: samples_step*.txt + top_ngrams_step*.{json,txt}
#       * seqs/:  seqs_step*.pt  (token ids)
#
# Batch:
#   bash exp/run_mdlm_llada.sh all_text8 0
#   bash exp/run_mdlm_llada.sh all_owt   0
#   bash exp/run_mdlm_llada.sh all_large 0
#   bash exp/run_mdlm_llada.sh all       0
#
# Optional env override:
#   N_EVAL_DEFAULT=256 bash exp/run_mdlm_llada.sh all_owt 0
#
# NEW (Feb 2026):
#   - Chunked eval to reduce VRAM:
#       N_CHUNK_DEFAULT=128 bash exp/run_mdlm_llada.sh owt_large_random_t1 0
#     (passed to run_llada.py as --N_chunk)
#
#   - Save token-id sequences (default ON in run_llada.py):
#       SAVE_SEQS_DEFAULT=1 SAVE_MAX_SEQS_DEFAULT=256 bash exp/run_mdlm_llada.sh owt_large_random_t1 0
# ============================================================

set -e
cd "$(dirname "$0")/.."

CONFIG="${1:-owt_large_lowconf_t1}"
GPU="${2:-0}"
export CUDA_VISIBLE_DEVICES=$GPU

# -----------------------
# UPDATED GT PATHS (yours)
# -----------------------
GT_TEXT8_T128="gt_text8_char_T128_N1000_topk27_lam1e-4.pt"
GT_TEXT8_T1024="gt_text8_char_withSpace_T1024_N128_topk27_eps0.pt"
GT_OWT="sampler_gt/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt"

# Defaults
TEMP_BETA_DEFAULT="1.0"
N_EVAL_DEFAULT="${N_EVAL_DEFAULT:-512}"

# Chunk size for eval sampling (VRAM saver)
N_CHUNK_DEFAULT="${N_CHUNK_DEFAULT:-128}"

# Save seq ids knobs (run_llada.py default save_seqs=True)
# Use 1/0 to enable/disable from bash.
SAVE_SEQS_DEFAULT="${SAVE_SEQS_DEFAULT:-1}"
# -1 => save all N_eval (can be big); suggest 256 or 512 if disk is a concern.
SAVE_MAX_SEQS_DEFAULT="${SAVE_MAX_SEQS_DEFAULT:--1}"

# For OWT dumps/decoding (your run_llada.py default too, but we pass explicitly for clarity)
TOKENIZER_OWT_DEFAULT="tokenizers/owt_bytebpe_v4096/tokenizer.json"

# Dump knobs (now meaningful for BOTH remasking modes)
DUMP_TEXT_N_DEFAULT="${DUMP_TEXT_N_DEFAULT:-20}"
DUMP_TOPK_NGRAMS_DEFAULT="${DUMP_TOPK_NGRAMS_DEFAULT:-50}"
DUMP_NGRAM_NS_DEFAULT="${DUMP_NGRAM_NS_DEFAULT:-1,2,3}"

run_llada () {
  local GT="$1"
  local STEPS="$2"
  local REMASK="$3"
  local TEMP="$4"
  local TOKJSON="$5"   # may be empty

  local cmd=(python -m sampler.llada.run_llada
    --gt "$GT"
    --device cuda:0
    --steps "$STEPS"
    --remasking "$REMASK"
    --temperature "$TEMP"
    --temp_beta "$TEMP_BETA_DEFAULT"
    --N_eval "$N_EVAL_DEFAULT"
    --N_chunk "$N_CHUNK_DEFAULT"
  )

  # Save seq ids toggles
  if [[ "$SAVE_SEQS_DEFAULT" == "1" ]]; then
    cmd+=(--save_seqs --save_max_seqs "$SAVE_MAX_SEQS_DEFAULT")
  else
    cmd+=(--no-save_seqs)
  fi

  # Only pass tokenizer_json if provided (mainly OWT bpe)
  if [[ -n "$TOKJSON" ]]; then
    cmd+=(--tokenizer_json "$TOKJSON")
  fi

  # Dump flags
  cmd+=(--dump_text_n "$DUMP_TEXT_N_DEFAULT"
        --dump_topk_ngrams "$DUMP_TOPK_NGRAMS_DEFAULT"
        --dump_ngram_ns "$DUMP_NGRAM_NS_DEFAULT")

  echo "[CMD] ${cmd[*]}"
  "${cmd[@]}"
}

case "$CONFIG" in

# ============================================================
# TEXT8 small (T=128)
# ============================================================
text8_small_lowconf_t1)
    echo "[LLaDA][NO-PROMPT][text8] T=128 | remasking=low_confidence | temp=1.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_TEXT8_T128" "8,16,32,64,128,256" "low_confidence" "1.0" ""
    ;;

text8_small_random_t1)
    echo "[LLaDA][NO-PROMPT][text8] T=128 | remasking=random | temp=1.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_TEXT8_T128" "8,16,32,64,128,256" "random" "1.0" ""
    ;;

text8_small_lowconf_t0)
    echo "[LLaDA][NO-PROMPT][text8] T=128 | remasking=low_confidence | temp=0.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_TEXT8_T128" "8,16,32,64,128,256" "low_confidence" "0.0" ""
    ;;

text8_small_random_t0)
    echo "[LLaDA][NO-PROMPT][text8] T=128 | remasking=random | temp=0.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_TEXT8_T128" "8,16,32,64,128,256" "random" "0.0" ""
    ;;

# ============================================================
# TEXT8 large (T=1024)
# ============================================================
text8_large_lowconf_t1)
    echo "[LLaDA][NO-PROMPT][text8] T=1024 | remasking=low_confidence | temp=1.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_TEXT8_T1024" "8,16,32,64,128,256,512,1024" "low_confidence" "1.0" ""
    ;;

text8_large_random_t1)
    echo "[LLaDA][NO-PROMPT][text8] T=1024 | remasking=random | temp=1.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_TEXT8_T1024" "8,16,32,64,128,256,512,1024" "random" "1.0" ""
    ;;

text8_large_lowconf_t0)
    echo "[LLaDA][NO-PROMPT][text8] T=1024 | remasking=low_confidence | temp=0.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_TEXT8_T1024" "8,16,32,64,128,256,512,1024" "low_confidence" "0.0" ""
    ;;

text8_large_random_t0)
    echo "[LLaDA][NO-PROMPT][text8] T=1024 | remasking=random | temp=0.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_TEXT8_T1024" "8,16,32,64,128,256,512,1024" "random" "0.0" ""
    ;;

# ============================================================
# OWT (T=1024) + tokenizer_json for decoding dumps
# ============================================================
owt_large_lowconf_t1)
    echo "[LLaDA][NO-PROMPT][OWT] T=1024 | remasking=low_confidence | temp=1.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_OWT" "8,16,32,64,128,256,512,1024" "low_confidence" "1.0" "$TOKENIZER_OWT_DEFAULT"
    ;;

owt_large_random_t1)
    echo "[LLaDA][NO-PROMPT][OWT] T=1024 | remasking=random | temp=1.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_OWT" "8,16,32,64,128,256,512,1024" "random" "1.0" "$TOKENIZER_OWT_DEFAULT"
    ;;

owt_large_lowconf_t0)
    echo "[LLaDA][NO-PROMPT][OWT] T=1024 | remasking=low_confidence | temp=0.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_OWT" "8,16,32,64,128,256,512,1024" "low_confidence" "0.0" "$TOKENIZER_OWT_DEFAULT"
    ;;

owt_large_random_t0)
    echo "[LLaDA][NO-PROMPT][OWT] T=1024 | remasking=random | temp=0.0 | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    run_llada "$GT_OWT" "8,16,32,64,128,256,512,1024" "random" "0.0" "$TOKENIZER_OWT_DEFAULT"
    ;;

# ============================================================
# Batch
# ============================================================
all_text8)
    echo "Running ALL text8 (NO-PROMPT): large lowconf + random (temp=1.0) | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    for cfg in text8_large_lowconf_t1 text8_large_random_t1; do
        echo ""
        echo "====== $cfg ======"
        bash "$0" "$cfg" "$GPU"
    done
    ;;

all_owt)
    echo "Running ALL OWT (NO-PROMPT): lowconf + random (temp=1.0) | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    for cfg in owt_large_lowconf_t1 owt_large_random_t1; do
        echo ""
        echo "====== $cfg ======"
        bash "$0" "$cfg" "$GPU"
    done
    ;;

all_large)
    echo "Running ALL LARGE (NO-PROMPT): text8 + owt (temp=1.0) | N_eval=${N_EVAL_DEFAULT} | N_chunk=${N_CHUNK_DEFAULT}"
    for cfg in \
        owt_large_lowconf_t1  owt_large_random_t1 \
        text8_large_lowconf_t1 text8_large_random_t1
    do
        echo ""
        echo "====== $cfg ======"
        bash "$0" "$cfg" "$GPU"
    done
    ;;

all)
    bash "$0" all_text8 "$GPU"
    bash "$0" all_owt "$GPU"
    ;;

*)
    echo "Usage: bash exp/run_mdlm_llada.sh <config> <gpu_id>"
    echo ""
    echo "NO-PROMPT configs:"
    echo "  text8_small_lowconf_t1, text8_small_random_t1"
    echo "  text8_large_lowconf_t1, text8_large_random_t1"
    echo "  owt_large_lowconf_t1,   owt_large_random_t1"
    echo ""
    echo "Batch:"
    echo "  all_text8, all_owt, all_large, all"
    echo ""
    echo "Output behavior (BOTH remasking):"
    echo "  - dumps/: samples_step*.txt + top_ngrams_step*.*"
    echo "  - seqs/ : seqs_step*.pt (token ids)"
    echo ""
    echo "Env overrides:"
    echo "  N_EVAL_DEFAULT=256 N_CHUNK_DEFAULT=128 SAVE_SEQS_DEFAULT=1 SAVE_MAX_SEQS_DEFAULT=256 \\"
    echo "    DUMP_TEXT_N_DEFAULT=50 bash exp/run_mdlm_llada.sh owt_large_random_t1 0"
    exit 1
    ;;
esac

echo "✅ Done!"
