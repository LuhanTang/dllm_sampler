#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# AR bigram temperature sweep + end-to-end evaluation
#
# Metrics produced by the runner:
#   - GenPPL (gpt2-large, end-to-end)
#   - unigram_entropy (token-marginal entropy; sensitive to low-temp collapse)   [NEW]
#   - total_entropy (empirical entropy over FULL sequences; changes w/ duplicates)
#   - unique-1 / unique-2 / unique-3 (corpus-level n-gram uniqueness)
#   - dup_rate (sequence-level duplication)
# ------------------------------------------------------------

PRIOR_PT="sampler/priors/owt_gpt2_bigram_topk128_eps1e-5.pt"
BETAS="1.0,1.2,1.5,2.0,3.0,4.0"
SEED=123
GPU=${1:-0}

export CUDA_VISIBLE_DEVICES=${GPU}

echo "[INFO] Using GPU=${GPU}"
echo "[INFO] PRIOR_PT=${PRIOR_PT}"
echo "[INFO] BETAS=${BETAS}"
echo "[INFO] Metrics: GenPPL + unigram_entropy + total_entropy + unique-n + dup_rate"

# -------------------------
# Short demo (fast sanity check)
# -------------------------
python -m sampler.run_ar_temp_genppl_gpt2 \
  --prior_pt "${PRIOR_PT}" \
  --out_dir "sampler_output/gpt2_bigram_demo_T128N256" \
  --T 128 --N 256 \
  --betas "${BETAS}" \
  --device cuda:0 \
  --genppl_batch 8 \
  --seed ${SEED}

# -------------------------
# Paper setting (long context)
#   - End-to-end regime where GenPPL can decrease
#     while unigram entropy + unique-n collapse
#   - Note: total_entropy may stay ~log(N) if dup_rate ~ 0
# -------------------------
python -m sampler.run_ar_temp_genppl_gpt2 \
  --prior_pt "${PRIOR_PT}" \
  --out_dir "sampler_output/gpt2_bigram_demo_T1024N128" \
  --T 1024 --N 128 \
  --betas "${BETAS}" \
  --device cuda:0 \
  --genppl_batch 4 \
  --seed ${SEED}

echo "[DONE] Results written to:"
echo "  - sampler_output/gpt2_bigram_demo_T128N256/"
echo "  - sampler_output/gpt2_bigram_demo_T1024N128/"
