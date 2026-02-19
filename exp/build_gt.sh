set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

echo "[RUN] python ${REPO_ROOT}/tokenizers/build_owt_bpe_gt.py $*"

python "${REPO_ROOT}/tokenizers/build_owt_bpe_gt.py" \
  --out sampler_gt/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt \
  --gt_samples_out sampler_gt/gt_samples_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt \
  --tokenizer_dir tokenizers/owt_bytebpe_v4096 \
  --owt_name stanford-cs336/owt-sample \
  --split train \
  --encode_batch_size 128 \
  --max_tokens_for_counts 2000000 \
  --max_chars_per_doc 4096 \
  --log_every_docs 2000 \
  --normalize_ws \
  --strip \
  --T 1024 \
  --N 1000 \
  --seed 123 \
  --nu unigram \
  --pi_iters 500 \
  --plot_dir sampler_plots/gt_build/owt_bytebpe_T1024_autoK_p90_mass0.99_eps1e-4_seed123 \
  --auto_topk \
  --target_mass 0.99 \
  --k_strategy p90 \
  --teleport_eps 1e-4
