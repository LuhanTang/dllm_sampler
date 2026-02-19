# DLLM Sampler Tutorial

This tutorial is for reproducibility and code navigation for paper readers.

## 1. Evaluation Goal

This project is designed to evaluate **sampler behavior** in dLLMs, not to compare mixed training + sampling errors.

Under the oracle setup:

1. Ground truth is a controllable Markov chain.
2. The learned denoiser is replaced by an exact HMM posterior.
3. Different samplers are compared under the same target distribution.

Key takeaway: improvements in `NLL`, `GenPPL`, or `MAUVE` alone do not guarantee distributionally correct sampling.

## 2. Environment

Recommended: Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3. Data and GT Preparation

Two active tracks in this repo:
- `text8`
- `owt`

### 3.1 text8

```bash
bash exp/build_gt.sh
```

### 3.2 OWT (Tokenizer + GT)

```bash
bash exp/build_tokenizer.sh all
```

Or step-by-step:

```bash
bash exp/build_tokenizer.sh tokenizer
bash exp/build_tokenizer.sh gt
bash exp/build_tokenizer.sh inspect
```

Back-end scripts:

- `tokenizers/train_owt_bytebpe_tokenizer.py`
- `tokenizers/build_owt_bpe_gt.py`
- `tokenizers/inspect_gt_samples.py`

## 4. Running Samplers

### 4.1 LLaDA / MDLM

```bash
bash exp/run_mdlm_llada.sh all_text8 0
bash exp/run_mdlm_llada.sh all_owt 0
```

Mapping in this repo:
- `low_confidence` = LLaDA
- `random` = MDLM

### 4.2 SEDD

```bash
bash exp/run_sedd.sh all_text8 0
bash exp/run_sedd.sh all_owt 0
```

### 4.3 ReMDM

```bash
bash exp/run_remdm.sh all_text8 0 512
bash exp/run_remdm.sh all_owt 0 512
```

The third argument is `N_eval`.

### 4.4 Unified Batch Runner

```bash
bash exp/run_all.sh text8 0 512
bash exp/run_all.sh owt 0 512
bash exp/run_all.sh large 0 512
bash exp/run_all.sh all 0 512
```

## 5. Metrics (Important)

Unified metric implementation:

- `sampler/metrics_full.py`

Recommended minimum metric set:

1. transition/distribution mismatch: `full_kl_rate`, `full_tv_rate`
2. likelihood-related: `nll_token`
3. coverage/diversity: `unigram_L1`, `dup_rate`, n-gram metrics

Use joint interpretation rather than a single scalar metric.

## 6. GenPPL and MAUVE

Related files:

- `sampler/eval_owt_mauve_genppl.py`
- `sampler/sedd/run_sedd_genppl.py`
- `sampler/sedd/run_sedd_mauve.py`
- `sampler/ar/run_ar_temp_genppl_gpt2.py`
- `exp/run_genppl_mauve.sh`

Treat these as complementary metrics to `metrics_full.py`, not replacements.

## 7. Outputs

- `sampler_output/`: per-run metrics and artifacts
- `sampler_plots/`: plots and visual summaries

For reproducibility, fix:

1. `seed`
2. `N_eval`
3. step list
4. script/version used for evaluation

## 8. Suggested Reading Order

1. `README.md` (problem + main findings)
2. `exp/` scripts (how to run)
3. `sampler/metrics_full.py` (how quality is measured)
4. `TUTORIAL.md` (reproducibility details)
