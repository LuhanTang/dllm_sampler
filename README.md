# DLLM Sampler

This repository is the code release for our paper on **sampler-centric evaluation** of discrete diffusion language models (dLLMs).

## Project Summary

dLLMs provide fast and flexible parallel token updates, but are harder to evaluate than autoregressive models (ARMs).  
Existing metrics often mix two error sources:

1. denoiser approximation error
2. sampler-induced error from sampling dynamics

We introduce an oracle evaluation framework that replaces the learned denoiser with an exact HMM posterior from a ground-truth Markov chain, so sampler error can be isolated under method-consistent settings.

Main findings:

1. Few-step discrete diffusion samplers are not distributionally correct, even with an oracle denoiser.
2. Transition-level mismatch is substantial at small step counts.
3. The mismatch diminishes only when the number of diffusion steps approaches sequence length.
4. Better `NLL / GenPPL / MAUVE` does not necessarily imply correct sampling.

## Repository Layout

```text
exp/
  build_gt.sh
  build_tokenizer.sh
  run_mdlm_llada.sh
  run_sedd.sh
  run_remdm.sh
  run_all.sh
  run_genppl_mauve.sh

sampler/
  oracle_hmm_posterior.py
  metrics_full.py
  gt_io.py
  build_text8_char_gt_k27_eps0.py
  eval_owt_mauve_genppl.py
  ar/
    run_ar_temp_genppl_gpt2.py
  llada/
    run_llada.py
  sedd/
    run_sedd.py
    run_sedd_genppl.py
    run_sedd_mauve.py
  remdm/
    run_remdm.py

tokenizers/
  train_owt_bytebpe_tokenizer.py
  build_owt_bpe_gt.py
  inspect_gt_samples.py
  owt_bytebpe_v4096/tokenizer.json
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Recommended Workflow

### 1. Prepare GT data

For text8:

```bash
bash exp/build_gt.sh
```

For OWT (tokenizer + GT):

```bash
bash exp/build_tokenizer.sh all
```

`build_tokenizer.sh` modes:
- `tokenizer`
- `gt`
- `all`
- `inspect`

### 2. Run samplers

text8:

```bash
bash exp/run_mdlm_llada.sh all_text8 0
bash exp/run_sedd.sh all_text8 0
bash exp/run_remdm.sh all_text8 0 512
```

OWT:

```bash
bash exp/run_mdlm_llada.sh all_owt 0
bash exp/run_sedd.sh all_owt 0
bash exp/run_remdm.sh all_owt 0 512
```

Unified runner:

```bash
bash exp/run_all.sh text8 0 512
bash exp/run_all.sh owt 0 512
bash exp/run_all.sh large 0 512
bash exp/run_all.sh all 0 512
```

## LLaDA vs MDLM in This Repo

In `exp/run_mdlm_llada.sh` + `sampler.llada.run_llada`:

- `--remasking low_confidence` => LLaDA
- `--remasking random` => MDLM

## Metrics (Highlight)

`sampler/metrics_full.py` is the central unified evaluation module.

Common metrics:

- `nll_token`
- `full_kl_rate`
- `full_tv_rate`
- `unigram_L1`
- `dup_rate`
- n-gram diversity metrics

Recommendation: prioritize transition/distribution metrics (`full_kl_rate`, `full_tv_rate`) and interpret `NLL`, `GenPPL`, and `MAUVE` jointly.

## MAUVE / GenPPL Related Files

- `sampler/eval_owt_mauve_genppl.py`
- `sampler/ar/run_ar_temp_genppl_gpt2.py`
- `exp/run_genppl_mauve.sh`

## Outputs

- `sampler_output/`: run outputs and metrics
- `sampler_plots/`: generated plots

## Release Scope

This release focuses on reproducible sampler evaluation pipelines (`exp/`, `sampler/`, `tokenizers/`).
Interactive visualization tools are not part of the current release scope.

## Documentation

For detailed reproduction instructions, see `TUTORIAL.md`.
