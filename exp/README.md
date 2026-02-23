# Experiment Scripts

This folder contains the runnable experiment entrypoints used in this repo.

## Directory Structure

```text
exp/
├── README.md
├── build_gt.sh
├── build_tokenizer.sh
├── run_mdlm_llada.sh
├── run_remdm.sh
├── run_sedd.sh
└── run_all.sh
```

## Quick Start

Run all commands from repo root.

### 1. Prepare data

If you only need existing tracked artifacts, make sure Git LFS files are pulled:

```bash
git lfs pull
```

If you need to (re)build OWT tokenizer + OWT GT:

```bash
bash exp/build_tokenizer.sh all
```

If you already have tokenizer and only want OWT GT:

```bash
bash exp/build_tokenizer.sh gt
```

### 2. Run experiments

Run one config per sampler:

```bash
bash exp/run_mdlm_llada.sh text8_large_lowconf_t1 0
bash exp/run_sedd.sh text8_large_accurate 0
bash exp/run_remdm.sh text8_large_conf 0 512
```

Run batched experiments:

```bash
bash exp/run_all.sh text8 0 512
bash exp/run_all.sh owt 0 512
bash exp/run_all.sh large 0 512
bash exp/run_all.sh all 0 512
```

## Experiment Configurations

### `run_mdlm_llada.sh`

Single configs:
- `text8_small_lowconf_t1`
- `text8_small_random_t1`
- `text8_small_lowconf_t0`
- `text8_small_random_t0`
- `text8_large_lowconf_t1`
- `text8_large_random_t1`
- `text8_large_lowconf_t0`
- `text8_large_random_t0`
- `owt_large_lowconf_t1`
- `owt_large_random_t1`
- `owt_large_lowconf_t0`
- `owt_large_random_t0`

Batch configs:
- `all_text8`
- `all_owt`
- `all_large`
- `all`

Notes:
- `--remasking low_confidence` = LLaDA.
- `--remasking random` = MDLM.

### `run_sedd.sh`

Single configs:
- `text8_small_accurate`
- `text8_small_inaccurate`
- `text8_large_accurate`
- `text8_large_inaccurate`
- `owt_large_accurate`
- `owt_large_inaccurate`

Batch configs:
- `all_text8`
- `all_owt`
- `all_large`
- `all`

Example: tune nucleus (top-p) in ReMDM with `NUCLEUS_P`:

```bash
NUCLEUS_P=0.9 bash exp/run_remdm.sh text8_large_conf 0 512
NUCLEUS_P=1   bash exp/run_remdm.sh owt_large_loop 0 512
```

### `run_remdm.sh`

Single configs:
- `text8_small_loop`
- `text8_small_conf`
- `text8_large_loop`
- `text8_large_conf`
- `owt_large_loop`
- `owt_large_conf`

Batch configs:
- `all_text8`
- `all_owt`
- `all_large`
- `all`

### `run_all.sh`

Targets:
- `text8`
- `owt`
- `large`
- `all`

Usage:

```bash
bash exp/run_all.sh <target> <gpu_id> [N_eval]
```

## GT Files Required

| Setting | GT file |
|---|---|
| text8 small (T=128) | `gt_text8_char_T128_N1000_topk27_lam1e-4.pt` |
| text8 large (T=1024) | `sampler_gt/gt_text8_char_withSpace_T1024_N128_topk27_eps0.pt` |
| OWT large (T=1024) | `sampler_gt/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt` |

## Outputs

Outputs are written by sampler scripts under:
- `sampler_output/`
- `sampler_plots/`

## Useful Env Overrides

Commonly used knobs:
- `N_EVAL_DEFAULT` and `N_CHUNK_DEFAULT` for `run_mdlm_llada.sh`
- `N_EVAL`, `SAVE_SEQS`, `SAVE_MAX_SEQS` for `run_sedd.sh`
- `N_CHUNK`, `NUCLEUS_P`, `DUMP_SAMPLES`, `DUMP_STEPS`, `DUMP_LIMIT` for `run_remdm.sh`
