# Experiment Scripts

This folder contains all experiment scripts for running and comparing different discrete diffusion samplers.

## Directory Structure

```
exp/
├── README.md                      # This file
├── build_gt.sh                    # Build Ground Truth files
├── build_tokenizer.sh             # Build shared BPE tokenizer
├── run_mdlm_llada.sh              # MDLM / LLaDA experiments
├── run_remdm.sh                   # ReMDM experiments
├── run_sedd.sh                    # SEDD experiments
├── run_all.sh                     # Run all samplers
└── results/                       # Output directory
```

## Quick Start

### 1. Build GT Files

```bash
# Small scale (T=128)
./build_gt.sh text8_small

# Large scale (T=1024)
./build_gt.sh text8_large
```

### 2. Run Experiments

```bash
# Run a single config
./run_mdlm_llada.sh text8_large_lowconf_t1 0
./run_remdm.sh text8_large_conf 0 512
./run_sedd.sh text8_large_accurate 0

# Run all text8 experiments
./run_all.sh text8 0 512

# Run all large experiments
./run_all.sh large 0
```

## Experiment Configurations

### LLaDA

| Config | T | Remasking | Temperature |
|--------|---|-----------|-------------|
| `small_lowconf_t0` | 128 | low_confidence | 0.0 |
| `small_lowconf_t1` | 128 | low_confidence | 1.0 |
| `small_random_t0` | 128 | random | 0.0 |
| `small_random_t1` | 128 | random | 1.0 |
| `large_*` | 1024 | (same as small) | (same) |

**Key Finding**: Temperature is important for LLaDA with HMM oracle!
- `temp=0`: KL ≈ 3.0 (bad, positions have similar distributions)
- `temp=1`: KL ≈ 1.3 (much better, Gumbel noise adds diversity)

### ReMDM

| Config | T | Sampler | Eta |
|--------|---|---------|-----|
| `small_rescale` | 128 | rescale only | 0.9 |
| `small_conf` | 128 | rescale + confidence | 0.9 |
| `large_*` | 1024 | (same) | (same) |

### SEDD

| Config | T | Sampler | Temp Beta |
|--------|---|---------|-----------|
| `small_accurate` | 128 | accurate | 1.0 |
| `small_inaccurate` | 128 | inaccurate | 1.2 |
| `large_accurate` | 1024 | accurate | 1.0 |
| `large_inaccurate` | 1024 | inaccurate | 1.06 |

## GT Files Required

| Experiment | GT File |
|------------|---------|
| Small (T=128) | `gt_text8_char_T128_N1000_topk27_lam1e-4.pt` |
| Large (T=1024) | `sampler_gt/gt_text8_char_withSpace_T1024_N128_topk27_eps0.pt` |
| OWT | `sampler_gt/gt_owt_bytebpe_T1024_N1000_autoK_p90_mass0.99_eps1e-4_seed123.pt` |
| Stack-Python | `gt_stackpy_T1024_N1000_sharedV4096_*.pt` |

## Outputs

Results are saved to:
- `sampler_output/{llada,remdm,sedd}/` - Metrics (JSON, CSV)
- `sampler_plots/{llada,remdm,sedd}/` - Plots (PNG)
