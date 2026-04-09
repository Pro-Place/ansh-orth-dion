# Ada-Dion V3

Per-mode adaptive error feedback for communication-efficient spectral optimization.

Ada-Dion V3 extends the [Dion](https://arxiv.org/abs/2504.05295) optimizer with three modifications: per-mode adaptive feedback coefficients, reduced base feedback coefficient ($\beta_0 = 0.1$), and optional adaptive rank. These changes are motivated by a systematic experimental investigation of the error-feedback mechanism across model scales from 45M to 300M parameters.

## Results

**LLaMA 300M** (WikiText-103, 15k steps, tuned LRs):

| Method | Val Loss | Comm (GB) | Mem (GB) |
|--------|----------|-----------|----------|
| Ada-Dion V3 Adaptive | 3.403 | 11,998 | 14.1 |
| Dion $\beta$=0.3 | 3.565 | 1,456 | 22.0 |
| Ada-Dion V3 | 3.638 | 1,514 | 13.3 |
| Ada-Dion V2 | 3.677 | 6,826 | 13.3 |
| Dion $\beta$=1.0 | 3.918 | 1,456 | 22.0 |

**FashionMNIST** (3-layer MLP, 50 epochs):

| Method | Test Acc | Comm (GB) |
|--------|----------|-----------|
| V3 Adaptive | 91.12% | 10.26 |
| AdamW | 90.95% | 272.40 |
| V3 | 90.74% | 21.71 |
| Dion | 90.46% | 25.02 |

## Repository Structure

```
benchmark/
  adadion_v3/
    adadion_v3.py            # Ada-Dion V3 optimizer
  lm/
    dion_variants.py         # All Dion variants (StrippedDion, OrthDion, PADion, etc.)
    model.py                 # GPT-2 45M model
    llama.py                 # LLaMA 300M model
    data.py                  # WikiText-103 data loading
    train.py                 # Training loop (GPT-2 experiments)
    llama_comparison.py      # LLaMA 300M comparison with HP tuning
    configs.py               # Experiment configurations
    right_factor_ablation.py # ColNorm vs QR investigation
    scale_diagnostics.py     # Per-parameter diagnostics at scale
    response_surface.py      # Buffer scaling analysis
    modewise_persistence.py  # Per-mode gradient persistence study
  adadion_v2_single.py       # Ada-Dion V2 (single-GPU reimplementation)
  full_comparison.py         # CIFAR-10 / FashionMNIST comparison
writeup/
  main.tex                   # Paper draft
  references.bib             # Bibliography
  figs/                      # Generated figures
  generate_final_figures.py  # Figure generation script
```

## Quick Start

### Install dependencies

```bash
pip install torch torchvision datasets transformers
```

### Run the beta sweep (GPT-2 45M, ~4 hours on 1 GPU)

```bash
python -m benchmark.lm.train --suite beta_sweep --max_steps 30000 --device cuda
```

### Run a single experiment

```bash
python -m benchmark.lm.train --variant stripped_dion --beta 0.3 --rank 64 --max_steps 30000
```

### Run the LLaMA 300M comparison with HP tuning

```bash
python -m benchmark.lm.llama_comparison --mode full --batch_size 4 --device cuda
```

### Run the CIFAR-10 / FashionMNIST comparison

```bash
python -m benchmark.full_comparison --dataset all --epochs 100 --device cuda
```

## Ablations

### Right-factor normalization (ColNorm vs QR vs partial orth)

```bash
python -m benchmark.lm.right_factor_ablation --suite core --max_steps 10000 --device cuda
```

### Per-mode gradient persistence

```bash
python -m benchmark.lm.modewise_persistence --max_steps 10000 --rank 64 --beta 0.3
```

### Response surface (buffer scaling analysis)

```bash
python -m benchmark.lm.response_surface --checkpoint results/lm/.../ckpt_step15000.pt
```

### Scale diagnostics (per-parameter behavior at 300M)

```bash
python -m benchmark.lm.scale_diagnostics --max_steps 500 --device cuda
```

## Optimizer Variants

All variants share the same power-iteration infrastructure and are implemented in `benchmark/lm/dion_variants.py`:

| Variant | Right Factor | Beta | Per-Mode | Description |
|---------|-------------|------|----------|-------------|
| `stripped_dion` | ColNorm | configurable | No | Standard Dion with configurable beta |
| `orth_dion` | QR | configurable | No | QR right factor (nu_t = 1) |
| `soft_isometry` | QR + diag S | configurable | No | QR with diagonal scaling |
| `modewise_beta` | configurable | configurable | Yes | Per-mode feedback coefficients |
| `pa_dion` | ColNorm | rho-adaptive | No | Persistence-aware adaptive beta |
| `rnorm_dion` | ColNorm | R_norm-adaptive | No | Targets buffer-to-gradient ratio |
| `reentry_dion` | ColNorm | reentry-adaptive | No | Targets re-entry norm |
| `polyak_dion` | ColNorm | N/A | No | Explicit Polyak momentum, R=0 |

The V3 optimizer (`benchmark/adadion_v3/adadion_v3.py`) combines ColNorm, per-mode adaptive beta, buffer ceiling, and optional adaptive rank in a single class.

## Key Findings

1. **Buffer retention ($\beta < 1$)** improves over standard Dion ($\beta = 1$) by 0.4+ nats at zero additional compute. The benefit is consistent across all ranks and scales tested.

2. **QR orthogonalization fails** at all ranks and scales. The theoretical $\sqrt{r}$ rate improvement does not materialize because the spectral gap assumption does not hold in LLM training.

3. **Per-mode persistence heterogeneity is 28x.** Individual singular modes have very different gradient persistence characteristics, motivating per-mode control.

4. **$\|R\|/\|G\|$ is the fundamental operating variable** for error feedback. Different control mechanisms converge to the same result when targeting the same ratio.

## Citation

```
@misc{adadionv3,
  title={Ada-Dion V3: Per-Mode Adaptive Error Feedback for Communication-Efficient Spectral Optimization},
  author={Tiwari, Ansh},
  year={2026}
}
```
