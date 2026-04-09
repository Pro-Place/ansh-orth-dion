"""
Experiment configurations for Dion-family optimizer ablations.

Default settings:
  - Model: SmallGPT2 (~45M params)
  - Dataset: WikiText-103
  - Rank: 64
  - Steps: 30,000
  - Batch size: 8
  - LR: 0.01, cosine schedule
  - Sequence length: 1024
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    hidden: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.0
    tie_weights: bool = True


@dataclass
class TrainConfig:
    max_steps: int = 30000
    batch_size: int = 8
    seq_len: int = 1024
    lr: float = 0.01
    lr_schedule: str = "cosine"     # "cosine" or "constant"
    warmup_steps: int = 3000          # 10% of 30k
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    eval_interval: int = 500
    eval_steps: int = 50            # number of val batches per eval
    log_interval: int = 100
    diag_interval: int = 500        # full diagnostics every N steps
    save_interval: int = 5000
    seed: int = 42
    device: str = "cuda"
    compile_model: bool = False
    num_workers: int = 2


@dataclass
class OptimizerConfig:
    """Configuration for a single Dion variant experiment."""
    variant: str = "stripped_dion"   # key into VARIANT_REGISTRY
    rank: int = 64
    beta: float = 1.0
    mu: float = 0.0
    power_iters: int = 1
    collect_diagnostics: bool = True

    # Variant-specific kwargs (passed to constructor)
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Scalar optimizer for non-matrix params
    scalar_lr: float = 3e-4
    scalar_betas: tuple = (0.9, 0.999)
    scalar_weight_decay: float = 0.1
    scalar_eps: float = 1e-8


@dataclass
class ExperimentConfig:
    """Full experiment specification."""
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    output_dir: str = "results"


# =========================================================================
# Pre-defined experiment suites
# =========================================================================

def _base_config(name: str, **opt_kwargs) -> ExperimentConfig:
    """Create a base config with default settings."""
    opt = OptimizerConfig(**opt_kwargs)
    return ExperimentConfig(
        name=name,
        model=ModelConfig(),
        train=TrainConfig(),
        optimizer=opt,
    )


# --- 1. Beta sweep ---

BETA_SWEEP_CONFIGS = {
    f"dion_beta_{b}": _base_config(
        f"dion_beta_{b}",
        variant="stripped_dion",
        rank=64,
        beta=b,
    )
    for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

# --- 2. Orth-Dion (QR right factor) ---

ORTH_DION_CONFIGS = {
    f"orth_dion_beta_{b}": _base_config(
        f"orth_dion_beta_{b}",
        variant="orth_dion",
        rank=64,
        beta=b,
    )
    for b in [0.3, 0.5, 0.7, 1.0]
}

# --- 3. Soft-Isometry AdaDion (the key new idea) ---

SOFT_ISOMETRY_CONFIGS = {}
for s_mode in ["normalized", "clipped", "fixed"]:
    for beta in [0.3, 0.5, 1.0]:
        name = f"soft_isometry_{s_mode}_beta_{beta}"
        SOFT_ISOMETRY_CONFIGS[name] = _base_config(
            name,
            variant="soft_isometry",
            rank=64,
            beta=beta,
            extra_kwargs={"s_mode": s_mode, "s_floor": 0.1},
        )

# --- 4. Modewise-beta Dion ---

MODEWISE_CONFIGS = {}
for b_mode in ["uniform", "per_mode", "per_mode_persistence"]:
    for use_orth in [True, False]:
        orth_tag = "orth" if use_orth else "colnorm"
        name = f"modewise_{b_mode}_{orth_tag}"
        MODEWISE_CONFIGS[name] = _base_config(
            name,
            variant="modewise_beta",
            rank=64,
            beta=0.3,
            extra_kwargs={
                "b_mode": b_mode,
                "use_orth": use_orth,
                "b_target_reentry": 0.3,
            },
        )

# --- 5. PA-Dion (persistence-aware) ---

PA_DION_CONFIGS = {
    "pa_dion_default": _base_config(
        "pa_dion_default",
        variant="pa_dion",
        rank=64,
        extra_kwargs={
            "beta_min": 0.0, "beta_max": 1.0,
            "sigmoid_sharpness": 10.0, "sigmoid_tau": 0.0,
        },
    ),
    "pa_dion_low_beta_max": _base_config(
        "pa_dion_low_beta_max",
        variant="pa_dion",
        rank=64,
        extra_kwargs={
            "beta_min": 0.0, "beta_max": 0.5,
            "sigmoid_sharpness": 10.0, "sigmoid_tau": 0.0,
        },
    ),
}

# --- 6. RNormDion (R_norm targeting) ---

RNORM_CONFIGS = {
    f"rnorm_target_{t}": _base_config(
        f"rnorm_target_{t}",
        variant="rnorm_dion",
        rank=64,
        extra_kwargs={"R_target": t, "k_p": 0.3},
    )
    for t in [0.6, 1.0, 1.2, 1.5, 1.8, 2.5]
}

# --- 7. ReEntryDion (re-entry norm controller) ---

REENTRY_CONFIGS = {
    f"reentry_target_{t}": _base_config(
        f"reentry_target_{t}",
        variant="reentry_dion",
        rank=64,
        extra_kwargs={"reentry_target": t, "k_p": 0.5},
    )
    for t in [0.1, 0.3, 0.5, 0.8, 1.0]
}

# --- 8. PolyakDion (explicit momentum baseline) ---

POLYAK_CONFIGS = {
    f"polyak_mu_{mu}": _base_config(
        f"polyak_mu_{mu}",
        variant="polyak_dion",
        rank=64,
        mu=mu,
    )
    for mu in [0.8, 0.9, 0.95, 0.99]
}

# --- 9. Local response-surface experiment (MY_RESPONSE §experiment 3) ---
# Freeze a checkpoint and scale the buffer by lambda: M(lambda) = G + lambda * R
# This is handled in train.py as a special mode, not a separate optimizer config.

LAMBDA_SWEEP_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


# --- Rank sweep (to check rank interaction with beta) ---

RANK_SWEEP_CONFIGS = {}
for r in [16, 32, 64, 128, 256]:
    for beta in [0.3, 1.0]:
        name = f"dion_r{r}_beta_{beta}"
        RANK_SWEEP_CONFIGS[name] = _base_config(
            name,
            variant="stripped_dion",
            rank=r,
            beta=beta,
        )


# =========================================================================
# Experiment suites (groups of experiments to run together)
# =========================================================================

EXPERIMENT_SUITES = {
    # Core beta sweep reproduction
    "beta_sweep": {
        **{k: v for k, v in BETA_SWEEP_CONFIGS.items()
           if any(k.endswith(f"_{b}") for b in ["0.0", "0.3", "0.5", "0.7", "1.0"])},
        "polyak_mu_0.95": POLYAK_CONFIGS["polyak_mu_0.95"],
    },

    # Geometry vs memory ablation
    "geometry_vs_memory": {
        # Stripped Dion baselines
        "dion_beta_0.3": BETA_SWEEP_CONFIGS["dion_beta_0.3"],
        "dion_beta_1.0": BETA_SWEEP_CONFIGS["dion_beta_1.0"],
        # Orth-Dion
        "orth_dion_beta_0.3": ORTH_DION_CONFIGS["orth_dion_beta_0.3"],
        "orth_dion_beta_1.0": ORTH_DION_CONFIGS["orth_dion_beta_1.0"],
        # Soft-Isometry
        "soft_isometry_normalized_beta_0.3": SOFT_ISOMETRY_CONFIGS["soft_isometry_normalized_beta_0.3"],
        "soft_isometry_normalized_beta_1.0": SOFT_ISOMETRY_CONFIGS["soft_isometry_normalized_beta_1.0"],
        # Modewise
        "modewise_per_mode_orth": MODEWISE_CONFIGS["modewise_per_mode_orth"],
        "modewise_per_mode_colnorm": MODEWISE_CONFIGS["modewise_per_mode_colnorm"],
    },

    # Controller comparison
    "controllers": {
        "dion_beta_0.3": BETA_SWEEP_CONFIGS["dion_beta_0.3"],
        "pa_dion_default": PA_DION_CONFIGS["pa_dion_default"],
        "pa_dion_low_beta_max": PA_DION_CONFIGS["pa_dion_low_beta_max"],
        "rnorm_target_1.8": RNORM_CONFIGS["rnorm_target_1.8"],
        "reentry_target_0.3": REENTRY_CONFIGS["reentry_target_0.3"],
        "reentry_target_0.5": REENTRY_CONFIGS["reentry_target_0.5"],
    },

    # Full beta sweep
    "beta_sweep": BETA_SWEEP_CONFIGS,

    # Rank x beta interaction
    "rank_sweep": RANK_SWEEP_CONFIGS,

    # R_norm sweep
    "rnorm_sweep": RNORM_CONFIGS,

    # ReEntry sweep
    "reentry_sweep": REENTRY_CONFIGS,

    # Everything
    "full": {
        **BETA_SWEEP_CONFIGS,
        **ORTH_DION_CONFIGS,
        **SOFT_ISOMETRY_CONFIGS,
        **MODEWISE_CONFIGS,
        **PA_DION_CONFIGS,
        **RNORM_CONFIGS,
        **REENTRY_CONFIGS,
        **POLYAK_CONFIGS,
        **RANK_SWEEP_CONFIGS,
    },
}
