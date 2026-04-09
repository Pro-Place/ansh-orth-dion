"""
Hyperparameter configurations and sweep grids for CIFAR-10 optimizer benchmark.

Each optimizer gets a carefully tuned default config plus a grid for sweeps.
Learning rates are scaled for CIFAR-10 (smaller models than LLM pretraining).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """Shared training configuration."""
    # Data
    dataset: str = "cifar10"
    batch_size: int = 128
    num_workers: int = 4

    # Training
    epochs: int = 200
    warmup_epochs: int = 5
    lr_schedule: str = "cosine"  # cosine, linear, constant
    min_lr_factor: float = 0.0
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Logging
    log_interval: int = 50  # steps
    eval_interval: int = 1  # epochs
    save_checkpoints: bool = True
    checkpoint_interval: int = 50  # epochs

    # Model
    model_name: str = "resnet18"  # resnet18, resnet34, vgg16_bn, vit_small

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True  # use torch.amp


@dataclass
class AdamWConfig:
    """AdamW hyperparameters."""
    name: str = "adamw"
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.05
    eps: float = 1e-8


@dataclass
class MuonConfig:
    """Muon hyperparameters for CIFAR-10."""
    name: str = "muon"
    # Matrix params (Muon)
    lr: float = 0.02
    mu: float = 0.95
    weight_decay: float = 0.01
    nesterov: bool = True
    adjust_lr: str = "rms_norm"  # spectral_norm, rms_norm, none
    flatten: bool = True  # flatten conv layers to 2D
    # Scalar params (AdamW)
    scalar_lr: float = 1e-3
    scalar_betas: tuple = (0.9, 0.999)
    scalar_weight_decay: float = 0.05
    scalar_eps: float = 1e-8


@dataclass
class DionConfig:
    """Dion hyperparameters for CIFAR-10."""
    name: str = "dion"
    # Matrix params (Dion)
    lr: float = 0.02
    rank_fraction: float = 0.25
    weight_decay: float = 0.01
    # Scalar params (AdamW)
    scalar_lr: float = 1e-3
    scalar_betas: tuple = (0.9, 0.999)
    scalar_weight_decay: float = 0.05
    scalar_eps: float = 1e-8


@dataclass
class Dion2Config:
    """Dion2 hyperparameters for CIFAR-10."""
    name: str = "dion2"
    # Matrix params (Dion2)
    lr: float = 0.02
    fraction: float = 0.25
    ef_decay: float = 0.95
    weight_decay: float = 0.01
    adjust_lr: str = "spectral_norm"
    flatten: bool = True
    # Scalar params (AdamW)
    scalar_lr: float = 1e-3
    scalar_betas: tuple = (0.9, 0.999)
    scalar_weight_decay: float = 0.05
    scalar_eps: float = 1e-8


@dataclass
class AdaDionConfig:
    """AdaDion V2 hyperparameters for CIFAR-10. Aligned with official config_registry.py."""
    name: str = "adadion"
    # Matrix params (AdaDion)
    lr: float = 0.012
    mu: float = 0.95
    rank_fraction_max: float = 0.7
    weight_decay: float = 0.1
    # Adaptive rank — aligned with official defaults
    adaptive_rank: bool = True
    init_rank_fraction: float = 0.5
    erank_ema_beta: float = 0.5
    rank_scale: float = 1.5
    rank_min: int = 16
    rank_quantize: int = 8
    rank_step_up: int = 16
    rank_step_down: int = 8
    # Scalar params (AdamW)
    scalar_lr: float = 0.012
    scalar_betas: tuple = (0.95, 0.95)
    scalar_weight_decay: float = 0.1
    scalar_eps: float = 1e-8


# Hyperparameter sweep grids

SWEEP_GRIDS = {
    "adamw": {
        "lr": [3e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        "weight_decay": [0.01, 0.05, 0.1],
    },
    "muon": {
        "lr": [0.005, 0.01, 0.02, 0.04],
        "mu": [0.9, 0.95],
        "weight_decay": [0.0, 0.01, 0.05],
        "nesterov": [True, False],
        "adjust_lr": ["spectral_norm", "rms_norm"],
    },
    "dion": {
        "lr": [0.005, 0.01, 0.02, 0.04],
        "rank_fraction": [0.125, 0.25, 0.5],
        "weight_decay": [0.0, 0.01, 0.05],
    },
    "dion2": {
        "lr": [0.005, 0.01, 0.02, 0.04],
        "fraction": [0.125, 0.25, 0.5],
        "ef_decay": [0.9, 0.95, 0.99],
        "weight_decay": [0.0, 0.01, 0.05],
    },
    "adadion": {
        "lr": [0.005, 0.01, 0.02, 0.04],
        "init_rank_fraction": [0.125, 0.25, 0.5],
        "rank_scale": [1.0, 1.5, 2.0],
        "weight_decay": [0.0, 0.01, 0.05],
        "erank_ema_beta": [0.5, 0.9],
    },
}


# Model configurations

MODEL_CONFIGS = {
    "resnet18": {"num_classes": 10},
    "resnet34": {"num_classes": 10},
    "vgg16_bn": {"num_classes": 10},
    "vit_small": {
        "num_classes": 10,
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 384,
        "depth": 6,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
}


# Preset experiment configs

def get_default_optimizer_config(name: str):
    """Return the default optimizer config for a given name."""
    configs = {
        "adamw": AdamWConfig,
        "muon": MuonConfig,
        "dion": DionConfig,
        "dion2": Dion2Config,
        "adadion": AdaDionConfig,
    }
    if name not in configs:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(configs.keys())}")
    return configs[name]()


def get_full_benchmark_configs():
    """Return all configs needed for the full benchmark run."""
    return {
        "base": BaseConfig(),
        "optimizers": {
            name: get_default_optimizer_config(name)
            for name in ["adamw", "muon", "dion", "dion2", "adadion"]
        },
        "models": ["resnet18", "vit_small"],
        "seeds": [42, 123, 456],
    }
