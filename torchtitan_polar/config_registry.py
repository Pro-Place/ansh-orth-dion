"""
Config registry for Polar-AdaDion experiments on LLaMA 320M.

Mirrors Tatsu's config_registry.py but uses PolarAdaDion with
interpolated polar right factor instead of ColNorm.

Usage:
    python -m torchtitan.train \
        --module ortho_matrix.polar_adadion \
        --config llama3_320m_polar \
        --training.steps 2000
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from ..common.base_container import BaseHybridOptimizersContainer
from ..common.model_configs import model_registry_320m
from ..common.training_configs import (
    base_metrics_config,
    base_training_config,
    debug_trainer_base,
)


class PolarAdaDionContainer(BaseHybridOptimizersContainer):
    """PolarAdaDion + AdamW scalar optimizer container."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseHybridOptimizersContainer.Config):
        name: str = "PolarAdaDion"
        # Polar interpolation
        tau: float = 1.0  # 0=ColNorm, 1=polar
        # Adaptive rank (inherited from AdaDion)
        adaptive_rank: bool = True
        rank_fraction_max: float = 1.0
        init_rank_fraction: float = 0.25
        erank_ema_beta: float = 0.5
        rank_scale: float = 2.0
        rank_min: int = 16
        rank_quantize: int = 8
        rank_step_up: int = 16
        rank_step_down: int = 8
        adapt_step: int = 1

    @staticmethod
    def _create_optimizer(
        config: "PolarAdaDionContainer.Config",
        param_groups: list[dict],
        mesh: DeviceMesh | None = None,
    ) -> Optimizer:
        from .polar_adadion import PolarAdaDion

        return PolarAdaDion(
            param_groups,
            tau=config.tau,
            outer_shard_mesh=mesh,
            lr=config.lr,
            rank_fraction_max=config.rank_fraction_max,
            weight_decay=config.weight_decay,
            adaptive_rank=config.adaptive_rank,
            init_rank_fraction=config.init_rank_fraction,
            erank_ema_beta=config.erank_ema_beta,
            rank_scale=config.rank_scale,
            rank_min=config.rank_min,
            rank_quantize=config.rank_quantize,
            rank_step_up=config.rank_step_up,
            rank_step_down=config.rank_step_down,
            adapt_step=config.adapt_step,
        )


def _320m_polar_base(tau: float = 1.0, **optimizer_kwargs) -> Trainer.Config:
    return Trainer.Config(
        model_spec=model_registry_320m(),
        optimizer=PolarAdaDionContainer.Config(tau=tau, **optimizer_kwargs),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=610,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=base_metrics_config(),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=True,
        ),
        validator=Validator.Config(
            freq=100,
            steps=20,
        ),
    )


# ColNorm baseline (tau=0, should match Tatsu's AdaDion exactly)
def llama3_320m_colnorm() -> Trainer.Config:
    return _320m_polar_base(
        tau=0.0,
        lr=0.012,
        adaptive_rank=True,
        rank_fraction_max=1.0,
        init_rank_fraction=0.5,
        erank_ema_beta=0.5,
        rank_scale=2.0,
        weight_decay=0.1,
        scalar_lr=0.012,
        scalar_beta1=0.95,
        scalar_beta2=0.95,
        scalar_eps=1e-8,
        scalar_weight_decay=0.1,
        output_head_lr_scaling=False,
    )


# Polar factor (tau=1)
def llama3_320m_polar() -> Trainer.Config:
    return _320m_polar_base(
        tau=1.0,
        lr=0.012,
        adaptive_rank=True,
        rank_fraction_max=1.0,
        init_rank_fraction=0.5,
        erank_ema_beta=0.5,
        rank_scale=2.0,
        weight_decay=0.1,
        scalar_lr=0.012,
        scalar_beta1=0.95,
        scalar_beta2=0.95,
        scalar_eps=1e-8,
        scalar_weight_decay=0.1,
        output_head_lr_scaling=False,
    )


# Interpolated polar sweep
def llama3_320m_interp_025() -> Trainer.Config:
    return _320m_polar_base(tau=0.25, lr=0.012, adaptive_rank=True,
        rank_fraction_max=1.0, init_rank_fraction=0.5, weight_decay=0.1,
        scalar_lr=0.012, scalar_beta1=0.95, scalar_beta2=0.95,
        scalar_weight_decay=0.1, output_head_lr_scaling=False)


def llama3_320m_interp_050() -> Trainer.Config:
    return _320m_polar_base(tau=0.5, lr=0.012, adaptive_rank=True,
        rank_fraction_max=1.0, init_rank_fraction=0.5, weight_decay=0.1,
        scalar_lr=0.012, scalar_beta1=0.95, scalar_beta2=0.95,
        scalar_weight_decay=0.1, output_head_lr_scaling=False)


def llama3_320m_interp_075() -> Trainer.Config:
    return _320m_polar_base(tau=0.75, lr=0.012, adaptive_rank=True,
        rank_fraction_max=1.0, init_rank_fraction=0.5, weight_decay=0.1,
        scalar_lr=0.012, scalar_beta1=0.95, scalar_beta2=0.95,
        scalar_weight_decay=0.1, output_head_lr_scaling=False)


# Debug configs
def llama3_debug_polar() -> Trainer.Config:
    config = debug_trainer_base()
    config.optimizer = PolarAdaDionContainer.Config(
        tau=1.0,
        lr=0.012,
        adaptive_rank=True,
        rank_fraction_max=1.0,
        init_rank_fraction=0.25,
        erank_ema_beta=0.5,
        rank_scale=2.0,
        rank_min=8,
        rank_quantize=8,
        weight_decay=0.1,
        scalar_lr=0.012,
        scalar_weight_decay=0.1,
    )
    config.training.steps = 50
    return config
