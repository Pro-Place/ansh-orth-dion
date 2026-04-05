"""
Unified optimizer factory for CIFAR-10 benchmarking.

Implements the hybrid param-grouping pattern from torchtitan:
  - 2D weight matrices -> spectral optimizer (Muon/Dion/Dion2/AdaDion)
  - 4D conv weights -> flattened to 2D via FlattenedParamWrapper for spectral opts
  - 1D params (norms, biases) -> AdamW with no weight decay
  - Embedding-like params -> AdamW with separate LR

For AdamW baseline, all params go through standard AdamW.
"""

import sys
import os
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer

# Spectral optimizers use @torch.compile internally. Increase cache limits
# for varied tensor shapes in vision models.
try:
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 64
except Exception:
    pass


class FlattenedParamWrapper:
    """Reshapes 4D conv params to 2D in-place for spectral optimizers.

    Spectral optimizers (Dion, AdaDion) require 2D tensors. This wrapper
    reshapes param.data from (C_out, C_in, kH, kW) to (C_out, C_in*kH*kW)
    before each optimizer step, and restores the original shape after.
    The model's forward pass always sees the original 4D shape.
    """

    def __init__(self):
        self._shapes = {}  # param -> original_shape

    def register(self, param: nn.Parameter):
        """Register a 4D+ param for flattening."""
        if param.ndim > 2:
            self._shapes[param] = param.shape

    def flatten_for_optimizer(self):
        """Reshape all registered params from 4D to 2D (in-place on .data and .grad)."""
        for param, orig_shape in self._shapes.items():
            param.data = param.data.flatten(1)
            if param.grad is not None:
                param.grad = param.grad.flatten(1)

    def restore_shapes(self):
        """Restore all registered params back to their original shapes."""
        for param, orig_shape in self._shapes.items():
            param.data = param.data.view(orig_shape)
            if param.grad is not None:
                param.grad = param.grad.view(orig_shape)


def group_params_for_hybrid(model: nn.Module):
    """
    Split model parameters into groups for hybrid spectral + AdamW optimization.
    All 2D+ weight tensors go to matrix_params (spectral optimizer).
    4D conv weights are included as-is — the optimizer or wrapper handles flattening.

    Returns:
        dict with keys: matrix_params, norm_params, embed_params
    """
    matrix_params = []
    norm_params = []
    embed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            norm_params.append(param)
        elif "cls_token" in name or "pos_embed" in name:
            embed_params.append(param)
        else:
            matrix_params.append(param)

    return {
        "matrix_params": matrix_params,
        "norm_params": norm_params,
        "embed_params": embed_params,
    }


def create_optimizer(model: nn.Module, opt_config):
    """
    Create an optimizer from config, applying hybrid param grouping
    for spectral optimizers.

    For Dion/AdaDion (which reject ndim>2), returns a FlattenedParamWrapper
    that must be used to flatten params before optimizer.step() and restore after.
    Returns (optimizer, wrapper_or_None).
    """
    name = opt_config.name
    wrapper = None

    # Dion, Dion2, and AdaDion require 2D params — create a wrapper that will
    # flatten/unflatten around each optimizer step
    if name in ("dion", "dion2", "adadion"):
        wrapper = FlattenedParamWrapper()
        for n, p in model.named_parameters():
            if p.requires_grad and p.ndim > 2 and "cls_token" not in n and "pos_embed" not in n:
                wrapper.register(p)
        # Temporarily flatten so optimizer init sees 2D shapes
        wrapper.flatten_for_optimizer()

    if name == "adamw":
        opt = _create_adamw(model, opt_config)
    elif name == "muon":
        opt = _create_muon(model, opt_config)
    elif name == "dion":
        opt = _create_dion(model, opt_config)
    elif name == "dion2":
        opt = _create_dion2(model, opt_config)
    elif name == "adadion":
        opt = _create_adadion(model, opt_config)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    # Restore shapes so model forward pass sees 4D
    if wrapper:
        wrapper.restore_shapes()

    return opt, wrapper


def _create_adamw(model: nn.Module, config) -> Optimizer:
    """Standard AdamW with weight decay only on 2D+ params."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )


def _build_scalar_groups(groups, config):
    """Build AdamW param groups for non-matrix parameters."""
    param_groups = []
    if groups["norm_params"]:
        param_groups.append({
            "params": groups["norm_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": 0.0,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })
    if groups["embed_params"]:
        param_groups.append({
            "params": groups["embed_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": config.scalar_weight_decay,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })
    return param_groups


def _create_muon(model: nn.Module, config) -> Optimizer:
    """Muon for all weight params + AdamW for scalars. Muon handles flatten internally."""
    from dion import Muon

    groups = group_params_for_hybrid(model)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})
    param_groups.extend(_build_scalar_groups(groups, config))

    adjust_lr = config.adjust_lr if config.adjust_lr != "none" else None
    return Muon(
        param_groups,
        lr=config.lr,
        mu=config.mu,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
        adjust_lr=adjust_lr,
        flatten=True,
        use_triton=False,
    )


def _create_dion(model: nn.Module, config) -> Optimizer:
    """Dion for all weight params + AdamW for scalars.

    Dion doesn't support flatten internally, so 4D conv params must be
    pre-flattened. We separate 2D and 4D params; 4D go through
    FlattenedParamWrapper (handled in create_optimizer).
    """
    from dion import Dion

    groups = group_params_for_hybrid(model)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})
    param_groups.extend(_build_scalar_groups(groups, config))

    return Dion(
        param_groups,
        lr=config.lr,
        rank_fraction=config.rank_fraction,
        weight_decay=config.weight_decay,
    )


def _create_dion2(model: nn.Module, config) -> Optimizer:
    """Dion2 for all weight params + AdamW for scalars.

    Dion2's dion2_pre_orthogonalize has a torch._inductor bug on this
    PyTorch version. Force eager mode to bypass it.
    """
    from dion import Dion2

    # Fully disable torch.compile for Dion2 — its compiled functions
    # hit InductorError on this PyTorch version
    torch._dynamo.disable(recursive=True)(lambda: None)()
    torch._dynamo.config.disable = True

    groups = group_params_for_hybrid(model)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})
    param_groups.extend(_build_scalar_groups(groups, config))

    adjust_lr = getattr(config, "adjust_lr", "spectral_norm")
    if adjust_lr == "none":
        adjust_lr = None

    return Dion2(
        param_groups,
        lr=config.lr,
        fraction=config.fraction,
        ef_decay=config.ef_decay,
        weight_decay=config.weight_decay,
        adjust_lr=adjust_lr,
        flatten=False,
        use_triton=False,
    )


def _create_adadion(model: nn.Module, config) -> Optimizer:
    """AdaDion V2 for all weight params + AdamW for scalars.

    AdaDion rejects ndim>2, so 4D conv params are pre-flattened to 2D
    via FlattenedParamWrapper (handled in create_optimizer).
    """
    from adadion_v2.adadion_v2 import AdaDionV2

    groups = group_params_for_hybrid(model)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})
    param_groups.extend(_build_scalar_groups(groups, config))

    return AdaDionV2(
        param_groups,
        lr=config.lr,
        mu=config.mu,
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
    )
