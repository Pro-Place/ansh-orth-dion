"""
Polar-AdaDion: AdaDion with polar right-factor normalization.

Extends Tatsu's AdaDion to replace column_normalize with the
interpolated polar family for the right factor:

  G_tau = (1-tau) * Diag(W^T W) + tau * (W^T W)
  V(tau) = W @ G_tau^{-1/2}

  tau=0: ColNorm (standard Dion/AdaDion behavior)
  tau=1: polar factor (descent-optimal orthonormal basis)

Also supports gauge-fixed polar (temporal continuity via mu * V_{t-1}).

Inherits all of AdaDion's distributed training support (DDP, FSDP, FSDP+TP)
and adaptive rank machinery.
"""

import math
import torch
import torch.nn.functional as F
import torch.distributed._functional_collectives as funcol
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from dion.dion import (
    _DionParamConfig,
    DionMixedPrecisionConfig,
    all_reduce_replicate_mesh,
    tensor_list_to_batch,
    fix_all_zero_or_nan,
    local_column_sum_sq,
    column_normalize,
    foreach_baddbmm_,
    orthogonalize,
)
from dion.opt_utils import (
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)

# Import AdaDion base
from .adadion import (
    AdaDion,
    _quantize_int,
    _compute_erank,
    _adapt_rank,
    _pad_Q_to_rcap,
)


def _polar_normalize(R_batch: Tensor, epsilon: float = 1e-8,
                     full_column_sum_sq: Optional[Tensor] = None) -> Tensor:
    """Polar right-factor normalization for batched R matrices.

    Computes V = R @ (R^T R)^{-1/2} for each matrix in the batch.
    This is the descent-optimal orthonormal basis within span(R).

    Args:
        R_batch: (batch, n, r) or (batch, m, r) right factor matrices
        epsilon: regularization for eigenvalue clamping
        full_column_sum_sq: if provided, used for distributed norm computation
    Returns:
        V_batch: (batch, n, r) polar-normalized right factors
    """
    batch, n, r = R_batch.shape
    # Compute R^T R for each batch element: (batch, r, r)
    RtR = R_batch.transpose(-2, -1) @ R_batch

    if full_column_sum_sq is not None:
        # In distributed setting, RtR diagonal needs the full column norms
        # Replace diagonal with the distributed sum
        diag_mask = torch.eye(r, device=R_batch.device, dtype=R_batch.dtype)
        diag_mask = diag_mask.unsqueeze(0).expand(batch, -1, -1)
        RtR = RtR * (1 - diag_mask) + full_column_sum_sq.unsqueeze(-1) * diag_mask

    # Eigendecomposition for inverse square root
    eigvals, eigvecs = torch.linalg.eigh(RtR)  # (batch, r), (batch, r, r)
    eigvals = eigvals.clamp(min=epsilon)
    inv_sqrt_eigvals = eigvals.rsqrt()  # (batch, r)

    # Reconstruct (R^T R)^{-1/2} = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs^T
    inv_sqrt = eigvecs * inv_sqrt_eigvals.unsqueeze(-2)  # broadcast multiply
    inv_sqrt = inv_sqrt @ eigvecs.transpose(-2, -1)  # (batch, r, r)

    return R_batch @ inv_sqrt


def _interpolated_polar_normalize(R_batch: Tensor, tau: float = 1.0,
                                   epsilon: float = 1e-8,
                                   full_column_sum_sq: Optional[Tensor] = None) -> Tensor:
    """Interpolated polar family for batched R matrices.

    G_tau = (1-tau) * Diag(R^T R) + tau * (R^T R)
    V(tau) = R @ G_tau^{-1/2}

    tau=0: column normalization
    tau=1: polar factor
    """
    if tau < 1e-6:
        return column_normalize(R_batch, full_column_sum_sq=full_column_sum_sq, epsilon=epsilon)
    if tau > 1.0 - 1e-6:
        return _polar_normalize(R_batch, epsilon=epsilon, full_column_sum_sq=full_column_sum_sq)

    batch, n, r = R_batch.shape
    RtR = R_batch.transpose(-2, -1) @ R_batch  # (batch, r, r)

    if full_column_sum_sq is not None:
        diag_mask = torch.eye(r, device=R_batch.device, dtype=R_batch.dtype)
        diag_mask = diag_mask.unsqueeze(0).expand(batch, -1, -1)
        RtR = RtR * (1 - diag_mask) + full_column_sum_sq.unsqueeze(-1) * diag_mask

    # G_tau = (1-tau) * Diag(RtR) + tau * RtR
    diag_RtR = torch.diagonal(RtR, dim1=-2, dim2=-1)  # (batch, r)
    diag_matrix = torch.diag_embed(diag_RtR)  # (batch, r, r)
    G_tau = (1 - tau) * diag_matrix + tau * RtR

    eigvals, eigvecs = torch.linalg.eigh(G_tau)
    eigvals = eigvals.clamp(min=epsilon)
    inv_sqrt_eigvals = eigvals.rsqrt()
    inv_sqrt = eigvecs * inv_sqrt_eigvals.unsqueeze(-2)
    inv_sqrt = inv_sqrt @ eigvecs.transpose(-2, -1)

    return R_batch @ inv_sqrt


def _dion_update_ddp_polar(
    X, G, M, Q, lr, mu, weight_decay, epsilon, param_config,
    replicate_mesh, replicate_mesh_grad_sync, oversample,
    r_list, ada_states, ada_config, outer_shard_mesh=None,
    tau=1.0,
) -> Generator[None, None, None]:
    """Adaptive Dion DDP update with polar right factor."""
    import torch.distributed as dist

    if isinstance(replicate_mesh, DeviceMesh):
        world_size = replicate_mesh.size()
        device_rank = replicate_mesh.get_rank()
    elif isinstance(replicate_mesh, ProcessGroup):
        world_size = dist.get_world_size(replicate_mesh)
        device_rank = dist.get_rank(replicate_mesh)
    else:
        world_size, device_rank = 1, 0

    num_real = len(ada_states)
    batch_size = len(X)
    transpose = param_config.is_transposed
    r_cap = ada_states[0]["r_cap"]
    orig_dtype = M[0].dtype

    if (replicate_mesh_grad_sync and not param_config.compressed_all_reduce
            and replicate_mesh is not None):
        G = all_reduce_replicate_mesh(G, replicate_mesh, return_dtensor=False)
        yield

    torch._foreach_add_(M, G)

    n = M[0].shape[0]
    m = M[0].shape[1]

    Q_padded, r_cap = _pad_Q_to_rcap(Q, ada_states, batch_size, transpose, n, m)
    M_batch, Q_batch = tensor_list_to_batch(M, Q_padded, transpose)
    P_batch = M_batch @ Q_batch

    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    if compressed_all_reduce and replicate_mesh is not None:
        P_single = funcol.reduce_scatter_tensor(
            P_batch, reduceOp="avg", scatter_dim=0, group=replicate_mesh)
        yield
    else:
        P_single = P_batch[device_rank:device_rank + 1]

    P_single = orthogonalize(P_single, oversample=oversample)

    if replicate_mesh is not None:
        P_batch = funcol.all_gather_tensor(P_single, gather_dim=0, group=replicate_mesh)
        yield
    else:
        P_batch = P_single

    R_batch = M_batch.mT @ P_batch

    if compressed_all_reduce and replicate_mesh is not None:
        R_batch = all_reduce_replicate_mesh(R_batch, replicate_mesh)
        yield

    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback
    foreach_baddbmm_(M, P_batch, R_batch, alpha=-(1 - mu), transpose=transpose)

    # POLAR right factor instead of column_normalize
    Q_batch_new = _interpolated_polar_normalize(R_batch, tau=tau, epsilon=epsilon)

    # Weight update
    fan_out, fan_in = X[0].size(0), X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr
    foreach_baddbmm_(
        X, P_batch, Q_batch_new,
        alpha=-scaled_lr, beta=1 - lr * weight_decay, transpose=transpose)

    # Update Q and adapt rank
    for i in range(num_real):
        state = ada_states[i]
        r_i = state["r"]
        q_dim = n if transpose else m
        Q[i][:q_dim, :r_i].copy_(Q_batch_new[i, :q_dim, :r_i])

        sig = torch.linalg.vector_norm(R_batch[i].float(), dim=0)
        erank = _compute_erank(sig, r_i)
        _adapt_rank(state, erank, ada_config, Q[i], orig_dtype)


def _dion_update_fsdp_polar(
    X, G, M, Q, lr, mu, weight_decay, epsilon, param_config,
    replicate_mesh, replicate_mesh_grad_sync, oversample,
    r_list, ada_states, ada_config, outer_shard_mesh=None,
    tau=1.0,
) -> Generator[None, None, None]:
    """Adaptive Dion FSDP update with polar right factor."""
    assert outer_shard_mesh is not None
    fsdp_group = outer_shard_mesh.get_group()

    num_real = len(ada_states)
    batch_size = len(X)
    transpose = param_config.is_transposed
    r_cap = ada_states[0]["r_cap"]
    orig_dtype = M[0].dtype

    G = [to_local(g) if isinstance(g, DTensor) else g for g in G]
    n_outer = M[0].shape[0]
    m = M[0].shape[1]

    torch._foreach_add_(M, G)

    Q_padded, r_cap = _pad_Q_to_rcap(Q, ada_states, batch_size, transpose, n_outer, m)
    M_batch, Q_batch = tensor_list_to_batch(M, Q_padded, transpose)
    P_batch = M_batch @ Q_batch

    P_single = funcol.reduce_scatter_tensor(
        P_batch, reduceOp="sum", scatter_dim=0, group=fsdp_group)
    yield

    P_single = orthogonalize(P_single, oversample=oversample)

    P_batch = funcol.all_gather_tensor(P_single, gather_dim=0, group=fsdp_group)
    yield

    R_batch = M_batch.mT @ P_batch
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    foreach_baddbmm_(M, P_batch, R_batch, alpha=-(1 - mu), transpose=transpose)

    # Distributed column norms for polar normalization
    R_sum_sq = local_column_sum_sq(R_batch)
    if param_config.outer_shard_mesh_dim is not None:
        R_sum_sq = funcol.all_reduce(R_sum_sq, reduceOp="sum", group=fsdp_group)
        yield

    # POLAR right factor
    Q_batch_new = _interpolated_polar_normalize(
        R_batch, tau=tau, epsilon=epsilon, full_column_sum_sq=R_sum_sq)

    fan_out, fan_in = X[0].size(0), X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr
    X_local = to_local(X)
    foreach_baddbmm_(
        X_local, P_batch, Q_batch_new,
        alpha=-scaled_lr, beta=1 - lr * weight_decay, transpose=transpose)

    sig_sq_batch = local_column_sum_sq(R_batch).squeeze(-2)
    if param_config.outer_shard_mesh_dim is not None:
        sig_sq_batch = funcol.all_reduce(sig_sq_batch, reduceOp="sum", group=fsdp_group)
    sig_batch = sig_sq_batch.sqrt()

    q_dim = n_outer if transpose else m
    for i in range(num_real):
        state = ada_states[i]
        r_i = state["r"]
        Q[i][:q_dim, :r_i].copy_(Q_batch_new[i, :q_dim, :r_i])
        erank = _compute_erank(sig_batch[i], r_i)
        _adapt_rank(state, erank, ada_config, Q[i], orig_dtype)


class PolarAdaDion(AdaDion):
    """AdaDion with interpolated polar right factor.

    tau=0: identical to AdaDion (ColNorm)
    tau=1: polar factor (descent-optimal orthonormal right factor)
    """

    def __init__(
        self,
        params,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._tau = tau

    def _create_dion_tasks(self, param_groups):
        if not self._adaptive_rank:
            yield from super()._create_dion_tasks(param_groups)
            return

        for group in param_groups:
            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            oversample = torch.tensor(group["oversample"])

            split_param_dict = self._split_params_by_sharding(group_params)
            for sharding_type, split_params in split_param_dict.items():
                if not split_params:
                    continue

                if sharding_type == "inner_sharded":
                    # Fall back to upstream (no polar TP support yet)
                    from dion.dion import dion_update_fsdp_tp
                    batch_size = self._inner_shard_mesh.size()
                    for params in create_param_batches(split_params, batch_size):
                        states = [self._get_or_initialize_state(p, group) for p in params]
                        param_config = self._get_dion_param_config(params[0])
                        yield AsyncTask(dion_update_fsdp_tp(
                            X=pad_batch(params, batch_size),
                            G=pad_batch([p.grad for p in params], batch_size),
                            M=pad_batch([s["momentum"] for s in states], batch_size),
                            Q=pad_batch([s["Q"] for s in states], batch_size),
                            lr=lr, mu=mu, weight_decay=weight_decay,
                            epsilon=epsilon, param_config=param_config,
                            replicate_mesh=self._replicate_mesh,
                            replicate_mesh_grad_sync=self._replicate_mesh_grad_sync,
                            oversample=oversample,
                        ))
                    continue

                if sharding_type == "outer_sharded":
                    update_func = _dion_update_fsdp_polar
                    batch_size = self._outer_shard_mesh.size()
                    use_dtensor = True
                elif sharding_type == "non_sharded":
                    update_func = _dion_update_ddp_polar
                    batch_size = self._replicate_world_size
                    use_dtensor = False
                else:
                    raise RuntimeError(f"Unknown sharding type: {sharding_type}")

                for params in create_param_batches(split_params, batch_size):
                    states = [self._get_or_initialize_state(p, group) for p in params]
                    momentums = [s["momentum"] for s in states]
                    Qs = [s["Q"] for s in states]
                    param_config = self._get_dion_param_config(params[0])
                    r_list_local = [s["r"] for s in states]

                    if not use_dtensor:
                        params = to_local(params)
                        gradients = to_local([p.grad for p in group_params[:len(params)]])
                        momentums = to_local(momentums)
                        Qs = to_local(Qs)
                    else:
                        gradients = [p.grad for p in params]

                    yield AsyncTask(update_func(
                        X=pad_batch(params, batch_size),
                        G=pad_batch(gradients, batch_size),
                        M=pad_batch(momentums, batch_size),
                        Q=pad_batch(Qs, batch_size),
                        lr=lr, mu=mu, weight_decay=weight_decay,
                        epsilon=epsilon, param_config=param_config,
                        replicate_mesh=self._replicate_mesh,
                        replicate_mesh_grad_sync=self._replicate_mesh_grad_sync,
                        oversample=oversample,
                        r_list=r_list_local,
                        ada_states=states,
                        ada_config=self._ada_config,
                        outer_shard_mesh=self._outer_shard_mesh,
                        tau=self._tau,
                    ))
