"""
Polar-Dion: right-factor normalization via the interpolated polar family.

Implements the theory from NEW_THEORY.txt:

  G_tau = (1-tau) * Diag(W^T W) + tau * (W^T W)
  V(tau) = W @ G_tau^{-1/2}

  tau=0: ColNorm (diagonal normalization)
  tau=1: polar factor (descent-optimal orthonormal basis)
  0<tau<1: partial whitening / soft orthogonalization

Also implements:
  - Gauge-fixed polar: W_tilde = W + mu * V_{t-1} before normalization
  - Compressed predictive memory: Z_t in R^{r x r} instead of dense R in R^{m x n}

The polar factor is the correct orthonormalization for maximizing
<M, U V^T> subject to V^T V = I. QR (Q-factor) does not solve this
problem and its failure in our experiments was testing the wrong primitive.
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


def _orth(W: Tensor) -> Tensor:
    Q, _ = torch.linalg.qr(W, mode="reduced")
    return Q


def _col_norm(W: Tensor, eps: float = 1e-8) -> Tensor:
    return W / W.norm(dim=0, keepdim=True).clamp(min=eps)


def _polar_factor(W: Tensor, eps: float = 1e-8) -> Tensor:
    """Polar right factor: V = W (W^T W)^{-1/2}.

    This is the descent-optimal orthonormal basis within span(W).
    It maximizes tr(W^T V) subject to V^T V = I.
    Symmetric, order-invariant, costs one r x r eigendecomposition.
    """
    r = W.shape[1]
    WtW = W.t() @ W  # (r, r)
    # Eigendecomposition for inverse square root
    eigvals, eigvecs = torch.linalg.eigh(WtW)
    eigvals = eigvals.clamp(min=eps)
    inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.t()  # (r, r)
    return W @ inv_sqrt


def _interpolated_polar(W: Tensor, tau: float = 1.0, eps: float = 1e-8) -> Tensor:
    """Interpolated polar family.

    G_tau = (1-tau) * Diag(W^T W) + tau * (W^T W)
    V(tau) = W @ G_tau^{-1/2}

    tau=0: ColNorm
    tau=1: polar factor
    """
    r = W.shape[1]
    WtW = W.t() @ W  # (r, r)

    if tau < 1e-6:
        return _col_norm(W, eps)
    elif tau > 1.0 - 1e-6:
        return _polar_factor(W, eps)
    else:
        diag_WtW = torch.diag(WtW.diag())  # (r, r)
        G_tau = (1 - tau) * diag_WtW + tau * WtW
        eigvals, eigvecs = torch.linalg.eigh(G_tau)
        eigvals = eigvals.clamp(min=eps)
        inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.t()
        return W @ inv_sqrt


def _gauge_fixed_polar(W: Tensor, V_prev: Tensor, mu: float = 0.1,
                       tau: float = 1.0, eps: float = 1e-8) -> Tensor:
    """Gauge-fixed interpolated polar.

    W_tilde = W + mu * V_{t-1}
    V_t = InterpolatedPolar(W_tilde, tau)

    The mu term enforces temporal continuity of the right basis,
    preventing jitter in the weak-gap regime.
    """
    W_tilde = W + mu * V_prev
    return _interpolated_polar(W_tilde, tau=tau, eps=eps)


class PolarDion(Optimizer):
    """Dion with interpolated polar right factor and optional gauge fixing.

    Implements the full family:
      right_factor="colnorm"    -> tau=0 (standard Dion)
      right_factor="polar"      -> tau=1 (descent-optimal)
      right_factor="interp"     -> tau in (0,1)
      right_factor="gauge_polar"-> polar + temporal gauge fixing
      right_factor="qr"         -> QR Q-factor (for comparison)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        rank: int = 64,
        beta: float = 0.3,
        weight_decay: float = 0.0,
        power_iters: int = 1,
        warmup_steps: int = 5,
        # Right factor control
        right_factor: str = "polar",  # colnorm, polar, interp, gauge_polar, qr
        tau: float = 1.0,             # interpolation parameter (0=colnorm, 1=polar)
        gauge_mu: float = 0.1,        # temporal gauge fixing strength
        # Compressed memory (optional)
        use_compressed_memory: bool = False,
        # Diagnostics
        collect_diagnostics: bool = False,
    ):
        defaults = dict(
            lr=lr, rank=rank, beta=beta, weight_decay=weight_decay,
            power_iters=power_iters, warmup_steps=warmup_steps,
            right_factor=right_factor, tau=tau, gauge_mu=gauge_mu,
            use_compressed_memory=use_compressed_memory,
        )
        super().__init__(params, defaults)
        self.collect_diagnostics = collect_diagnostics
        self._comm_volume = 0.0
        self._step_diagnostics = {}

    def get_comm_volume_gb(self) -> float:
        return self._comm_volume * 4 / 1e9

    def get_diagnostics(self) -> Dict:
        return self._step_diagnostics

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_diag = {}

        for gid, group in enumerate(self.param_groups):
            lr = group["lr"]
            rank = group["rank"]
            beta = group["beta"]
            wd = group["weight_decay"]
            warmup = group["warmup_steps"]
            rf = group["right_factor"]
            tau = group["tau"]
            gauge_mu = group["gauge_mu"]
            use_cm = group["use_compressed_memory"]

            for pid, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                G = p.grad

                # 1D params: plain SGD
                if G.ndim == 1:
                    if wd > 0:
                        p.mul_(1 - lr * wd)
                    p.add_(G, alpha=-lr)
                    continue

                # Flatten 4D conv to 2D
                orig_shape = None
                if G.ndim > 2:
                    orig_shape = G.shape
                    G = G.view(G.shape[0], -1)

                m, n = G.shape
                r = min(rank, m, n)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["R"] = torch.zeros(m, n, device=G.device, dtype=G.dtype)
                    state["V"] = _col_norm(torch.randn(n, r, device=G.device, dtype=G.dtype))
                    if use_cm:
                        state["Z"] = torch.zeros(r, r, device=G.device, dtype=G.dtype)

                state["step"] += 1
                step = state["step"]
                R = state["R"]
                V_prev = state["V"]

                effective_lr = lr
                if warmup > 0 and step <= warmup:
                    effective_lr = lr * step / warmup

                if wd > 0:
                    p.mul_(1 - effective_lr * wd)

                # Buffer
                if use_cm and "Z" in state:
                    # Compressed memory: M = G + U_{t-1} Z_{t-1} V_{t-1}^T
                    # Approximate: reconstruct low-rank memory from compressed state
                    U_prev = state.get("U_prev", None)
                    if U_prev is not None:
                        Z = state["Z"]
                        M = G + U_prev @ Z @ V_prev.t()
                    else:
                        M = G
                else:
                    M = G + R

                # Power iteration
                for _ in range(group["power_iters"]):
                    U = _orth(M @ V_prev)
                    W = M.t() @ U
                    V_prev = _col_norm(W)

                W = M.t() @ U

                # Right factor normalization
                if rf == "colnorm":
                    V_bar = _col_norm(W)
                elif rf == "qr":
                    V_bar = _orth(W)
                elif rf == "polar":
                    V_bar = _polar_factor(W)
                elif rf == "interp":
                    V_bar = _interpolated_polar(W, tau=tau)
                elif rf == "gauge_polar":
                    V_bar = _gauge_fixed_polar(W, state["V"], mu=gauge_mu, tau=tau)
                else:
                    V_bar = _col_norm(W)

                # Diagnostics
                if self.collect_diagnostics and step % 500 == 0:
                    sv = torch.linalg.svdvals(V_bar)
                    nu_t = sv[0].item()
                    D = U @ V_bar.t()
                    inner = (M * D).sum().item()
                    col_norms = W.norm(dim=0)
                    KF_r = col_norms.sum().item()  # approximate
                    delta_t = KF_r - inner
                    eps_hat = (M - U @ (U.t() @ M)).norm().item() / max(M.norm().item(), 1e-12)

                    from benchmark.lm.dion_variants import _effective_rank
                    diag = {
                        "nu_t": nu_t,
                        "delta_t": delta_t,
                        "epsilon_hat": eps_hat,
                        "effective_rank": _effective_rank(col_norms),
                        "R_norm": R.norm().item() / max(G.norm().item(), 1e-12),
                        "col_norm_ratio": col_norms.max().item() / max(col_norms.min().item(), 1e-12),
                        "basis_drift": 0.0,
                    }
                    # Basis drift: sin(angle) between V_t and V_{t-1}
                    cross = V_bar.t() @ state["V"]
                    sv_cross = torch.linalg.svdvals(cross)
                    if len(sv_cross) > 0:
                        diag["basis_drift"] = math.sqrt(max(1 - sv_cross[-1].item()**2, 0))

                    all_diag[f"g{gid}_p{pid}"] = diag

                # Parameter update
                p_flat = p.data.view(m, n) if orig_shape is not None else p.data
                p_flat.addmm_(U, V_bar.t(), alpha=-effective_lr)

                # Error feedback
                if use_cm:
                    # Compressed memory update: Z_{t+1} = (1-beta) * U^T M V + beta * Z_t decay
                    UtM = U.t() @ M  # (r, n)
                    Z_new = (1 - beta) * (UtM @ V_bar)  # (r, r)
                    if "Z" in state:
                        Z_new = Z_new + beta * 0.9 * state["Z"]  # decay old memory
                    state["Z"] = Z_new
                    state["U_prev"] = U.clone()
                    state["R"] = torch.zeros_like(state["R"])  # no dense buffer
                else:
                    captured = U @ (U.t() @ M)
                    state["R"] = M - beta * captured

                # Warm-start
                state["V"] = V_bar.clone()

                # Communication
                self._comm_volume += (m + n) * r

        self._step_diagnostics = all_diag
        return loss


# Convenience constructors
def create_polar_dion(params, **kwargs) -> PolarDion:
    return PolarDion(params, right_factor="polar", **kwargs)

def create_gauge_polar_dion(params, **kwargs) -> PolarDion:
    return PolarDion(params, right_factor="gauge_polar", **kwargs)

def create_interp_polar_dion(params, tau=0.5, **kwargs) -> PolarDion:
    return PolarDion(params, right_factor="interp", tau=tau, **kwargs)
