"""
AdaDion V2 single-GPU implementation.

Reimplements the core algorithm of Tatsu's AdaDion V2 without distributed
dependencies (DeviceMesh, DTensor, FSDP). For fair comparison only.

Core V2 algorithm:
  - Polyak momentum: M_t = mu * M_{t-1} + G_t
  - Power iteration + ColNorm right factor
  - Error feedback with beta = 1.0 (standard Dion)
  - Adaptive rank via effective rank estimation
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from typing import Tuple, Dict


def _orth(W: Tensor) -> Tensor:
    Q, _ = torch.linalg.qr(W, mode="reduced")
    return Q


def _col_norm(W: Tensor, eps: float = 1e-8) -> Tensor:
    return W / W.norm(dim=0, keepdim=True).clamp(min=eps)


def _qr_norm(W: Tensor) -> Tensor:
    Q, _ = torch.linalg.qr(W, mode="reduced")
    return Q


def _effective_rank(sigma: Tensor, eps: float = 1e-12) -> float:
    sigma = sigma.clamp(min=eps)
    p = sigma / sigma.sum()
    H = -(p * p.log()).sum()
    return H.exp().item()


class AdaDionV2Single(Optimizer):
    """
    Single-GPU AdaDion V2 (faithful to Tatsu's distributed implementation).

    Key features:
      - Polyak momentum (mu=0.95)
      - ColNorm right factor
      - Error feedback with beta=1 (via momentum subtraction)
      - Adaptive rank from effective rank estimation
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        rank: int = 64,
        mu: float = 0.95,
        weight_decay: float = 0.0,
        power_iters: int = 1,
        warmup_steps: int = 5,
        use_qr: bool = False,
        # Adaptive rank
        adaptive_rank: bool = True,
        erank_ema_beta: float = 0.9,
        rank_scale: float = 1.5,
        rank_min: int = 8,
        rank_quantize: int = 8,
        rank_fraction_max: float = 0.7,
        init_rank_fraction: float = 0.25,
        # Scalar params
        scalar_lr: float = 1e-3,
        scalar_betas: Tuple[float, float] = (0.9, 0.95),
        scalar_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, rank=rank, mu=mu, weight_decay=weight_decay,
            power_iters=power_iters, warmup_steps=warmup_steps,
            adaptive_rank=adaptive_rank,
            erank_ema_beta=erank_ema_beta, rank_scale=rank_scale,
            rank_min=rank_min, rank_quantize=rank_quantize,
            rank_fraction_max=rank_fraction_max,
            init_rank_fraction=init_rank_fraction,
            scalar_lr=scalar_lr, scalar_betas=scalar_betas, scalar_eps=scalar_eps,
        )
        super().__init__(params, defaults)
        self._comm_volume = 0.0
        self._use_qr = use_qr

    def get_comm_volume_gb(self) -> float:
        return self._comm_volume * 4 / 1e9

    def get_rank_info(self) -> Dict[str, int]:
        info = {}
        for gid, group in enumerate(self.param_groups):
            for pid, p in enumerate(group["params"]):
                state = self.state.get(p, {})
                if "rank" in state:
                    info[f"g{gid}_p{pid}"] = state["rank"]
        return info

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            wd = group["weight_decay"]
            warmup = group["warmup_steps"]
            rank_default = group["rank"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad

                # 1D params: AdamW
                if G.ndim != 2:
                    self._adamw_step(p, G, group)
                    continue

                # 2D params: Dion V2 update
                m, n = G.shape
                state = self.state[p]

                if len(state) == 0:
                    r_cap = max(group["rank_min"],
                                min(int(group["rank_fraction_max"] * min(m, n)), min(m, n)))
                    if group["adaptive_rank"]:
                        r = max(group["rank_min"],
                                min(int(group["init_rank_fraction"] * min(m, n)), r_cap))
                        q = group["rank_quantize"]
                        r = ((r + q - 1) // q) * q
                    else:
                        r = min(rank_default, m, n)

                    state["step"] = 0
                    state["rank"] = r
                    state["r_cap"] = r_cap
                    state["momentum"] = torch.zeros_like(G)
                    state["Q"] = _col_norm(torch.randn(n, r, device=G.device, dtype=G.dtype))
                    state["erank_ema"] = float(r)

                state["step"] += 1
                step = state["step"]
                r = state["rank"]

                # LR warmup
                effective_lr = lr
                if warmup > 0 and step <= warmup:
                    effective_lr = lr * step / warmup

                # Weight decay
                if wd > 0:
                    p.mul_(1 - effective_lr * wd)

                # === V2 Core: Polyak momentum ===
                M = state["momentum"]
                M.mul_(mu).add_(G)  # M = mu * M + G

                # === Power iteration ===
                Q = state["Q"]
                for _ in range(group["power_iters"]):
                    P = _orth(M @ Q[:, :r])       # (m, r)
                    R_mat = M.t() @ P              # (n, r)
                    Q_new = _qr_norm(R_mat) if self._use_qr else _col_norm(R_mat)

                # === Error feedback (V2 style): subtract from momentum ===
                # M -= (1 - mu) * P @ R^T  (this is V2's error feedback)
                M.add_(P @ R_mat.t(), alpha=-(1 - mu))

                # === Parameter update ===
                # Scale by lr / r for normalization
                scaled_lr = effective_lr * math.sqrt(min(m, n) / r)
                p.addmm_(P, Q_new[:, :r].t(), alpha=-scaled_lr)

                # === Update Q for warm-start ===
                state["Q"][:, :r] = Q_new[:, :r]

                # === Adaptive rank ===
                if group["adaptive_rank"]:
                    col_norms = R_mat.norm(dim=0)
                    erank = _effective_rank(col_norms)
                    alpha = group["erank_ema_beta"]
                    state["erank_ema"] = alpha * state["erank_ema"] + (1 - alpha) * erank

                    desired = int(math.ceil(group["rank_scale"] * state["erank_ema"]))
                    q = group["rank_quantize"]
                    desired = max(group["rank_min"], ((desired + q - 1) // q) * q)
                    desired = min(desired, state["r_cap"])

                    if desired != r:
                        old_Q = state["Q"]
                        if desired > r:
                            extra = _col_norm(torch.randn(
                                n, desired - r, device=G.device, dtype=G.dtype
                            ))
                            state["Q"] = torch.cat([old_Q[:, :r], extra], dim=1)
                        else:
                            state["Q"] = old_Q[:, :desired]
                        state["rank"] = desired

                # Communication accounting
                self._comm_volume += (m + n) * r

        return loss

    def _adamw_step(self, p, G, group):
        state = self.state[p]
        slr = group["scalar_lr"]
        b1, b2 = group["scalar_betas"]
        eps = group["scalar_eps"]
        wd = group["weight_decay"]

        if len(state) == 0:
            state["step"] = 0
            state["m"] = torch.zeros_like(G)
            state["v"] = torch.zeros_like(G)

        state["step"] += 1
        step = state["step"]
        m, v = state["m"], state["v"]

        if wd > 0:
            p.mul_(1 - slr * wd)

        m.mul_(b1).add_(G, alpha=1 - b1)
        v.mul_(b2).addcmul_(G, G, value=1 - b2)
        bc1 = 1 - b1 ** step
        bc2 = 1 - b2 ** step
        step_size = slr / bc1
        denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
        p.addcdiv_(m, denom, value=-step_size)
