"""
Dion optimizer variants for single-GPU experiments.

Implements the full family from the unified framework:
  1. StrippedDion      – baseline with ColNorm, configurable beta
  2. OrthDion          – QR right factor (nu_t = 1 exactly)
  3. SoftIsometryDion  – diagonal S_t in [0,1] + QR right factor
  4. ModewiseBetaDion  – per-mode b_i feedback coefficients
  5. PADion            – persistence-aware adaptive beta (rho_t controller)
  6. RNormDion         – R_norm targeting controller
  7. PolyakDion        – explicit Polyak momentum, no error feedback

All variants share the same base class and power-iteration infrastructure.
The diagnostic hooks (residual decomposition, ReEntryNorm, etc.) live in
diagnostics.py and are called from the base class step().
"""

import math
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


# Helpers

def _col_norm(W: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize each column of W to unit length."""
    norms = W.norm(dim=0, keepdim=True).clamp(min=eps)
    return W / norms


def _orth(W: Tensor) -> Tensor:
    """QR orthogonalization – returns Q factor with orthonormal columns."""
    Q, _ = torch.linalg.qr(W, mode="reduced")
    return Q


def _col_norms(W: Tensor) -> Tensor:
    """Return per-column L2 norms (proxy for singular values)."""
    return W.norm(dim=0)


def _effective_rank(sigma: Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank from singular value estimates."""
    sigma = sigma.clamp(min=eps)
    p = sigma / sigma.sum()
    H = -(p * p.log()).sum()
    return H.exp().item()


# Base class

class DionBase(Optimizer):
    """
    Base Dion optimizer with power iteration, error feedback, and
    pluggable right-factor normalization.

    Subclasses override:
      - _normalize_right(W_t) -> V_bar_t   (ColNorm vs QR vs soft-isometry)
      - _compute_update(U_t, V_bar_t, ...) -> D_t
      - _error_feedback(M_t, U_t, P_captured, beta_t, ...) -> R_{t+1}
      - _adapt_beta(state, diagnostics) -> beta_t for next step
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        rank: int = 64,
        beta: float = 1.0,
        mu: float = 0.0,
        weight_decay: float = 0.0,
        power_iters: int = 1,
        warmup_steps: int = 0,
        collect_diagnostics: bool = False,
    ):
        defaults = dict(
            lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)
        self.collect_diagnostics = collect_diagnostics
        self._step_diagnostics: Dict[str, Any] = {}

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from the most recent step() call."""
        return self._step_diagnostics

    # Subclass hooks

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        """Produce V_bar from W = M^T @ U.  Override in subclasses."""
        return _col_norm(W)

    def _compute_update(
        self, U: Tensor, V_bar: Tensor, state: dict
    ) -> Tensor:
        """Produce the parameter update D = U @ V_bar^T (or with S_t)."""
        return U @ V_bar.t()

    def _error_feedback(
        self, M: Tensor, U: Tensor, beta: float, state: dict
    ) -> Tensor:
        """Compute R_{t+1} = M - beta * U @ U^T @ M."""
        P_captured = U @ (U.t() @ M)
        return M - beta * P_captured

    def _adapt_beta(
        self, state: dict, group: dict, diag: dict
    ) -> float:
        """Return beta for this step.  Override for adaptive variants."""
        return group["beta"]

    # Core step

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
            mu = group["mu"]
            wd = group["weight_decay"]
            power_iters = group["power_iters"]
            warmup = group["warmup_steps"]

            for pid, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                G = p.grad

                # 1D params (biases, norms): plain SGD
                if G.ndim == 1:
                    if wd > 0:
                        p.mul_(1 - lr * wd)
                    p.add_(G, alpha=-lr)
                    continue

                # 4D conv params: flatten to 2D (out_channels, in*kH*kW)
                orig_shape = None
                if G.ndim > 2:
                    orig_shape = G.shape
                    G = G.view(G.shape[0], -1)

                m, n = G.shape
                r = min(rank, m, n)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros(m, n, device=G.device, dtype=G.dtype)
                    V_init = torch.randn(n, r, device=G.device, dtype=G.dtype)
                    state["V"] = _orth(V_init)
                    state["R"] = torch.zeros(m, n, device=G.device, dtype=G.dtype)
                    state["prev_out_of_subspace_grad"] = None
                    state["rho_ema"] = 0.0
                    state["R_norm_ema"] = 0.0
                    state["beta_current"] = group["beta"]
                    state["orig_shape"] = orig_shape

                state["step"] += 1
                step_num = state["step"]
                M_buf = state["momentum"]
                V_prev = state["V"]
                R = state["R"]

                # --- Momentum update ---
                if mu > 0:
                    M_buf.mul_(mu).add_(G)
                else:
                    M_buf.copy_(G)

                # --- Buffer: M_t = momentum + error feedback ---
                M = M_buf + R

                # --- Power iteration (one or more steps) ---
                for _ in range(power_iters):
                    U = _orth(M @ V_prev)           # (m x r)
                    W = M.t() @ U                    # (n x r)
                    V_prev = _orth(W)                # warm-start for next iter

                # --- Right-factor normalization (subclass hook) ---
                W = M.t() @ U
                V_bar = self._normalize_right(W, state)

                # --- Diagnostics (before update) ---
                diag = {}
                if self.collect_diagnostics:
                    diag = self._collect_diagnostics(
                        G, M, U, V_bar, W, R, state
                    )

                # --- Adaptive beta ---
                beta = self._adapt_beta(state, group, diag)
                state["beta_current"] = beta

                # --- Compute update direction ---
                D = self._compute_update(U, V_bar, state)

                # --- Weight decay ---
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # --- Learning rate warmup ---
                effective_lr = lr
                if warmup > 0 and step_num <= warmup:
                    effective_lr = lr * step_num / warmup

                # --- Parameter update (apply to 2D view for conv params) ---
                p_flat = p.data.view(m, n) if orig_shape is not None else p.data
                p_flat.add_(D, alpha=-effective_lr)

                # --- Error feedback ---
                R_new = self._error_feedback(M, U, beta, state)
                state["R"] = R_new

                # --- Update V for warm-start ---
                state["V"] = V_prev

                # --- Store diagnostics ---
                if self.collect_diagnostics:
                    key = f"g{gid}_p{pid}"
                    diag["beta"] = beta
                    diag["step"] = step_num
                    diag["rank"] = r
                    all_diag[key] = diag

        self._step_diagnostics = all_diag
        return loss

    # Diagnostic collection

    def _collect_diagnostics(
        self,
        G: Tensor,        # gradient
        M: Tensor,        # buffer (momentum + R)
        U: Tensor,        # left factor (m x r)
        V_bar: Tensor,    # right factor after normalization (n x r)
        W: Tensor,        # M^T @ U before normalization (n x r)
        R: Tensor,        # current error feedback buffer
        state: dict,
    ) -> Dict[str, float]:
        """
        Compute the residual decomposition 
          R_{t+1} = T* + E + S
        where:
          T* = (I - P*) M     structural tail (unavoidable rank-r tail)
          E  = (P* - P) M     tracking defect (algorithmic error)
          S  = (1-beta) P M   persistence term (deliberate memory)

        Also compute:
          ReEntryNorm = |P_{t+1} R_t|_F / |G_t|_F
          R_norm = |R_t|_F / |G_t|_F
          rho_t = cosine similarity of consecutive out-of-subspace gradients
          nu_t = ||D_hat||_{(r)}
          epsilon_hat = ||(I - P) M||_F / ||M||_F
          q_t, u_t, s_t (innovation scale, re-entry, survival)
        """
        diag = {}
        m, n = M.shape
        r = U.shape[1]

        G_norm = G.norm().item()
        M_norm = M.norm().item()
        R_norm_val = R.norm().item()

        diag["G_norm"] = G_norm
        diag["M_norm"] = M_norm

        # --- R_norm = ||R||_F / ||G||_F ---
        diag["R_norm"] = R_norm_val / max(G_norm, 1e-12)

        # --- Tracked projector P = U @ U^T ---
        # P @ M = U @ (U^T @ M)
        UtM = U.t() @ M                      # (r x n)
        PM = U @ UtM                          # (m x n) = P @ M
        I_minus_P_M = M - PM                  # (I - P) @ M

        # --- Exact top-r projector P* via truncated SVD ---
        # Use full SVD only on M for diagnostics (expensive but exact)
        try:
            U_star, S_star, Vh_star = torch.linalg.svd(M, full_matrices=False)
        except Exception:
            # Fallback if SVD fails
            return diag
        U_star_r = U_star[:, :r]              # (m x r) exact top-r left vecs
        # P* @ M = U_r* @ U_r*^T @ M
        Ustar_tM = U_star_r.t() @ M           # (r x n)
        PstarM = U_star_r @ Ustar_tM          # (m x n) = P* @ M

        # --- Residual decomposition ---
        T_star = M - PstarM                   # structural tail (I - P*)M
        E = PstarM - PM                       # tracking defect (P* - P)M
        # S is computed at feedback time with beta, but we report norm here
        # S = (1 - beta) * PM, so its norm depends on current beta
        beta_cur = state.get("beta_current", 1.0)
        S = (1.0 - beta_cur) * PM             # persistence term

        diag["T_star_norm"] = T_star.norm().item()
        diag["E_norm"] = E.norm().item()
        diag["S_norm"] = S.norm().item()

        # --- Singular value diagnostics ---
        sigma = S_star[:min(2 * r, len(S_star))]
        diag["sigma_r"] = sigma[r - 1].item() if len(sigma) > r - 1 else 0.0
        diag["sigma_r1"] = sigma[r].item() if len(sigma) > r else 0.0
        diag["spectral_ratio"] = (
            diag["sigma_r1"] / max(diag["sigma_r"], 1e-12)
        )
        diag["effective_rank"] = _effective_rank(sigma[:r])

        # --- epsilon_hat = ||(I - P)M||_F / ||M||_F ---
        eps_hat = I_minus_P_M.norm().item() / max(M_norm, 1e-12)
        diag["epsilon_hat"] = eps_hat

        # --- nu_t = ||D_hat||_{(r)} ---
        D = self._compute_update(U, V_bar, state)
        sv_D = torch.linalg.svdvals(D)
        s1 = sv_D[0].item() if len(sv_D) > 0 else 0.0
        D_frob = D.norm().item()
        nu_t = max(s1, D_frob / max(math.sqrt(r), 1.0))
        diag["nu_t"] = nu_t

        # --- delta_t (oracle defect) = ||M||_{KF,r} - <M, D> ---
        KF_r = sigma[:r].sum().item()
        inner = (M * D).sum().item()
        diag["delta_t"] = KF_r - inner
        diag["KF_r_norm"] = KF_r

        # --- ReEntryNorm = ||P_{new} @ R||_F / ||G||_F ---
        # P_{new} is the projector we just computed (U @ U^T)
        # This measures how much of the OLD buffer re-enters the subspace
        PR = U @ (U.t() @ R)
        re_entry = PR.norm().item() / max(G_norm, 1e-12)
        diag["ReEntryNorm"] = re_entry

        # --- rho_t (gradient persistence) ---
        # cos angle between (I-P_t)G_t and (I-P_{t-1})G_{t-1}
        out_of_subspace_G = G - U @ (U.t() @ G)  # (I - P)G
        prev_oos = state.get("prev_out_of_subspace_grad", None)
        if prev_oos is not None:
            num = (out_of_subspace_G * prev_oos).sum().item()
            denom = (
                out_of_subspace_G.norm().item() *
                prev_oos.norm().item()
            )
            rho_t = num / max(denom, 1e-12)
        else:
            rho_t = 0.0
        state["prev_out_of_subspace_grad"] = out_of_subspace_G.clone()
        diag["rho_t"] = rho_t

        # --- EMA of rho ---
        alpha_rho = 0.95
        state["rho_ema"] = alpha_rho * state.get("rho_ema", 0.0) + (1 - alpha_rho) * rho_t
        diag["rho_ema"] = state["rho_ema"]

        # --- EMA of R_norm ---
        alpha_R = 0.95
        cur_Rnorm = diag["R_norm"]
        state["R_norm_ema"] = alpha_R * state.get("R_norm_ema", 0.0) + (1 - alpha_R) * cur_Rnorm
        diag["R_norm_ema"] = state["R_norm_ema"]

        # --- q_t, u_t, s_t (state variables) ---
        # q_t = ||(I - P)G||_F / ||G||_F  (innovation scale)
        diag["q_t"] = out_of_subspace_G.norm().item() / max(G_norm, 1e-12)

        # u_t = ||P_{t+1} R_t||_F / ||R_t||_F  (re-entry coefficient)
        diag["u_t"] = PR.norm().item() / max(R_norm_val, 1e-12)

        # s_t = ||(I - P_{t+1}) R_t||_F / ||R_t||_F  (survival coefficient)
        I_minus_P_R = R - PR
        diag["s_t"] = I_minus_P_R.norm().item() / max(R_norm_val, 1e-12)

        # --- Per-mode singular value diagnostics ---
        col_norms = _col_norms(W)
        diag["col_norms"] = col_norms.tolist()

        # --- Projector error epsilon_t = ||UU^T - U_r* U_r*^T||_op ---
        # Approximated as largest singular value of (UU^T - U_r* U_r*^T)
        # For efficiency, use: ||P - P*||_op = sin(theta_max)
        # where theta_max is the largest principal angle
        cross = U.t() @ U_star_r  # (r x r)
        svs_cross = torch.linalg.svdvals(cross)
        # sin(theta_max) = sqrt(1 - sigma_min(cross)^2)
        sigma_min_cross = svs_cross[-1].item() if len(svs_cross) > 0 else 0.0
        eps_t = math.sqrt(max(1.0 - sigma_min_cross ** 2, 0.0))
        diag["epsilon_t"] = eps_t

        # --- Shadowing error ||R - R*||_op (from MY_RESPONSE §14.1) ---
        # R* = (I - beta * P*) M = M - beta * P* M
        R_star = M - beta_cur * PstarM
        R_actual = self._error_feedback(M, U, beta_cur, state)
        shadow_err = (R_actual - R_star).norm().item()
        diag["shadowing_error"] = shadow_err
        # Theoretical bound: beta * epsilon_t * sigma_1(M)
        sigma1_M = S_star[0].item() if len(S_star) > 0 else 0.0
        diag["shadowing_bound"] = beta_cur * eps_t * sigma1_M

        return diag


# 1. Stripped Dion (ColNorm, configurable beta)

class StrippedDion(DionBase):
    """
    Standard Dion with ColNorm right factor and configurable beta.
    beta=1.0 is original Dion. beta<1 retains buffer for implicit gradient memory.
    """

    def __init__(self, params, lr=0.01, rank=64, beta=1.0, mu=0.0,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=False):
        super().__init__(
            params, lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        return _col_norm(W)


# 2. Orth-Dion (QR right factor, nu_t = 1)

class OrthDion(DionBase):
    """
    Orth-Dion: replace ColNorm with QR in right factor.
    Guarantees nu_t = 1 exactly. Convergence rate O(sqrt(L_r / T)).
    """

    def __init__(self, params, lr=0.01, rank=64, beta=1.0, mu=0.0,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=False):
        super().__init__(
            params, lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        return _orth(W)


# 3. Soft-Isometry AdaDion (diagonal S_t in [0,1], QR right factor)

class SoftIsometryDion(DionBase):
    """
          D_t = U_t @ S_t @ V_bar_t^T
      where S_t = diag(s_1, ..., s_r), s_i in [0, 1]
      V_bar = orth(M^T @ U)

    This keeps nu_t <= 1 (geometric benefit) while allowing per-mode
    scaling that can preserve epsilon_hat > 0 (memory benefit).

    S_t is computed from the column norms of W = M^T @ U:
      s_i = min(1, ||w_i|| / tau)
    where tau is a learnable or fixed threshold.

    Paired with modewise beta B_t for error feedback.
    """

    def __init__(self, params, lr=0.01, rank=64, beta=1.0, mu=0.0,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=False,
                 s_mode="normalized",
                 s_floor=0.1):
        """
        Args:
            s_mode: how to compute S_t diagonal entries.
                "normalized" – s_i = sigma_hat_i / sigma_hat_1 (relative to largest)
                "clipped"    – s_i = min(1, sigma_hat_i / tau) with adaptive tau
                "fixed"      – s_i = 1 for all i (reduces to Orth-Dion)
            s_floor: minimum value for s_i (prevents zero entries)
        """
        super().__init__(
            params, lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )
        self.s_mode = s_mode
        self.s_floor = s_floor

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        return _orth(W)

    def _compute_update(
        self, U: Tensor, V_bar: Tensor, state: dict
    ) -> Tensor:
        """D = U @ S @ V_bar^T where S = diag(s_1, ..., s_r)."""
        # Compute s_i from the column norms of W = M^T @ U
        # (stored by the base class in state during power iteration)
        # We recompute: W was passed to _normalize_right, but we need
        # the pre-normalization norms
        M = state["momentum"] + state["R"]
        W = M.t() @ U   # (n x r)
        col_norms = _col_norms(W)  # proxy for singular values

        if self.s_mode == "normalized":
            # s_i = sigma_i / sigma_1, clamped to [s_floor, 1]
            max_norm = col_norms.max().clamp(min=1e-12)
            s = (col_norms / max_norm).clamp(min=self.s_floor, max=1.0)
        elif self.s_mode == "clipped":
            # s_i = min(1, sigma_i / median_sigma)
            tau = col_norms.median().clamp(min=1e-12)
            s = (col_norms / tau).clamp(min=self.s_floor, max=1.0)
        else:  # "fixed"
            s = torch.ones(U.shape[1], device=U.device, dtype=U.dtype)

        state["_S_diag"] = s  # store for diagnostics
        # D = U @ diag(s) @ V_bar^T = (U * s[None, :]) @ V_bar^T
        return (U * s.unsqueeze(0)) @ V_bar.t()


# 4. Modewise-Beta Dion (per-mode b_i feedback)

class ModewiseBetaDion(DionBase):
    """
          R_{t+1} = M_t - U_t @ B_t @ U_t^T @ M_t
      where B_t = diag(b_1, ..., b_r), b_i in [0, 1]

    Per-mode beta allows different retention for different singular
    directions. Modes with high re-entry get high b_i (clear them);
    modes with low re-entry get low b_i (retain them).

    Can be combined with Soft-Isometry (QR right factor + S_t).
    """

    def __init__(self, params, lr=0.01, rank=64, beta=0.3, mu=0.0,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=False,
                 use_orth=True,
                 b_mode="uniform",
                 b_target_reentry=0.3):
        """
        Args:
            use_orth: if True, use QR for right factor; else ColNorm.
            b_mode: how to set per-mode beta.
                "uniform"   – all b_i = beta (reduces to scalar beta)
                "per_mode"  – b_i based on per-mode re-entry coefficient
                "per_mode_persistence" – b_i based on per-mode rho
            b_target_reentry: target re-entry fraction per mode.
        """
        super().__init__(
            params, lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )
        self.use_orth = use_orth
        self.b_mode = b_mode
        self.b_target_reentry = b_target_reentry

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        return _orth(W) if self.use_orth else _col_norm(W)

    def _error_feedback(
        self, M: Tensor, U: Tensor, beta: float, state: dict
    ) -> Tensor:
        """R_{t+1} = M - U @ B @ U^T @ M with per-mode B."""
        r = U.shape[1]
        UtM = U.t() @ M  # (r x n)

        if self.b_mode == "uniform":
            # Standard scalar beta
            return M - beta * (U @ UtM)

        elif self.b_mode == "per_mode":
            # Per-mode beta based on per-mode energy
            # b_i = beta * (1 + alpha * relative_energy_i)
            mode_energy = UtM.norm(dim=1)  # (r,) energy per mode
            total = mode_energy.sum().clamp(min=1e-12)
            relative = mode_energy / total  # fraction per mode

            # High-energy modes: clear more (higher b_i)
            # Low-energy modes: retain more (lower b_i)
            b = (beta * (0.5 + relative * r)).clamp(0.0, 1.0)  # (r,)
            state["_B_diag"] = b

            # R = M - U @ diag(b) @ U^T @ M
            scaled_UtM = b.unsqueeze(1) * UtM  # (r x n)
            return M - U @ scaled_UtM

        elif self.b_mode == "per_mode_persistence":
            # Per-mode beta based on per-mode persistence of
            # out-of-subspace gradient component
            # Requires tracking per-mode state across steps
            if "prev_mode_oos" not in state:
                state["prev_mode_oos"] = torch.zeros_like(UtM)
                state["mode_rho_ema"] = torch.zeros(r, device=M.device)

            # Current out-of-subspace component projected per mode
            G = state["momentum"] if "momentum" in state else M
            UtG = U.t() @ G  # (r x n) per-mode gradient projection

            # Per-mode persistence
            prev = state["prev_mode_oos"]
            if prev.shape == UtG.shape:
                cos_sim = torch.nn.functional.cosine_similarity(
                    UtG, prev, dim=1
                )  # (r,)
            else:
                cos_sim = torch.zeros(r, device=M.device)
            state["prev_mode_oos"] = UtG.clone()

            # EMA of per-mode rho
            alpha = 0.95
            state["mode_rho_ema"] = alpha * state["mode_rho_ema"] + (1 - alpha) * cos_sim

            # Sigmoid: high rho -> high beta (clear), low rho -> low beta (retain)
            s = 10.0  # sharpness
            b = torch.sigmoid(s * state["mode_rho_ema"]).clamp(0.05, 0.95)
            state["_B_diag"] = b

            scaled_UtM = b.unsqueeze(1) * UtM
            return M - U @ scaled_UtM

        return M - beta * (U @ UtM)


# 5. PA-Dion (Persistence-Aware adaptive beta)

class PADion(DionBase):
    """
    Persistence-aware adaptive beta controller:
      rho_ema = alpha * rho_ema + (1-alpha) * rho_t
      beta_t = beta_min + (beta_max - beta_min) * sigmoid(s * (rho_ema - tau))
    """

    def __init__(self, params, lr=0.01, rank=64, beta=0.5, mu=0.0,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=False,
                 use_orth=False,
                 beta_min=0.0, beta_max=1.0,
                 rho_alpha=0.95, sigmoid_sharpness=10.0, sigmoid_tau=0.0):
        super().__init__(
            params, lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )
        self.use_orth = use_orth
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.rho_alpha = rho_alpha
        self.sigmoid_sharpness = sigmoid_sharpness
        self.sigmoid_tau = sigmoid_tau

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        return _orth(W) if self.use_orth else _col_norm(W)

    def _adapt_beta(self, state: dict, group: dict, diag: dict) -> float:
        rho_ema = state.get("rho_ema", 0.0)
        s = self.sigmoid_sharpness
        tau = self.sigmoid_tau
        sig = 1.0 / (1.0 + math.exp(-s * (rho_ema - tau)))
        beta = self.beta_min + (self.beta_max - self.beta_min) * sig
        return beta


# 6. RNormDion (R_norm targeting controller)

class RNormDion(DionBase):
    """
    R_norm targeting controller:
      R_ema = alpha_R * R_ema + (1-alpha_R) * R_t
      beta_t = clip(0.5 + k_p * (R_ema - R_target), beta_min, beta_max)
    """

    def __init__(self, params, lr=0.01, rank=64, beta=0.5, mu=0.0,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=False,
                 use_orth=False,
                 R_target=1.8, k_p=0.3,
                 beta_min=0.1, beta_max=0.95,
                 R_alpha=0.95):
        super().__init__(
            params, lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )
        self.use_orth = use_orth
        self.R_target = R_target
        self.k_p = k_p
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.R_alpha = R_alpha

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        return _orth(W) if self.use_orth else _col_norm(W)

    def _adapt_beta(self, state: dict, group: dict, diag: dict) -> float:
        R_norm_ema = state.get("R_norm_ema", 0.0)
        beta = 0.5 + self.k_p * (R_norm_ema - self.R_target)
        return max(self.beta_min, min(self.beta_max, beta))


# 7. ReEntryDion (targets ReEntryNorm instead of R_norm)

class ReEntryDion(DionBase):
    """
    Targets ReEntryNorm = |P_{t+1} R_t|_F / |G_t|_F.

    Controller targets a specific ReEntryNorm value.
    """

    def __init__(self, params, lr=0.01, rank=64, beta=0.5, mu=0.0,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=False,
                 use_orth=False,
                 reentry_target=0.5, k_p=0.5,
                 beta_min=0.1, beta_max=0.95,
                 reentry_alpha=0.95):
        super().__init__(
            params, lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )
        self.use_orth = use_orth
        self.reentry_target = reentry_target
        self.k_p = k_p
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.reentry_alpha = reentry_alpha

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        return _orth(W) if self.use_orth else _col_norm(W)

    def _adapt_beta(self, state: dict, group: dict, diag: dict) -> float:
        # Compute ReEntryNorm from current state
        R = state.get("R", None)
        if R is None or R.norm().item() < 1e-12:
            return group["beta"]

        M = state["momentum"] + R
        r = state["V"].shape[1]
        U = _orth(M @ state["V"])
        PR = U @ (U.t() @ R)
        G_norm = state["momentum"].norm().item()  # approximate
        re_entry = PR.norm().item() / max(G_norm, 1e-12)

        # EMA
        if "reentry_ema" not in state:
            state["reentry_ema"] = re_entry
        state["reentry_ema"] = (
            self.reentry_alpha * state["reentry_ema"] +
            (1 - self.reentry_alpha) * re_entry
        )

        beta = 0.5 + self.k_p * (state["reentry_ema"] - self.reentry_target)
        return max(self.beta_min, min(self.beta_max, beta))


# 8. PolyakDion (explicit momentum, no error feedback)

class PolyakDion(DionBase):
    """
    Explicit Polyak momentum (mu=0.95), R = 0 by design.
    Explicit Polyak momentum baseline (no error feedback buffer).
    """

    def __init__(self, params, lr=0.01, rank=64, mu=0.95,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=False,
                 use_orth=False, beta=None, **kwargs):
        super().__init__(
            params, lr=lr, rank=rank, beta=1.0, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )
        self.use_orth = use_orth

    def _normalize_right(self, W: Tensor, state: dict) -> Tensor:
        return _orth(W) if self.use_orth else _col_norm(W)

    def _error_feedback(
        self, M: Tensor, U: Tensor, beta: float, state: dict
    ) -> Tensor:
        """No error feedback — R is always zero."""
        return torch.zeros_like(M)


# Factory

VARIANT_REGISTRY = {
    "stripped_dion": StrippedDion,
    "orth_dion": OrthDion,
    "soft_isometry": SoftIsometryDion,
    "modewise_beta": ModewiseBetaDion,
    "pa_dion": PADion,
    "rnorm_dion": RNormDion,
    "reentry_dion": ReEntryDion,
    "polyak_dion": PolyakDion,
}


def create_dion_variant(name: str, params, **kwargs) -> DionBase:
    """Create a Dion optimizer variant by name."""
    if name not in VARIANT_REGISTRY:
        raise ValueError(
            f"Unknown variant '{name}'. Choose from: {list(VARIANT_REGISTRY.keys())}"
        )
    return VARIANT_REGISTRY[name](params, **kwargs)
