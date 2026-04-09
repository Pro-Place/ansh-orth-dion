"""
Investigation: Why does QR right-factor kill training?

Hypotheses to test:
  H1. Spectral gap is too small at rank 64 → QR/power-iteration can't distinguish
      signal from noise in the tail directions
  H2. QR Gram-Schmidt propagates errors from dominant to weak columns → effective
      rank collapses
  H3. ColNorm's per-column independence preserves weak-but-real directions that
      QR destroys
  H4. The issue is specific to high rank; QR works at low rank where the gap is larger
  H5. More power iterations can rescue QR by improving the left factor U

Experiments:
  A. Rank sweep (r=4,8,16,32,64) × {ColNorm, QR} — find crossover rank
  B. Power iteration sweep (1,2,3,5 iters) for QR at r=64 — can more iters help?
  C. Column-norm spectrum analysis — what does W = M^T @ U look like pre-normalization?
  D. Block-diagonal QR — QR within blocks of k, ColNorm across blocks
  E. Exact SVD baseline — isolate power-iteration vs normalization
  F. ColNorm→QR schedule — start ColNorm, switch to QR at step N
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.lm.model import SmallGPT2, create_small_gpt2
from benchmark.lm.data import load_wikitext103, create_dataloaders
from benchmark.lm.dion_variants import (
    DionBase, _col_norm, _orth, _col_norms, _effective_rank
)
from benchmark.lm.train import create_optimizer_for_experiment, get_lr, evaluate
from benchmark.lm.configs import ModelConfig, TrainConfig, OptimizerConfig, ExperimentConfig


# New normalization methods for investigation

def _block_qr(W: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """QR within blocks, independent across blocks.

    Splits r columns into blocks of size k.
    Within each block: full QR (orthonormal).
    Across blocks: no orthogonalization.

    nu_t <= sqrt(block_size) instead of sqrt(r).
    Preserves inter-block column independence like ColNorm.
    """
    n, r = W.shape
    blocks = []
    for i in range(0, r, block_size):
        block = W[:, i:i+block_size]
        Q, _ = torch.linalg.qr(block, mode="reduced")
        blocks.append(Q)
    return torch.cat(blocks, dim=1)


def _partial_orth(W: torch.Tensor, n_corrections: int = 1) -> torch.Tensor:
    """ColNorm + partial Gram-Schmidt corrections.

    Start with ColNorm (independent columns).
    Then do n_corrections rounds of pairwise orthogonalization
    to reduce off-diagonal correlations without fully destroying
    the column magnitude structure.

    n_corrections=0: pure ColNorm
    n_corrections→∞: approaches QR
    """
    eps = 1e-8
    V = W / W.norm(dim=0, keepdim=True).clamp(min=eps)  # ColNorm

    for _ in range(n_corrections):
        # One round of modified Gram-Schmidt (but don't renormalize)
        _, r = V.shape
        for j in range(1, r):
            for i in range(j):
                proj = (V[:, j] @ V[:, i]) * V[:, i]
                V[:, j] = V[:, j] - 0.5 * proj  # 0.5 = partial correction
        # Re-normalize columns to unit length
        V = V / V.norm(dim=0, keepdim=True).clamp(min=eps)

    return V


def _svd_right_factor(M: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact truncated SVD — the gold standard.
    Returns (U_r, V_r) with orthonormal columns.
    Cost: O(mn min(m,n)) — expensive but exact.
    """
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return U[:, :r], Vh[:r, :].t()


# Modified Dion with pluggable normalization

class InvestigationDion(DionBase):
    """Dion with configurable right-factor normalization for investigation."""

    def __init__(self, params, lr=0.01, rank=64, beta=0.3, mu=0.0,
                 weight_decay=0.0, power_iters=1, warmup_steps=0,
                 collect_diagnostics=True,
                 right_factor="colnorm",  # colnorm, qr, block_qr, partial_orth, exact_svd
                 block_size=8,            # for block_qr
                 n_corrections=1,         # for partial_orth
                 switch_step=None,        # for colnorm→qr schedule
                 switch_to="qr"):
        super().__init__(
            params, lr=lr, rank=rank, beta=beta, mu=mu,
            weight_decay=weight_decay, power_iters=power_iters,
            warmup_steps=warmup_steps,
            collect_diagnostics=collect_diagnostics,
        )
        self.right_factor = right_factor
        self.block_size = block_size
        self.n_corrections = n_corrections
        self.switch_step = switch_step
        self.switch_to = switch_to

    def _normalize_right(self, W, state):
        step = state.get("step", 0)

        # Handle schedule: switch normalization at a specific step
        method = self.right_factor
        if self.switch_step is not None and step >= self.switch_step:
            method = self.switch_to

        if method == "colnorm":
            return _col_norm(W)
        elif method == "qr":
            return _orth(W)
        elif method == "block_qr":
            return _block_qr(W, self.block_size)
        elif method == "partial_orth":
            return _partial_orth(W, self.n_corrections)
        elif method == "exact_svd":
            # For exact SVD, we bypass the normal flow.
            # Return QR of W (the right factor will be from SVD in _compute_update)
            return _orth(W)
        else:
            raise ValueError(f"Unknown right_factor: {method}")

    def _compute_update(self, U, V_bar, state):
        if self.right_factor == "exact_svd":
            # Use exact SVD instead of power iteration
            M = state["momentum"] + state["R"]
            r = min(U.shape[1], *M.shape)
            U_exact, V_exact = _svd_right_factor(M, r)
            state["_exact_U"] = U_exact  # for diagnostics
            return U_exact @ V_exact.t()
        return U @ V_bar.t()

    def _collect_diagnostics(self, G, M, U, V_bar, W, R, state):
        """Extended diagnostics: also log column norms of W (pre-normalization)."""
        diag = super()._collect_diagnostics(G, M, U, V_bar, W, R, state)

        # Column norm spectrum of W = M^T @ U (BEFORE normalization)
        col_norms = _col_norms(W)
        sorted_norms = col_norms.sort(descending=True).values

        if len(sorted_norms) > 0:
            diag["w_col_norm_max"] = sorted_norms[0].item()
            diag["w_col_norm_min"] = sorted_norms[-1].item()
            diag["w_col_norm_ratio"] = (
                sorted_norms[0].item() / max(sorted_norms[-1].item(), 1e-12)
            )
            # How many columns have norm > 10% of max?
            threshold = 0.1 * sorted_norms[0].item()
            diag["w_active_columns"] = (sorted_norms > threshold).sum().item()
            # Effective rank of W column norms
            diag["w_effective_rank"] = _effective_rank(sorted_norms)

            # Log the full spectrum (first 10 and last 5)
            n = len(sorted_norms)
            top10 = sorted_norms[:min(10, n)].tolist()
            bot5 = sorted_norms[max(0, n-5):].tolist()
            diag["w_spectrum_top10"] = top10
            diag["w_spectrum_bot5"] = bot5

        # Gram matrix deviation from identity (for V_bar)
        gram = V_bar.t() @ V_bar
        r = gram.shape[0]
        gram_dev = (gram - torch.eye(r, device=gram.device)).norm().item()
        diag["gram_deviation"] = gram_dev

        # nu_t exact (largest singular value of V_bar)
        sv = torch.linalg.svdvals(V_bar)
        diag["nu_t_exact"] = sv[0].item()
        diag["nu_t_min"] = sv[-1].item() if len(sv) > 0 else 0.0

        return diag


# Experiment runner

def run_investigation(
    name: str,
    right_factor: str = "colnorm",
    rank: int = 64,
    beta: float = 0.3,
    power_iters: int = 1,
    max_steps: int = 10000,
    block_size: int = 8,
    n_corrections: int = 1,
    switch_step: int = None,
    switch_to: str = "qr",
    device: str = "cuda",
    output_dir: str = "results/lm/qr_investigation",
    diag_interval: int = 200,
    seed: int = 42,
) -> dict:
    """Run a single investigation experiment."""

    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    # Model
    model = create_small_gpt2().to(device)

    # Data
    train_ds, val_ds = load_wikitext103(seq_len=1024)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=8)

    # Separate params
    matrix_params = []
    scalar_params = []
    embedding_params = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "wte" not in pname and "wpe" not in pname and "lm_head" not in pname:
            matrix_params.append(p)
        elif "wte" in pname or "wpe" in pname or "lm_head" in pname:
            embedding_params.append(p)
        else:
            scalar_params.append(p)

    # Investigation optimizer
    dion_opt = InvestigationDion(
        matrix_params, lr=0.01, rank=rank, beta=beta,
        power_iters=power_iters, warmup_steps=3000,
        collect_diagnostics=True,
        right_factor=right_factor,
        block_size=block_size,
        n_corrections=n_corrections,
        switch_step=switch_step,
        switch_to=switch_to,
    )

    # AdamW for scalars
    adamw_groups = []
    if scalar_params:
        adamw_groups.append({"params": scalar_params, "lr": 3e-4, "weight_decay": 0.1})
    if embedding_params:
        adamw_groups.append({"params": embedding_params, "lr": 3e-4, "weight_decay": 0.1})
    adamw_opt = AdamW(adamw_groups, lr=3e-4) if adamw_groups else None

    # Training
    model.train()
    step = 0
    epoch = 0
    train_iter = iter(train_loader)
    start_time = time.time()

    step_metrics = []
    diag_metrics = []
    val_metrics_list = []

    cfg = TrainConfig(max_steps=max_steps, warmup_steps=3000)

    print(f"\n{'='*60}")
    print(f"Investigation: {name}")
    print(f"right_factor={right_factor}, rank={rank}, beta={beta}, "
          f"power_iters={power_iters}")
    if switch_step:
        print(f"Schedule: {right_factor}→{switch_to} at step {switch_step}")
    print(f"{'='*60}\n")

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule
        lr = get_lr(step, cfg)
        for group in dion_opt.param_groups:
            group["lr"] = lr
        if adamw_opt:
            for group in adamw_opt.param_groups:
                group["lr"] = 3e-4  # fixed for scalar params

        # Forward + backward
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        # Clip
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

        # Diagnostics toggle
        dion_opt.collect_diagnostics = (step % diag_interval == 0)

        # Step
        dion_opt.step()
        if adamw_opt:
            adamw_opt.step()
        dion_opt.zero_grad(set_to_none=True)
        if adamw_opt:
            adamw_opt.zero_grad(set_to_none=True)

        train_loss = loss.item()
        step_metrics.append({
            "step": step, "train_loss": train_loss, "lr": lr, "grad_norm": grad_norm,
        })

        # Diagnostics
        if dion_opt.collect_diagnostics:
            diag = dion_opt.get_diagnostics()
            if diag:
                agg = {"step": step}
                # Aggregate
                for key in ["epsilon_hat", "nu_t", "nu_t_exact", "nu_t_min",
                            "R_norm", "delta_t", "effective_rank",
                            "gram_deviation", "w_col_norm_max", "w_col_norm_min",
                            "w_col_norm_ratio", "w_active_columns", "w_effective_rank",
                            "rho_t", "ReEntryNorm", "q_t", "u_t", "s_t",
                            "T_star_norm", "E_norm", "S_norm",
                            "spectral_ratio", "epsilon_t",
                            "shadowing_error", "beta"]:
                    vals = [d[key] for d in diag.values() if key in d and isinstance(d[key], (int, float))]
                    if vals:
                        agg[f"{key}_mean"] = sum(vals) / len(vals)
                        agg[f"{key}_max"] = max(vals)
                        agg[f"{key}_min"] = min(vals)

                # Column norm spectrum from first param
                first = next(iter(diag.values()))
                if "w_spectrum_top10" in first:
                    agg["w_spectrum_top10"] = first["w_spectrum_top10"]
                    agg["w_spectrum_bot5"] = first["w_spectrum_bot5"]

                diag_metrics.append(agg)

        # Logging
        if step % 500 == 0:
            elapsed = time.time() - start_time
            extra = ""
            if dion_opt.collect_diagnostics and dion_opt.get_diagnostics():
                d0 = next(iter(dion_opt.get_diagnostics().values()))
                ef = d0.get("effective_rank", 0)
                nu = d0.get("nu_t_exact", d0.get("nu_t", 0))
                dt = d0.get("delta_t", 0)
                extra = f" erank={ef:.1f} nu={nu:.3f} delta={dt:.4f}"
            print(f"step {step:6d} | loss {train_loss:.4f} | lr {lr:.5f} | "
                  f"gnorm {grad_norm:.3f}{extra}")

        # Eval
        if step > 0 and step % 1000 == 0:
            val_result = evaluate(model, val_loader, device, 50)
            val_result["step"] = step
            val_metrics_list.append(val_result)
            print(f"  >> EVAL: val_loss={val_result['val_loss']:.4f}")

        step += 1

    # Final eval
    final_val = evaluate(model, val_loader, device, 50)
    final_val["step"] = step
    val_metrics_list.append(final_val)

    # Save
    results = {
        "name": name, "right_factor": right_factor, "rank": rank, "beta": beta,
        "power_iters": power_iters, "block_size": block_size,
        "n_corrections": n_corrections, "switch_step": switch_step,
        "final_val_loss": final_val["val_loss"],
        "total_time_s": time.time() - start_time,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "diag_metrics.json"), "w") as f:
        json.dump(diag_metrics, f)
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics_list, f)
    with open(os.path.join(out_dir, "step_metrics.json"), "w") as f:
        json.dump(step_metrics, f)

    print(f"\nFinished {name}: val_loss={final_val['val_loss']:.4f}")
    return results


# Full investigation suite

INVESTIGATIONS = {
    # --- A. Rank sweep: find the crossover ---
    "colnorm_r4":    dict(right_factor="colnorm",  rank=4,   beta=0.3),
    "colnorm_r8":    dict(right_factor="colnorm",  rank=8,   beta=0.3),
    "colnorm_r16":   dict(right_factor="colnorm",  rank=16,  beta=0.3),
    "colnorm_r32":   dict(right_factor="colnorm",  rank=32,  beta=0.3),
    "colnorm_r64":   dict(right_factor="colnorm",  rank=64,  beta=0.3),
    "qr_r4":         dict(right_factor="qr",       rank=4,   beta=0.3),
    "qr_r8":         dict(right_factor="qr",       rank=8,   beta=0.3),
    "qr_r16":        dict(right_factor="qr",       rank=16,  beta=0.3),
    "qr_r32":        dict(right_factor="qr",       rank=32,  beta=0.3),
    "qr_r64":        dict(right_factor="qr",       rank=64,  beta=0.3),

    # --- B. Power iteration sweep for QR ---
    "qr_r64_pow1":   dict(right_factor="qr", rank=64, beta=0.3, power_iters=1),
    "qr_r64_pow2":   dict(right_factor="qr", rank=64, beta=0.3, power_iters=2),
    "qr_r64_pow3":   dict(right_factor="qr", rank=64, beta=0.3, power_iters=3),
    "qr_r64_pow5":   dict(right_factor="qr", rank=64, beta=0.3, power_iters=5),

    # --- C. Block-diagonal QR (novel middle ground) ---
    "block_qr_k4":   dict(right_factor="block_qr", rank=64, beta=0.3, block_size=4),
    "block_qr_k8":   dict(right_factor="block_qr", rank=64, beta=0.3, block_size=8),
    "block_qr_k16":  dict(right_factor="block_qr", rank=64, beta=0.3, block_size=16),
    "block_qr_k32":  dict(right_factor="block_qr", rank=64, beta=0.3, block_size=32),

    # --- D. Partial orthogonalization (smooth interpolation) ---
    "partial_orth_0": dict(right_factor="partial_orth", rank=64, beta=0.3, n_corrections=0),  # = ColNorm
    "partial_orth_1": dict(right_factor="partial_orth", rank=64, beta=0.3, n_corrections=1),
    "partial_orth_2": dict(right_factor="partial_orth", rank=64, beta=0.3, n_corrections=2),
    "partial_orth_3": dict(right_factor="partial_orth", rank=64, beta=0.3, n_corrections=3),

    # --- E. Exact SVD (gold standard, expensive) ---
    "exact_svd_r64":  dict(right_factor="exact_svd", rank=64, beta=0.3),

    # --- F. ColNorm → QR schedule ---
    "switch_at_5k":   dict(right_factor="colnorm", rank=64, beta=0.3, switch_step=5000, switch_to="qr"),
    "switch_at_3k":   dict(right_factor="colnorm", rank=64, beta=0.3, switch_step=3000, switch_to="qr"),

    # --- G. QR with beta=1 vs ColNorm with beta=1 (control) ---
    "colnorm_r64_b1": dict(right_factor="colnorm", rank=64, beta=1.0),
    "qr_r64_b1":      dict(right_factor="qr",      rank=64, beta=1.0),
}

# Curated subsets for faster runs
INVESTIGATION_SUITES = {
    # Quick: rank crossover + block QR (12 experiments)
    "quick": {k: v for k, v in INVESTIGATIONS.items()
              if k.startswith(("colnorm_r", "qr_r")) and k.endswith(("r8", "r16", "r32", "r64"))
              or k.startswith("block_qr")},

    # Core: the most important experiments (16 experiments)
    "core": {k: v for k, v in INVESTIGATIONS.items()
             if k in [
                 "colnorm_r8", "colnorm_r16", "colnorm_r32", "colnorm_r64",
                 "qr_r8", "qr_r16", "qr_r32", "qr_r64",
                 "block_qr_k4", "block_qr_k8", "block_qr_k16",
                 "partial_orth_1", "partial_orth_2",
                 "exact_svd_r64",
                 "switch_at_5k",
                 "qr_r64_pow3",
             ]},

    # Full: everything
    "full": INVESTIGATIONS,
}


def main():
    parser = argparse.ArgumentParser(description="QR failure investigation")
    parser.add_argument("--suite", default="core", choices=list(INVESTIGATION_SUITES.keys()))
    parser.add_argument("--experiment", default=None, help="Run single experiment by name")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/lm/qr_investigation")
    parser.add_argument("--diag_interval", type=int, default=200)
    args = parser.parse_args()

    if args.experiment:
        if args.experiment not in INVESTIGATIONS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(INVESTIGATIONS.keys())}")
            return
        kwargs = INVESTIGATIONS[args.experiment]
        run_investigation(
            args.experiment, max_steps=args.max_steps,
            device=args.device, output_dir=args.output_dir,
            diag_interval=args.diag_interval, **kwargs
        )
    else:
        suite = INVESTIGATION_SUITES[args.suite]
        print(f"\nRunning investigation suite: {args.suite} ({len(suite)} experiments)\n")

        all_results = {}
        for exp_name, kwargs in suite.items():
            try:
                result = run_investigation(
                    exp_name, max_steps=args.max_steps,
                    device=args.device, output_dir=args.output_dir,
                    diag_interval=args.diag_interval, **kwargs
                )
                all_results[exp_name] = result
            except Exception as e:
                print(f"ERROR in {exp_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[exp_name] = {"error": str(e)}

        # Summary
        summary_path = os.path.join(args.output_dir, "investigation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"{'Experiment':<30} {'Method':<15} {'Rank':>5} {'Val Loss':>10}")
        print(f"{'-'*70}")
        for name, r in sorted(all_results.items(),
                               key=lambda x: x[1].get("final_val_loss", 999)):
            if "error" in r:
                print(f"{name:<30} {'':15} {'':>5} {'ERROR':>10}")
            else:
                print(f"{name:<30} {r.get('right_factor','?'):<15} "
                      f"{r.get('rank','?'):>5} {r['final_val_loss']:>10.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
