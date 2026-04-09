"""
Main training script for Dion variant experiments on GPT-2 / WikiText-103.

Usage:
  # Single experiment
  python -m benchmark.lm.train --variant stripped_dion --beta 0.3

  # Run a pre-defined suite
  python -m benchmark.lm.train --suite laura_reproduction

  # Run the local response-surface experiment
  python -m benchmark.lm.train --mode response_surface --checkpoint path/to/ckpt.pt

  # Quick smoke test (100 steps)
  python -m benchmark.lm.train --variant stripped_dion --beta 1.0 --max_steps 100
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from benchmark.lm.model import SmallGPT2, create_small_gpt2
from benchmark.lm.data import load_wikitext103, create_dataloaders
from benchmark.lm.dion_variants import create_dion_variant, DionBase
from benchmark.lm.configs import (
    ExperimentConfig, ModelConfig, TrainConfig, OptimizerConfig,
    EXPERIMENT_SUITES, LAMBDA_SWEEP_LAMBDAS,
)


# Optimizer setup

def create_optimizer_for_experiment(
    model: SmallGPT2, opt_cfg: OptimizerConfig, train_cfg: TrainConfig
) -> tuple:
    """
    Create the hybrid optimizer setup:
      - Dion variant for 2D matrix params
      - AdamW for non-matrix params (embeddings, layernorms)

    Returns:
        (dion_optimizer, adamw_optimizer)
    """
    matrix_params = []
    scalar_params = []
    embedding_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name:
            matrix_params.append(p)
        elif "wte" in name or "wpe" in name or "lm_head" in name:
            embedding_params.append(p)
        else:
            scalar_params.append(p)

    print(f"Matrix params: {sum(p.numel() for p in matrix_params)/1e6:.1f}M "
          f"({len(matrix_params)} tensors)")
    print(f"Scalar params: {sum(p.numel() for p in scalar_params)/1e6:.1f}M "
          f"({len(scalar_params)} tensors)")
    print(f"Embedding params: {sum(p.numel() for p in embedding_params)/1e6:.1f}M "
          f"({len(embedding_params)} tensors)")

    # Dion variant for matrix params
    extra = opt_cfg.extra_kwargs.copy()
    dion_opt = create_dion_variant(
        opt_cfg.variant,
        matrix_params,
        lr=train_cfg.lr,
        rank=opt_cfg.rank,
        beta=opt_cfg.beta,
        mu=opt_cfg.mu,
        weight_decay=train_cfg.weight_decay,
        power_iters=opt_cfg.power_iters,
        warmup_steps=train_cfg.warmup_steps,
        collect_diagnostics=opt_cfg.collect_diagnostics,
        **extra,
    )

    # AdamW for scalar + embedding params
    adamw_groups = []
    if scalar_params:
        adamw_groups.append({
            "params": scalar_params,
            "lr": opt_cfg.scalar_lr,
            "weight_decay": opt_cfg.scalar_weight_decay,
        })
    if embedding_params:
        adamw_groups.append({
            "params": embedding_params,
            "lr": opt_cfg.scalar_lr,
            "weight_decay": opt_cfg.scalar_weight_decay,
        })

    adamw_opt = AdamW(
        adamw_groups,
        lr=opt_cfg.scalar_lr,
        betas=opt_cfg.scalar_betas,
        eps=opt_cfg.scalar_eps,
    ) if adamw_groups else None

    return dion_opt, adamw_opt


# Learning rate schedule

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    if cfg.lr_schedule == "constant":
        return cfg.lr
    # Cosine decay with min_lr = 0.1 * peak
    min_lr = cfg.lr * 0.1
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return min_lr + (cfg.lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizers: list, lr: float, scalar_lr: Optional[float] = None):
    """Set learning rate on all optimizers."""
    for opt in optimizers:
        if opt is None:
            continue
        for group in opt.param_groups:
            if scalar_lr is not None and group.get("_is_scalar", False):
                group["lr"] = scalar_lr
            else:
                group["lr"] = lr


# Evaluation

@torch.no_grad()
def evaluate(model: SmallGPT2, val_loader, device: str, max_batches: int = 50) -> dict:
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "val_loss": avg_loss,
        "val_perplexity": math.exp(min(avg_loss, 20)),
    }


# Main training loop

def train_experiment(cfg: ExperimentConfig) -> dict:
    """Run a single experiment and return results."""
    torch.manual_seed(cfg.train.seed)
    device = cfg.train.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        cfg.train.device = device

    # --- Setup ---
    out_dir = os.path.join(cfg.output_dir, cfg.name)
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "name": cfg.name,
            "model": cfg.model.__dict__,
            "train": cfg.train.__dict__,
            "optimizer": {
                k: v for k, v in cfg.optimizer.__dict__.items()
                if k != "extra_kwargs"
            } | {"extra_kwargs": cfg.optimizer.extra_kwargs},
        }, f, indent=2, default=str)

    # --- Model ---
    model = create_small_gpt2(**cfg.model.__dict__).to(device)
    if cfg.train.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    # --- Data ---
    train_ds, val_ds = load_wikitext103(seq_len=cfg.train.seq_len)
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )

    # --- Optimizers ---
    dion_opt, adamw_opt = create_optimizer_for_experiment(model, cfg.optimizer, cfg.train)

    # --- Training ---
    model.train()
    step = 0
    epoch = 0
    best_val_loss = float("inf")

    # Metrics storage
    step_metrics = []
    diag_metrics = []
    val_metrics = []

    train_iter = iter(train_loader)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Starting: {cfg.name}")
    print(f"Variant: {cfg.optimizer.variant}, rank={cfg.optimizer.rank}, "
          f"beta={cfg.optimizer.beta}, mu={cfg.optimizer.mu}")
    print(f"Steps: {cfg.train.max_steps}, batch={cfg.train.batch_size}, "
          f"lr={cfg.train.lr}, seq_len={cfg.train.seq_len}")
    print(f"{'='*60}\n")

    while step < cfg.train.max_steps:
        # Get batch (cycle through epochs)
        try:
            x, y = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # --- LR schedule ---
        lr = get_lr(step, cfg.train)
        for group in dion_opt.param_groups:
            group["lr"] = lr
        if adamw_opt:
            for group in adamw_opt.param_groups:
                group["lr"] = lr  # same schedule for simplicity

        # --- Forward + backward ---
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        # --- Gradient clipping ---
        grad_norm = 0.0
        if cfg.train.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.gradient_clip
            ).item()

        # --- Optimizer step ---
        # Enable full diagnostics at intervals
        if isinstance(dion_opt, DionBase):
            dion_opt.collect_diagnostics = (
                cfg.optimizer.collect_diagnostics and
                step % cfg.train.diag_interval == 0
            )

        dion_opt.step()
        if adamw_opt:
            adamw_opt.step()

        dion_opt.zero_grad(set_to_none=True)
        if adamw_opt:
            adamw_opt.zero_grad(set_to_none=True)

        # --- Log step metrics ---
        train_loss = loss.item()
        step_data = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": math.exp(min(train_loss, 20)),
            "lr": lr,
            "grad_norm": grad_norm,
            "time": time.time() - start_time,
        }
        step_metrics.append(step_data)

        # --- Log diagnostics ---
        if isinstance(dion_opt, DionBase) and dion_opt.collect_diagnostics:
            diag = dion_opt.get_diagnostics()
            if diag:
                # Aggregate across all params
                agg = _aggregate_diagnostics(diag, step)
                diag_metrics.append(agg)

        # --- Logging ---
        if step % cfg.train.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * cfg.train.batch_size * cfg.train.seq_len / elapsed
            beta_str = ""
            if isinstance(dion_opt, DionBase) and dion_opt.get_diagnostics():
                # Get beta from first param
                first_diag = next(iter(dion_opt.get_diagnostics().values()), {})
                if "beta" in first_diag:
                    beta_str = f" beta={first_diag['beta']:.3f}"
            print(
                f"step {step:6d} | loss {train_loss:.4f} | "
                f"ppl {math.exp(min(train_loss, 20)):8.2f} | "
                f"lr {lr:.5f} | gnorm {grad_norm:.3f} | "
                f"{tokens_per_sec:.0f} tok/s{beta_str}"
            )

        # --- Evaluation ---
        if step % cfg.train.eval_interval == 0 and step > 0:
            val_result = evaluate(
                model, val_loader, device, cfg.train.eval_steps
            )
            val_result["step"] = step
            val_metrics.append(val_result)
            print(
                f"  >> EVAL step {step}: val_loss={val_result['val_loss']:.4f} "
                f"val_ppl={val_result['val_perplexity']:.2f}"
            )
            if val_result["val_loss"] < best_val_loss:
                best_val_loss = val_result["val_loss"]

        # --- Checkpoint ---
        if step > 0 and step % cfg.train.save_interval == 0:
            ckpt_path = os.path.join(out_dir, f"ckpt_step{step}.pt")
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "dion_state": dion_opt.state_dict(),
                "adamw_state": adamw_opt.state_dict() if adamw_opt else None,
                "config": cfg,
            }, ckpt_path)
            print(f"  >> Saved checkpoint: {ckpt_path}")

        step += 1

    # --- Final evaluation ---
    final_val = evaluate(model, val_loader, device, cfg.train.eval_steps)
    final_val["step"] = step
    val_metrics.append(final_val)

    # --- Save results ---
    results = {
        "name": cfg.name,
        "variant": cfg.optimizer.variant,
        "rank": cfg.optimizer.rank,
        "beta": cfg.optimizer.beta,
        "mu": cfg.optimizer.mu,
        "final_val_loss": final_val["val_loss"],
        "final_val_ppl": final_val["val_perplexity"],
        "best_val_loss": best_val_loss,
        "total_steps": step,
        "total_time_s": time.time() - start_time,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, "step_metrics.json"), "w") as f:
        json.dump(step_metrics, f)

    with open(os.path.join(out_dir, "diag_metrics.json"), "w") as f:
        json.dump(diag_metrics, f)

    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f)

    print(f"\n{'='*60}")
    print(f"Finished: {cfg.name}")
    print(f"Final val loss: {final_val['val_loss']:.4f}, "
          f"ppl: {final_val['val_perplexity']:.2f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}\n")

    return results


# Local response-surface experiment (MY_RESPONSE §experiment 3)

def response_surface_experiment(
    checkpoint_path: str,
    lambdas: list = None,
    output_dir: str = "results/response_surface",
    device: str = "cuda",
):
    """
    Freeze a checkpoint and scale the buffer by lambda:
      M(lambda) = G + lambda * R
    Measure one-step loss, two-step loss, subspace change.
    Fit: Delta_f ≈ a*lambda - b*lambda^2 - c*lambda^3
    Then a/(2b) gives first approximation to the sweet spot.
    """
    if lambdas is None:
        lambdas = LAMBDA_SWEEP_LAMBDAS

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    # Rebuild model
    model = create_small_gpt2(**cfg.model.__dict__).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Load data
    train_ds, val_ds = load_wikitext103(seq_len=cfg.train.seq_len)
    train_loader, _ = create_dataloaders(train_ds, val_ds, batch_size=cfg.train.batch_size)

    # Get a batch
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    # Compute gradient at checkpoint
    model.train()
    logits = model(x)
    loss_0 = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss_0.backward()

    # Extract gradients and optimizer state
    dion_state = ckpt["dion_state"]

    results = []
    for lam in lambdas:
        # Clone model for each lambda
        model_copy = create_small_gpt2(**cfg.model.__dict__).to(device)
        model_copy.load_state_dict(ckpt["model_state"])

        # Apply one step with scaled buffer
        with torch.no_grad():
            for name, p in model_copy.named_parameters():
                if p.ndim == 2 and p.grad is not None:
                    G = p.grad
                    # Get R from optimizer state (approximate)
                    # In practice, reconstruct from the saved dion state
                    R = torch.zeros_like(G)  # placeholder
                    M = G + lam * R
                    # Simple rank-64 power iteration update
                    r = min(64, *M.shape)
                    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                    D = U[:, :r] @ Vh[:r, :]
                    p.add_(D, alpha=-cfg.train.lr)

        # Evaluate
        model_copy.eval()
        with torch.no_grad():
            logits_new = model_copy(x)
            loss_new = F.cross_entropy(
                logits_new.view(-1, logits_new.size(-1)), y.view(-1)
            ).item()

        delta_f = loss_new - loss_0.item()
        results.append({
            "lambda": lam,
            "loss_0": loss_0.item(),
            "loss_1step": loss_new,
            "delta_f": delta_f,
        })
        print(f"  lambda={lam:.2f}: loss={loss_new:.4f}, delta_f={delta_f:+.4f}")

    # Fit cubic: delta_f ≈ a*lam - b*lam^2 - c*lam^3
    import numpy as np
    lams = np.array([r["lambda"] for r in results])
    deltas = np.array([r["delta_f"] for r in results])
    # Fit: delta_f = c0 + c1*lam + c2*lam^2 + c3*lam^3
    coeffs = np.polyfit(lams, deltas, 3)
    c3, c2, c1, c0 = coeffs
    # Sweet spot ≈ -c1 / (2*c2) for the quadratic approximation
    if abs(c2) > 1e-12:
        sweet_spot = -c1 / (2 * c2)
    else:
        sweet_spot = float("nan")

    fit_result = {
        "coefficients": {"c0": c0, "c1": c1, "c2": c2, "c3": c3},
        "sweet_spot_estimate": sweet_spot,
    }
    results.append({"fit": fit_result})
    print(f"\nCubic fit: delta_f ≈ {c1:.4f}*λ + {c2:.4f}*λ² + {c3:.4f}*λ³")
    print(f"Estimated sweet spot: λ* ≈ {sweet_spot:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "response_surface.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# Diagnostic aggregation

def _aggregate_diagnostics(diag: Dict[str, Dict], step: int) -> dict:
    """Aggregate per-param diagnostics into summary statistics."""
    agg = {"step": step}
    keys_to_avg = [
        "R_norm", "epsilon_hat", "nu_t", "delta_t", "rho_t", "rho_ema",
        "ReEntryNorm", "q_t", "u_t", "s_t", "spectral_ratio",
        "effective_rank", "epsilon_t", "shadowing_error", "shadowing_bound",
        "T_star_norm", "E_norm", "S_norm", "beta", "R_norm_ema",
    ]
    for key in keys_to_avg:
        vals = [d[key] for d in diag.values() if key in d]
        if vals:
            agg[f"{key}_mean"] = sum(vals) / len(vals)
            agg[f"{key}_max"] = max(vals)
            agg[f"{key}_min"] = min(vals)
    return agg


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Dion variants LM experiment")

    parser.add_argument("--mode", default="train",
                        choices=["train", "suite", "response_surface"])

    # Single experiment
    parser.add_argument("--variant", default="stripped_dion",
                        help="Optimizer variant name")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--name", default=None,
                        help="Experiment name (auto-generated if not set)")

    # Extra kwargs as JSON string
    parser.add_argument("--extra", default="{}",
                        help="Extra kwargs for optimizer as JSON string")

    # Suite mode
    parser.add_argument("--suite", default=None,
                        help="Run a pre-defined experiment suite")

    # Response surface mode
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for response_surface mode")

    # Diagnostics
    parser.add_argument("--no_diagnostics", action="store_true")
    parser.add_argument("--diag_interval", type=int, default=500)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "suite" or args.suite:
        suite_name = args.suite or "laura_reproduction"
        if suite_name not in EXPERIMENT_SUITES:
            print(f"Unknown suite '{suite_name}'. Available: {list(EXPERIMENT_SUITES.keys())}")
            return

        suite = EXPERIMENT_SUITES[suite_name]
        print(f"\nRunning suite: {suite_name} ({len(suite)} experiments)\n")

        all_results = {}
        for exp_name, cfg in suite.items():
            cfg.output_dir = os.path.join(args.output_dir, suite_name)
            cfg.train.device = args.device
            cfg.train.max_steps = args.max_steps
            cfg.train.seed = args.seed
            try:
                result = train_experiment(cfg)
                all_results[exp_name] = result
            except Exception as e:
                print(f"ERROR in {exp_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[exp_name] = {"error": str(e)}

        # Save summary
        summary_path = os.path.join(args.output_dir, suite_name, "suite_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSuite summary saved to: {summary_path}")

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"{'Experiment':<35} {'Val Loss':>10} {'Val PPL':>10}")
        print(f"{'-'*70}")
        for name, r in sorted(all_results.items(), key=lambda x: x[1].get("final_val_loss", 999)):
            if "error" in r:
                print(f"{name:<35} {'ERROR':>10}")
            else:
                print(f"{name:<35} {r['final_val_loss']:>10.4f} {r['final_val_ppl']:>10.2f}")
        print(f"{'='*70}")

    elif args.mode == "response_surface":
        if not args.checkpoint:
            print("Error: --checkpoint required for response_surface mode")
            return
        response_surface_experiment(
            args.checkpoint,
            output_dir=os.path.join(args.output_dir, "response_surface"),
            device=args.device,
        )

    else:  # single experiment
        name = args.name or f"{args.variant}_r{args.rank}_beta{args.beta}_mu{args.mu}"
        extra = json.loads(args.extra)

        cfg = ExperimentConfig(
            name=name,
            model=ModelConfig(),
            train=TrainConfig(
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
                diag_interval=args.diag_interval,
            ),
            optimizer=OptimizerConfig(
                variant=args.variant,
                rank=args.rank,
                beta=args.beta,
                mu=args.mu,
                collect_diagnostics=not args.no_diagnostics,
                extra_kwargs=extra,
            ),
            output_dir=args.output_dir,
        )
        train_experiment(cfg)


if __name__ == "__main__":
    main()
