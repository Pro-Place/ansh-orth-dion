#!/usr/bin/env python3
"""
CIFAR-10 Optimizer Benchmark Suite.

Comprehensive benchmarking of Dion1, Dion2, AdaDion V2, Muon, and AdamW
on CIFAR-10 with multiple model architectures, seeds, and metrics.

Usage:
    # Full benchmark (all optimizers, all models, 3 seeds)
    python cifar10_benchmark.py --mode full

    # Single optimizer run
    python cifar10_benchmark.py --optimizer muon --model resnet18 --seed 42

    # LR sweep for a specific optimizer
    python cifar10_benchmark.py --mode sweep --optimizer muon

    # Quick smoke test
    python cifar10_benchmark.py --mode smoke
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# GradScaler moved in PyTorch 2.4+
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from configs import (
    AdaDionConfig,
    AdamWConfig,
    BaseConfig,
    Dion2Config,
    DionConfig,
    MuonConfig,
    SWEEP_GRIDS,
    get_default_optimizer_config,
    get_full_benchmark_configs,
    MODEL_CONFIGS,
)
from metrics import MetricsCollector, compute_accuracy, compute_gradient_norm
from models import count_parameters, create_model, get_param_summary
from optimizers import create_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Data loading

def get_cifar10_loaders(batch_size: int, num_workers: int, seed: int):
    """Create CIFAR-10 train and test data loaders with standard augmentation."""
    # Standard CIFAR-10 augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
        transforms.RandomErasing(p=0.1),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# LR Scheduler

def create_lr_scheduler(optimizer, base_config: BaseConfig, steps_per_epoch: int):
    """Create a cosine LR scheduler with linear warmup."""
    total_steps = base_config.epochs * steps_per_epoch
    warmup_steps = base_config.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if base_config.lr_schedule == "cosine":
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        elif base_config.lr_schedule == "linear":
            factor = 1.0 - progress
        else:
            factor = 1.0
        return max(base_config.min_lr_factor, factor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Training and evaluation

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             use_amp: bool = True):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    total_correct_1 = 0
    total_correct_5 = 0
    total_samples = 0

    amp_enabled = use_amp and device.type == "cuda"

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            output = model(images)
            loss = F.cross_entropy(output, targets)

        total_loss += loss.item() * targets.size(0)
        top1, top5 = compute_accuracy(output, targets, topk=(1, 5))
        total_correct_1 += top1 * targets.size(0) / 100
        total_correct_5 += top5 * targets.size(0) / 100
        total_samples += targets.size(0)

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct_1 / total_samples * 100,
        "top5_acc": total_correct_5 / total_samples * 100,
    }


def train_one_run(
    base_config: BaseConfig,
    opt_config,
    output_dir: str,
    run_name: str,
) -> dict:
    """Execute a single training run and return summary metrics."""

    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Starting run: {run_name} on {device} ===")

    # Seed everything
    seed = base_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if base_config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        base_config.batch_size, base_config.num_workers, seed,
    )
    steps_per_epoch = len(train_loader)

    # Model
    model_kwargs = MODEL_CONFIGS.get(base_config.model_name, {"num_classes": 10})
    model = create_model(base_config.model_name, **model_kwargs).to(device)
    param_summary = get_param_summary(model)
    logger.info(
        f"Model: {base_config.model_name}, "
        f"Params: {param_summary['total']:,} "
        f"(matrix: {param_summary['matrix_pct']:.1f}%)"
    )

    # Optimizer
    optimizer, flatten_wrapper = create_optimizer(model, opt_config)
    logger.info(f"Optimizer: {opt_config.name}")

    # LR Scheduler
    scheduler = create_lr_scheduler(optimizer, base_config, steps_per_epoch)

    # Mixed precision: disable entirely for spectral optimizers.
    # Their @torch.compile internals conflict with fp16 gradients
    # and the GradScaler scale/unscale/inf-check pattern.
    is_spectral = opt_config.name in ("muon", "dion", "dion2", "adadion")
    use_amp = base_config.mixed_precision and torch.cuda.is_available() and (not is_spectral)
    try:
        scaler = GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)

    # Metrics
    metrics = MetricsCollector(output_dir, run_name)

    # Save run config
    run_config = {
        "base_config": asdict(base_config) if hasattr(base_config, '__dataclass_fields__') else vars(base_config),
        "opt_config": asdict(opt_config) if hasattr(opt_config, '__dataclass_fields__') else vars(opt_config),
        "param_summary": param_summary,
        "device": str(device),
        "use_amp": use_amp,
    }
    os.makedirs(os.path.join(output_dir, run_name), exist_ok=True)
    with open(os.path.join(output_dir, run_name, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Loss function with label smoothing
    train_criterion = nn.CrossEntropyLoss(label_smoothing=base_config.label_smoothing)

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(base_config.epochs):
        model.train()
        metrics.start_epoch()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            metrics.start_step()

            # Forward + backward
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(images)
                loss = train_criterion(output, targets)

            scaler.scale(loss).backward()

            # Unscale gradients for correct norm computation and clipping
            scaler.unscale_(optimizer)

            # Gradient clipping and norm
            if base_config.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), base_config.gradient_clip
                ).item()
            else:
                grad_norm = compute_gradient_norm(model)

            # Flatten conv params to 2D for Dion/AdaDion optimizer step
            if flatten_wrapper:
                flatten_wrapper.flatten_for_optimizer()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Restore conv params to 4D for model forward pass
            if flatten_wrapper:
                flatten_wrapper.restore_shapes()

            # Track accuracy
            with torch.no_grad():
                pred = output.argmax(dim=1)
                correct = pred.eq(targets).sum().item()

            batch_acc = correct / targets.size(0) * 100
            epoch_loss += loss.item() * targets.size(0)
            epoch_correct += correct
            epoch_samples += targets.size(0)

            # Get current LR
            current_lr = scheduler.get_last_lr()[0]

            step_metrics = metrics.end_step(
                step=global_step,
                epoch=epoch,
                loss=loss.item(),
                acc=batch_acc,
                lr=current_lr,
                grad_norm=grad_norm,
                batch_size=targets.size(0),
            )

            # Log step metrics
            if global_step % base_config.log_interval == 0:
                logger.info(
                    f"[{run_name}] Epoch {epoch}/{base_config.epochs} "
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {batch_acc:.1f}% | "
                    f"LR: {current_lr:.6f} | "
                    f"GradNorm: {grad_norm:.4f} | "
                    f"Time: {step_metrics['step_time_ms']:.1f}ms | "
                    f"Mem: {step_metrics['gpu_mem_mb']:.0f}MB"
                )

            # Log optimizer diagnostics (for AdaDion)
            if hasattr(optimizer, "get_rank") and global_step % (base_config.log_interval * 5) == 0:
                opt_diag = {}
                if hasattr(optimizer, "get_rank"):
                    opt_diag.update(optimizer.get_rank())
                if hasattr(optimizer, "get_effective_rank"):
                    opt_diag.update(optimizer.get_effective_rank())
                if opt_diag:
                    metrics.log_optimizer_metrics(global_step, opt_diag)

            global_step += 1

        # End of epoch: evaluate
        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples * 100

        val_metrics = evaluate(model, test_loader, device, use_amp=use_amp)

        epoch_metrics = metrics.end_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["acc"],
            val_top5_acc=val_metrics["top5_acc"],
            lr=current_lr,
        )

        logger.info(
            f"[{run_name}] Epoch {epoch}/{base_config.epochs} DONE | "
            f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
            f"Val: {val_metrics['loss']:.4f}/{val_metrics['acc']:.1f}% "
            f"(top5: {val_metrics['top5_acc']:.1f}%) | "
            f"Best: {metrics.best_val_acc:.1f}% (ep {metrics.best_val_epoch})"
        )

        # Save checkpoint
        if base_config.save_checkpoints and (epoch + 1) % base_config.checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, run_name, f"checkpoint_ep{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": metrics.best_val_acc,
            }, ckpt_path)

        # Save best model
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_path = os.path.join(output_dir, run_name, "best_model.pt")
            torch.save(model.state_dict(), best_path)

    # Save all metrics
    metrics.save()
    summary = metrics.get_summary()
    logger.info(f"[{run_name}] FINISHED | Best val acc: {summary['best_val_acc']:.2f}% "
                f"(epoch {summary['best_val_epoch']}) | Time: {summary['total_train_time_sec']:.0f}s")

    return summary


# Sweep mode

def run_lr_sweep(
    base_config: BaseConfig,
    optimizer_name: str,
    output_dir: str,
    seeds: list[int],
):
    """Run an LR sweep for a single optimizer."""
    grid = SWEEP_GRIDS.get(optimizer_name, {})
    if not grid:
        logger.warning(f"No sweep grid defined for {optimizer_name}")
        return []

    # Only sweep LR by default for efficiency
    lr_values = grid.get("lr", [])
    if not lr_values:
        logger.warning(f"No LR values in sweep grid for {optimizer_name}")
        return []

    results = []
    for lr in lr_values:
        for seed in seeds:
            opt_config = get_default_optimizer_config(optimizer_name)
            opt_config.lr = lr
            if hasattr(opt_config, "scalar_lr"):
                # Scale scalar_lr proportionally for spectral optimizers
                default_config = get_default_optimizer_config(optimizer_name)
                ratio = lr / default_config.lr
                opt_config.scalar_lr = default_config.scalar_lr * ratio

            bc = copy.deepcopy(base_config)
            bc.seed = seed

            run_name = f"sweep_{optimizer_name}_lr{lr}_seed{seed}"
            summary = train_one_run(bc, opt_config, output_dir, run_name)
            summary["lr"] = lr
            summary["seed"] = seed
            summary["optimizer"] = optimizer_name
            results.append(summary)

    return results


# Full benchmark mode

def run_full_benchmark(output_dir: str):
    """Run the complete benchmark suite."""
    configs = get_full_benchmark_configs()
    base = configs["base"]
    optimizers = configs["optimizers"]
    models = configs["models"]
    seeds = configs["seeds"]

    all_results = []

    for model_name in models:
        for opt_name, opt_config in optimizers.items():
            for seed in seeds:
                bc = copy.deepcopy(base)
                bc.model_name = model_name
                bc.seed = seed

                run_name = f"{model_name}_{opt_name}_seed{seed}"

                try:
                    summary = train_one_run(bc, opt_config, output_dir, run_name)
                    summary["model"] = model_name
                    summary["optimizer"] = opt_name
                    summary["seed"] = seed
                    all_results.append(summary)
                except Exception as e:
                    logger.error(f"Run {run_name} FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "run_name": run_name,
                        "model": model_name,
                        "optimizer": opt_name,
                        "seed": seed,
                        "error": str(e),
                    })

                # Save intermediate results
                results_path = os.path.join(output_dir, "all_results.json")
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    return all_results


# Smoke test mode

def run_smoke_test(output_dir: str):
    """Quick smoke test: 3 epochs, all optimizers, resnet18 only."""
    base = BaseConfig(
        epochs=3,
        warmup_epochs=1,
        batch_size=128,
        log_interval=10,
        save_checkpoints=False,
        model_name="resnet18",
    )

    all_results = []
    for opt_name in ["adamw", "muon", "dion", "dion2", "adadion"]:
        opt_config = get_default_optimizer_config(opt_name)
        run_name = f"smoke_{opt_name}"
        try:
            summary = train_one_run(base, opt_config, output_dir, run_name)
            summary["optimizer"] = opt_name
            all_results.append(summary)
            logger.info(f"Smoke test {opt_name}: OK")
        except Exception as e:
            logger.error(f"Smoke test {opt_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"optimizer": opt_name, "error": str(e)})

    return all_results


# CLI

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Optimizer Benchmark")
    parser.add_argument("--mode", choices=["single", "sweep", "full", "smoke"],
                        default="single", help="Benchmark mode")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "muon", "dion", "dion2", "adadion"],
                        help="Optimizer for single/sweep mode")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "vgg16_bn", "vit_small"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "smoke":
        results = run_smoke_test(output_dir)

    elif args.mode == "full":
        results = run_full_benchmark(output_dir)

    elif args.mode == "sweep":
        base = BaseConfig(
            batch_size=args.batch_size,
            model_name=args.model,
            num_workers=args.num_workers,
            mixed_precision=not args.no_amp,
        )
        if args.epochs:
            base.epochs = args.epochs
        results = run_lr_sweep(base, args.optimizer, output_dir, seeds=[42, 123, 456])

    else:  # single
        base = BaseConfig(
            batch_size=args.batch_size,
            seed=args.seed,
            model_name=args.model,
            num_workers=args.num_workers,
            mixed_precision=not args.no_amp,
        )
        if args.epochs:
            base.epochs = args.epochs

        opt_config = get_default_optimizer_config(args.optimizer)
        if args.lr:
            opt_config.lr = args.lr

        run_name = f"{args.model}_{args.optimizer}_seed{args.seed}"
        summary = train_one_run(base, opt_config, output_dir, run_name)
        results = [summary]

    # Final summary
    results_path = os.path.join(output_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"All results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Run':<40} {'Val Acc':>10} {'Best Acc':>10} {'Time (s)':>10} {'Mem (MB)':>10}")
    print("=" * 90)
    for r in results:
        if "error" in r:
            print(f"{r.get('run_name', r.get('optimizer', '?')):<40} {'FAILED':>10}")
        else:
            print(
                f"{r.get('run_name', ''):<40} "
                f"{r.get('final_val_acc', 0):>9.2f}% "
                f"{r.get('best_val_acc', 0):>9.2f}% "
                f"{r.get('total_train_time_sec', 0):>10.0f} "
                f"{r.get('peak_gpu_mem_mb', 0):>10.0f}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
