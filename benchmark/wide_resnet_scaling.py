#!/usr/bin/env python3
"""
Wide ResNet scaling experiment: how does AdaDion behave as model width increases?

Tests AdaDion V2 vs AdamW, Muon, Dion on WideResNet-28 with width factors
2, 4, 10 on CIFAR-10 and CIFAR-100.

Goals:
  1. Does AdaDion outperform AdamW in end-to-end training time on larger models?
  2. How does adaptive rank behave as weight matrices grow?
  3. Is Dion's fixed d/4 rank rule robust, or does AdaDion's adaptation help?
  4. Communication overhead vs accuracy at different model scales.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_wide import create_wide_resnet, WIDE_RESNET_CONFIGS
from optimizers import create_optimizer, group_params_for_hybrid
from configs import AdaDionConfig, AdamWConfig, MuonConfig, DionConfig
from metrics import compute_accuracy, compute_gradient_norm

# GradScaler compat
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


def get_data_loaders(dataset="cifar10", batch_size=128, num_workers=4, seed=42):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.1),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    DatasetClass = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    num_classes = 10 if dataset == "cifar10" else 100

    train_ds = DatasetClass(root="./data", train=True, download=True, transform=train_transform)
    test_ds = DatasetClass(root="./data", train=False, download=True, transform=test_transform)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


def make_optimizer_config(opt_name, model_scale="small"):
    """Create optimizer config tuned for Wide ResNet scale."""
    if opt_name == "adamw":
        return AdamWConfig(lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
    elif opt_name == "muon":
        return MuonConfig(
            lr=0.01, mu=0.95, nesterov=True, adjust_lr="rms_norm",
            weight_decay=0.01, flatten=True,
            scalar_lr=1e-3, scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
        )
    elif opt_name == "dion":
        return DionConfig(
            lr=0.01, rank_fraction=0.25, weight_decay=0.01,
            scalar_lr=1e-3, scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
        )
    elif opt_name == "adadion":
        return AdaDionConfig(
            lr=0.01, mu=0.95, rank_fraction_max=0.7, weight_decay=0.1,
            adaptive_rank=True, init_rank_fraction=0.5,
            erank_ema_beta=0.5, rank_scale=1.5,
            rank_min=16, rank_quantize=8, rank_step_up=16, rank_step_down=8,
            scalar_lr=0.01, scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
        )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_c1, total_c5, total_n = 0, 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        output = model(images)
        loss = F.cross_entropy(output, targets)
        total_loss += loss.item() * targets.size(0)
        t1, t5 = compute_accuracy(output, targets, topk=(1, 5))
        total_c1 += t1 * targets.size(0) / 100
        total_c5 += t5 * targets.size(0) / 100
        total_n += targets.size(0)
    return {"loss": total_loss / total_n, "acc": total_c1 / total_n * 100,
            "top5": total_c5 / total_n * 100}


def train_single_run(wrn_depth, wrn_width, opt_name, dataset, epochs, seed, output_dir, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    num_classes = 10 if dataset == "cifar10" else 100
    run_name = f"wrn{wrn_depth}_{wrn_width}_{opt_name}_{dataset}_seed{seed}"

    logger.info(f"=== {run_name} on GPU {gpu_id} ===")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    train_loader, test_loader, num_classes = get_data_loaders(dataset, 128, 4, seed)
    steps_per_epoch = len(train_loader)

    model = create_wide_resnet(wrn_depth, wrn_width, num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    matrix_params = sum(p.numel() for p in model.parameters() if p.ndim >= 2)
    logger.info(f"  Model: WRN-{wrn_depth}-{wrn_width}, params: {total_params:,}, matrix: {matrix_params:,}")

    # Describe largest weight matrix
    largest = max((p.numel(), n, p.shape) for n, p in model.named_parameters() if p.ndim >= 2)
    logger.info(f"  Largest matrix: {largest[1]} {largest[2]}")

    opt_config = make_optimizer_config(opt_name)
    is_spectral = opt_name in ("muon", "dion", "adadion")
    optimizer, flatten_wrapper = create_optimizer(model, opt_config)

    # LR scheduler: cosine with warmup
    total_steps = epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    epoch_history = []
    rank_history = []
    best_val_acc = 0
    total_time = 0

    for epoch in range(epochs):
        model.train()
        t0 = time.perf_counter()
        epoch_loss, epoch_correct, epoch_samples = 0, 0, 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if flatten_wrapper:
                flatten_wrapper.flatten_for_optimizer()
            optimizer.step()
            scheduler.step()
            if flatten_wrapper:
                flatten_wrapper.restore_shapes()

            with torch.no_grad():
                epoch_correct += output.argmax(1).eq(targets).sum().item()
            epoch_loss += loss.item() * targets.size(0)
            epoch_samples += targets.size(0)

        epoch_time = time.perf_counter() - t0
        total_time += epoch_time
        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples * 100

        val = evaluate(model, test_loader, device)
        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]

        # Track rank
        rank_data = {}
        if hasattr(optimizer, "get_rank"):
            rank_data = optimizer.get_rank()
        if hasattr(optimizer, "get_effective_rank"):
            rank_data.update({"erank_" + k: v for k, v in optimizer.get_effective_rank().items()})

        epoch_entry = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val["loss"], "val_acc": val["acc"], "val_top5": val["top5"],
            "epoch_time": epoch_time, "best_val_acc": best_val_acc,
        }
        epoch_history.append(epoch_entry)
        if rank_data:
            rank_history.append({"epoch": epoch, **rank_data})

        if epoch % 10 == 0 or epoch == epochs - 1:
            mem = torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0
            logger.info(
                f"  [{run_name}] Ep {epoch}/{epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                f"Val: {val['loss']:.4f}/{val['acc']:.1f}% | "
                f"Best: {best_val_acc:.1f}% | "
                f"Time: {epoch_time:.1f}s | Mem: {mem:.0f}MB"
                + (f" | Ranks: {rank_data}" if rank_data else "")
            )

    # Save results
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0
    summary = {
        "run_name": run_name,
        "model": f"wrn-{wrn_depth}-{wrn_width}",
        "optimizer": opt_name,
        "dataset": dataset,
        "seed": seed,
        "total_params": total_params,
        "matrix_params": matrix_params,
        "best_val_acc": best_val_acc,
        "final_val_loss": epoch_history[-1]["val_loss"],
        "final_val_acc": epoch_history[-1]["val_acc"],
        "total_time_sec": total_time,
        "avg_epoch_time_sec": total_time / epochs,
        "peak_mem_mb": peak_mem,
        "epochs": epochs,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "epoch_metrics.json"), "w") as f:
        json.dump(epoch_history, f, indent=2)
    if rank_history:
        with open(os.path.join(run_dir, "rank_history.json"), "w") as f:
            json.dump(rank_history, f, indent=2)

    logger.info(f"  [{run_name}] FINISHED | Best: {best_val_acc:.2f}% | Time: {total_time:.0f}s")
    return summary


def run_scaling_experiment(output_dir, dataset="cifar10", epochs=100):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    width_configs = [
        (28, 2),   # 1.5M params
        (28, 4),   # 5.9M params
        (28, 10),  # 36.5M params
    ]

    optimizers = ["adamw", "dion", "adadion"]

    for depth, width in width_configs:
        for opt_name in optimizers:
            summary = train_single_run(depth, width, opt_name, dataset, epochs, 42, output_dir)
            all_results.append(summary)

            with open(os.path.join(output_dir, "scaling_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 90)
    print(f"{'Model':<15} {'Optimizer':<12} {'Params':>10} {'Val Acc':>10} {'Time (s)':>10} {'Mem (MB)':>10}")
    print("=" * 90)
    for r in all_results:
        print(f"{r['model']:<15} {r['optimizer']:<12} {r['total_params']:>10,} "
              f"{r['best_val_acc']:>9.2f}% {r['total_time_sec']:>10.0f} {r['peak_mem_mb']:>10.0f}")
    print("=" * 90)

    with open(os.path.join(output_dir, "scaling_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results/wide_resnet")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    # Single run mode
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.optimizer:
        # Single run
        train_single_run(args.depth, args.width, args.optimizer,
                         args.dataset, args.epochs, args.seed, args.output_dir, args.gpu)
    else:
        # Full scaling experiment
        run_scaling_experiment(args.output_dir, args.dataset, args.epochs)
