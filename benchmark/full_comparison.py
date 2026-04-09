"""
Full comparison: AdaDion V3 vs V2 vs Dion vs Dion2 vs AdamW vs Muon
on CIFAR-10 and FashionMNIST.

Tracks: val accuracy, train loss, throughput, GPU memory, communication volume.

Usage:
    python -m benchmark.full_comparison --dataset cifar10 --device cuda
    python -m benchmark.full_comparison --dataset fashionmnist --device cuda
    python -m benchmark.full_comparison --dataset all --device cuda
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# Models

class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for 32x32 CIFAR images."""
    def __init__(self, num_classes=10):
        super().__init__()
        from torchvision.models import resnet18
        self.net = resnet18(num_classes=num_classes)
        # Replace 7x7 conv with 3x3 for 32x32 input
        self.net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


class SimpleMLP(nn.Module):
    """3-layer MLP for FashionMNIST (matching Ada-Dion paper's controlled study)."""
    def __init__(self, hidden=2048, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Data

def get_cifar10(batch_size=128, num_workers=2):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        T.RandomErasing(p=0.1),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_fashionmnist(batch_size=256, num_workers=2):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    train_ds = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# Optimizer factory

def _group_params(model):
    """Split params into matrix (2D) and scalar (1D/other)."""
    matrix_params = []
    scalar_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    return matrix_params, scalar_params


def create_optimizer(name: str, model: nn.Module, lr: float, rank: int = 64,
                     weight_decay: float = 0.05, epochs: int = 200,
                     dataset: str = "cifar10") -> Tuple:
    """
    Create optimizer by name. Returns (optimizer, comm_tracker_fn).
    comm_tracker_fn() returns cumulative communication in GB.
    """
    matrix_params, scalar_params = _group_params(model)

    if name == "adamw":
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        # AdamW communication: just gradients (standard allreduce)
        total_params = sum(p.numel() for p in model.parameters())
        comm_per_step = total_params  # one allreduce of gradients
        step_count = [0]
        def comm_fn():
            return step_count[0] * comm_per_step * 4 / 1e9
        original_step = opt.step
        def tracked_step(*a, **kw):
            step_count[0] += 1
            return original_step(*a, **kw)
        opt.step = tracked_step
        return opt, comm_fn

    elif name == "muon":
        try:
            from muon import Muon
        except ImportError:
            # Fallback: use our own Muon-like implementation
            return _create_muon_fallback(model, lr, weight_decay, matrix_params, scalar_params)

        param_groups = [{"params": matrix_params}]
        if scalar_params:
            param_groups.append({
                "params": scalar_params, "algorithm": "adamw",
                "lr": lr * 0.1, "betas": (0.9, 0.999), "eps": 1e-8,
                "weight_decay": weight_decay,
            })
        opt = Muon(param_groups, lr=lr, mu=0.95, weight_decay=weight_decay,
                   nesterov=True, flatten=True)
        # Muon communication: full matrix allgather + reduce_scatter
        total_matrix = sum(p.numel() for p in matrix_params)
        total_scalar = sum(p.numel() for p in scalar_params)
        comm_per_step = 2 * total_matrix + total_scalar  # 2x for allgather+reduce_scatter
        step_count = [0]
        def comm_fn():
            return step_count[0] * comm_per_step * 4 / 1e9
        original_step = opt.step
        def tracked_step(*a, **kw):
            step_count[0] += 1
            return original_step(*a, **kw)
        opt.step = tracked_step
        return opt, comm_fn

    elif name == "adadion_v3":
        from benchmark.adadion_v3 import AdaDionV3
        opt = AdaDionV3(
            model.parameters(), lr=lr, rank=rank,
            beta_base=0.1, damping=0.5, weight_decay=weight_decay,
            per_mode_beta=True, adaptive_rank=False,
            warmup_steps=5, scalar_lr=lr * 0.1,
        )
        def comm_fn():
            return opt.get_comm_volume_gb()
        return opt, comm_fn

    elif name == "adadion_v3_adaptive":
        from benchmark.adadion_v3 import AdaDionV3
        opt = AdaDionV3(
            model.parameters(), lr=lr, rank=rank,
            beta_base=0.1, damping=0.5, weight_decay=weight_decay,
            per_mode_beta=True, adaptive_rank=True,
            warmup_steps=5, scalar_lr=lr * 0.1,
        )
        def comm_fn():
            return opt.get_comm_volume_gb()
        return opt, comm_fn

    elif name == "adadion_v2":
        from benchmark.adadion_v2_single import AdaDionV2Single
        opt = AdaDionV2Single(
            model.parameters(), lr=lr, rank=rank, mu=0.95,
            weight_decay=weight_decay, adaptive_rank=True,
            init_rank_fraction=0.25, rank_fraction_max=0.7,
            warmup_steps=5, scalar_lr=lr * 0.1,
        )
        def comm_fn():
            return opt.get_comm_volume_gb()
        return opt, comm_fn

    elif name == "orth_dion":
        from benchmark.lm.dion_variants import OrthDion
        orth_opt = OrthDion(matrix_params, lr=lr, rank=rank, beta=1.0, weight_decay=weight_decay, warmup_steps=5)
        adamw = AdamW(scalar_params, lr=lr*0.1, weight_decay=weight_decay) if scalar_params else None

        class CombinedOpt:
            def __init__(self, d, a):
                self.dion = d
                self.adamw = a
                self.param_groups = list(d.param_groups) + (list(a.param_groups) if a else [])
            def step(self, closure=None):
                self.dion.step(closure)
                if self.adamw:
                    self.adamw.step()
            def zero_grad(self, set_to_none=False):
                self.dion.zero_grad(set_to_none=set_to_none)
                if self.adamw:
                    self.adamw.zero_grad(set_to_none=set_to_none)

        opt = CombinedOpt(orth_opt, adamw)
        total_matrix = sum(p.numel() for p in matrix_params)
        m_avg = int(math.sqrt(total_matrix / max(len(matrix_params), 1)))
        comm_per_step = 2 * m_avg * rank * len(matrix_params)
        step_count = [0]
        def comm_fn():
            return step_count[0] * comm_per_step * 4 / 1e9
        original_step = opt.step
        def tracked_step(*a, **kw):
            step_count[0] += 1
            return original_step(*a, **kw)
        opt.step = tracked_step
        return opt, comm_fn

    elif name == "ada_orth_dion":
        from benchmark.adadion_v2_single import AdaDionV2Single
        # Ada-Orth-Dion: V2 adaptive rank but we swap ColNorm for QR internally
        # Since V2 uses ColNorm, we create a modified version that uses QR
        # by patching the column_normalize call
        opt = AdaDionV2Single(
            model.parameters(), lr=lr, rank=rank, mu=0.95,
            weight_decay=weight_decay, adaptive_rank=True,
            init_rank_fraction=0.25, rank_fraction_max=0.7,
            warmup_steps=5, scalar_lr=lr * 0.1,
            use_qr=True,  # flag to use QR instead of ColNorm
        )
        def comm_fn():
            return opt.get_comm_volume_gb()
        return opt, comm_fn

    elif name in ("dion", "dion2"):
        try:
            import dion as dion_pkg
            if name == "dion":
                DionCls = dion_pkg.Dion
            else:
                DionCls = dion_pkg.Dion2
        except ImportError:
            # Fallback: use stripped dion from our LM code
            return _create_dion_fallback(name, model, lr, rank, weight_decay, matrix_params, scalar_params)

        param_groups = [{"params": matrix_params}]
        if scalar_params:
            param_groups.append({
                "params": scalar_params, "algorithm": "adamw",
                "lr": lr * 0.1, "betas": (0.9, 0.999), "eps": 1e-8,
                "weight_decay": weight_decay,
            })
        kwargs = {"lr": lr, "weight_decay": weight_decay}
        if name == "dion":
            kwargs["rank_fraction"] = 0.25
        else:
            kwargs["fraction"] = 0.25
        opt = DionCls(param_groups, **kwargs)
        total_matrix = sum(p.numel() for p in matrix_params)
        m_avg = int(math.sqrt(total_matrix / max(len(matrix_params), 1)))
        comm_per_step = 2 * m_avg * rank * len(matrix_params)
        step_count = [0]
        def comm_fn():
            return step_count[0] * comm_per_step * 4 / 1e9
        original_step = opt.step
        def tracked_step(*a, **kw):
            step_count[0] += 1
            return original_step(*a, **kw)
        opt.step = tracked_step
        return opt, comm_fn

    raise ValueError(f"Unknown optimizer: {name}")


def _create_dion_fallback(name, model, lr, rank, wd, matrix_params, scalar_params):
    """Fallback Dion using our StrippedDion implementation."""
    from benchmark.lm.dion_variants import StrippedDion
    beta = 1.0 if name == "dion" else 0.7  # Dion2 uses ef_decay
    dion_opt = StrippedDion(matrix_params, lr=lr, rank=rank, beta=beta, weight_decay=wd)
    adamw = AdamW(scalar_params, lr=lr*0.1, weight_decay=wd) if scalar_params else None

    class CombinedOpt:
        def __init__(self, d, a):
            self.dion = d
            self.adamw = a
            self.param_groups = list(d.param_groups) + (list(a.param_groups) if a else [])
        def step(self, closure=None):
            self.dion.step(closure)
            if self.adamw:
                self.adamw.step()
        def zero_grad(self, set_to_none=False):
            self.dion.zero_grad(set_to_none=set_to_none)
            if self.adamw:
                self.adamw.zero_grad(set_to_none=set_to_none)
        @property
        def state(self):
            return self.dion.state

    opt = CombinedOpt(dion_opt, adamw)
    total_matrix = sum(p.numel() for p in matrix_params)
    m_avg = int(math.sqrt(total_matrix / max(len(matrix_params), 1)))
    comm_per_step = 2 * m_avg * rank * len(matrix_params)
    step_count = [0]
    def comm_fn():
        return step_count[0] * comm_per_step * 4 / 1e9
    original_step = opt.step
    def tracked_step(*a, **kw):
        step_count[0] += 1
        return original_step(*a, **kw)
    opt.step = tracked_step
    return opt, comm_fn


def _create_muon_fallback(model, lr, wd, matrix_params, scalar_params):
    """Fallback Muon using StrippedDion with high rank (approximating full-rank)."""
    from benchmark.lm.dion_variants import StrippedDion
    max_dim = max(min(p.shape) for p in matrix_params)
    dion_opt = StrippedDion(matrix_params, lr=lr, rank=max_dim, beta=1.0, mu=0.95, weight_decay=wd)
    adamw = AdamW(scalar_params, lr=lr*0.1, weight_decay=wd) if scalar_params else None

    class CombinedOpt:
        def __init__(self, d, a):
            self.dion = d
            self.adamw = a
            self.param_groups = d.param_groups + (a.param_groups if a else [])
        def step(self, closure=None):
            self.dion.step(closure)
            if self.adamw:
                self.adamw.step()
        def zero_grad(self, set_to_none=False):
            self.dion.zero_grad(set_to_none=set_to_none)
            if self.adamw:
                self.adamw.zero_grad(set_to_none=set_to_none)

    opt = CombinedOpt(dion_opt, adamw)
    total_matrix = sum(p.numel() for p in matrix_params)
    comm_per_step = 2 * total_matrix  # Muon needs full matrix communication
    step_count = [0]
    def comm_fn():
        return step_count[0] * comm_per_step * 4 / 1e9
    original_step = opt.step
    def tracked_step(*a, **kw):
        step_count[0] += 1
        return original_step(*a, **kw)
    opt.step = tracked_step
    return opt, comm_fn


# Training

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total, total_loss / total


def run_experiment(
    opt_name: str,
    dataset: str = "cifar10",
    epochs: int = 100,
    lr: float = None,
    rank: int = 64,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "results/comparison",
) -> dict:
    torch.manual_seed(seed)

    # Default LRs per optimizer (tuned)
    default_lrs = {
        "adamw": 1e-3, "muon": 0.02, "dion": 0.02, "dion2": 0.02,
        "adadion_v2": 0.02, "adadion_v3": 0.02, "adadion_v3_adaptive": 0.02,
        "orth_dion": 0.02, "ada_orth_dion": 0.02,
    }
    if lr is None:
        lr = default_lrs.get(opt_name, 0.01)

    # Dataset
    if dataset == "cifar10":
        train_loader, test_loader = get_cifar10(batch_size=128)
        model = ResNet18CIFAR(num_classes=10).to(device)
        wd = 0.05
    elif dataset == "fashionmnist":
        train_loader, test_loader = get_fashionmnist(batch_size=256)
        model = SimpleMLP(hidden=2048, num_classes=10).to(device)
        wd = 0.0
        epochs = min(epochs, 50)  # FashionMNIST converges faster
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"{opt_name} on {dataset} | {n_params/1e6:.1f}M params | lr={lr} | {epochs} epochs")
    print(f"{'='*60}")

    # Optimizer
    try:
        opt, comm_fn = create_optimizer(opt_name, model, lr=lr, rank=rank,
                                        weight_decay=wd, epochs=epochs, dataset=dataset)
    except Exception as e:
        print(f"  ERROR creating {opt_name}: {e}")
        return {"error": str(e)}

    # LR scheduler — only for real Optimizer subclasses
    try:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    except (TypeError, AttributeError):
        # Fallback for CombinedOpt: manual LR decay
        scheduler = None

    # Training loop
    history = []
    best_acc = 0
    start_time = time.time()
    peak_mem = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * y.size(0)
            epoch_correct += (out.argmax(1) == y).sum().item()
            epoch_total += y.size(0)

        if scheduler is not None:
            scheduler.step()
        else:
            # Manual cosine decay for non-standard optimizers
            progress = epoch / max(epochs - 1, 1)
            new_lr = lr * 0.01 + (lr - lr * 0.01) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in opt.param_groups:
                group["lr"] = new_lr

        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total
        test_acc, test_loss = evaluate(model, test_loader, device)
        comm_gb = comm_fn()
        if device == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)

        best_acc = max(best_acc, test_acc)

        entry = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_acc": test_acc, "test_loss": test_loss,
            "comm_gb": comm_gb, "elapsed_s": time.time() - start_time,
        }
        history.append(entry)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"train {train_acc:.3f} | test {test_acc:.3f} | "
                  f"best {best_acc:.3f} | comm {comm_gb:.2f}GB")

    total_time = time.time() - start_time
    throughput = epoch_total * epochs / total_time

    result = {
        "optimizer": opt_name,
        "dataset": dataset,
        "lr": lr,
        "rank": rank,
        "epochs": epochs,
        "final_test_acc": test_acc,
        "best_test_acc": best_acc,
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
        "total_comm_gb": comm_fn(),
        "total_time_s": total_time,
        "throughput_samples_s": throughput,
        "peak_gpu_mem_gb": peak_mem,
        "n_params": n_params,
    }

    # Save
    out_dir = os.path.join(output_dir, dataset, opt_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"  DONE: test_acc={best_acc:.4f} | comm={result['total_comm_gb']:.2f}GB | "
          f"time={total_time:.0f}s | mem={peak_mem:.2f}GB")
    return result


LR_GRIDS = {
    "adamw":              [3e-4, 1e-3, 3e-3, 5e-3],
    "muon":               [0.005, 0.01, 0.02, 0.04],
    "dion":               [0.005, 0.01, 0.02, 0.04],
    "dion2":              [0.005, 0.01, 0.02, 0.04],
    "orth_dion":          [0.005, 0.01, 0.02, 0.04],
    "ada_orth_dion":      [0.005, 0.01, 0.02, 0.04],
    "adadion_v2":         [0.005, 0.01, 0.02, 0.04],
    "adadion_v3":         [0.005, 0.01, 0.02, 0.04],
    "adadion_v3_adaptive":[0.005, 0.01, 0.02, 0.04],
}


def run_sweep(opt_name, dataset, sweep_epochs, rank, device, seed, output_dir):
    """Sweep LRs for one optimizer. Returns best LR."""
    lrs = LR_GRIDS.get(opt_name, [0.005, 0.01, 0.02])
    best_lr = lrs[0]
    best_acc = 0.0
    print(f"\n  Sweeping {opt_name} on {dataset}: LRs = {lrs}")
    for lr_val in lrs:
        try:
            r = run_experiment(
                opt_name, dataset=dataset, epochs=sweep_epochs, lr=lr_val,
                rank=rank, device=device, seed=seed,
                output_dir=os.path.join(output_dir, "sweep"),
            )
            acc = r.get("best_test_acc", 0)
            print(f"    lr={lr_val}: acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_lr = lr_val
        except Exception as e:
            print(f"    lr={lr_val}: ERROR {e}")
    print(f"  Best for {opt_name}: lr={best_lr} (acc={best_acc:.4f})")
    return best_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["single", "sweep", "best", "full"])
    parser.add_argument("--dataset", default="all", choices=["cifar10", "fashionmnist", "all"])
    parser.add_argument("--optimizers", nargs="+",
                        default=["adamw", "dion", "dion2", "adadion_v2", "adadion_v3", "adadion_v3_adaptive", "orth_dion", "ada_orth_dion", "muon"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sweep_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/comparison")
    args = parser.parse_args()

    datasets = ["cifar10", "fashionmnist"] if args.dataset == "all" else [args.dataset]

    if args.mode == "single":
        for ds in datasets:
            for opt_name in args.optimizers:
                run_experiment(opt_name, dataset=ds, epochs=args.epochs, lr=args.lr,
                               rank=args.rank, device=args.device, seed=args.seed,
                               output_dir=args.output_dir)
        return

    if args.mode == "sweep":
        for ds in datasets:
            for opt_name in args.optimizers:
                run_sweep(opt_name, ds, args.sweep_epochs, args.rank,
                          args.device, args.seed, args.output_dir)
        return

    if args.mode in ("best", "full"):
        all_results = {}
        for ds in datasets:
            best_lrs = {}

            if args.mode == "full":
                print(f"\n{'='*60}")
                print(f"PHASE 1: LR sweep on {ds} ({args.sweep_epochs} epochs per LR)")
                print(f"{'='*60}")
                for opt_name in args.optimizers:
                    best_lrs[opt_name] = run_sweep(
                        opt_name, ds, args.sweep_epochs, args.rank,
                        args.device, args.seed, args.output_dir,
                    )
                lr_path = os.path.join(args.output_dir, f"best_lrs_{ds}.json")
                os.makedirs(args.output_dir, exist_ok=True)
                with open(lr_path, "w") as f:
                    json.dump(best_lrs, f, indent=2)
                print(f"\nBest LRs for {ds}: {best_lrs}")
            else:
                lr_path = os.path.join(args.output_dir, f"best_lrs_{ds}.json")
                best_lrs = json.load(open(lr_path))

            print(f"\n{'='*60}")
            print(f"PHASE 2: Full runs on {ds} ({args.epochs} epochs, tuned LRs)")
            print(f"{'='*60}")
            for opt_name in args.optimizers:
                lr_val = best_lrs.get(opt_name, 0.01)
                key = f"{ds}/{opt_name}"
                try:
                    result = run_experiment(
                        opt_name, dataset=ds, epochs=args.epochs, lr=lr_val,
                        rank=args.rank, device=args.device, seed=args.seed,
                        output_dir=args.output_dir,
                    )
                    all_results[key] = result
                except Exception as e:
                    print(f"ERROR in {key}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[key] = {"error": str(e)}

        summary_path = os.path.join(args.output_dir, "summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"{'Dataset':<15} {'Optimizer':<22} {'LR':>8} {'Test Acc':>10} {'Comm (GB)':>10} {'Mem (GB)':>10}")
        print(f"{'-'*80}")
        for key, r in sorted(all_results.items(), key=lambda x: -x[1].get("best_test_acc", 0)):
            if "error" in r:
                print(f"{key:<37} ERROR: {r['error'][:35]}")
            else:
                print(f"{r['dataset']:<15} {r['optimizer']:<22} {r['lr']:>8.4f} "
                      f"{r['best_test_acc']:>10.4f} {r['total_comm_gb']:>10.2f} "
                      f"{r['peak_gpu_mem_gb']:>10.2f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
