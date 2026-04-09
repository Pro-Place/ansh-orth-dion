"""
LLaMA 300M optimizer comparison with HP tuning.

Runs a grid of learning rates for each optimizer, picks the best,
then does a full-length run. Tracks: val loss/ppl, communication volume,
throughput, GPU memory, per-step diagnostics.

Usage:
    # Full comparison (HP sweep + best runs)
    python -m benchmark.lm.llama_comparison --mode full --device cuda

    # Quick test (500 steps)
    python -m benchmark.lm.llama_comparison --mode smoke --device cuda

    # Single optimizer
    python -m benchmark.lm.llama_comparison --mode single --optimizer adadion_v3 --lr 0.005
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.lm.llama import LLaMA, create_llama_300m
from benchmark.lm.data import load_wikitext103, create_dataloaders
from benchmark.adadion_v3 import AdaDionV3
from benchmark.adadion_v2_single import AdaDionV2Single
from benchmark.lm.dion_variants import StrippedDion


# Optimizer factory

def create_optimizer(name: str, model: nn.Module, lr: float, rank: int = 64,
                     weight_decay: float = 0.1) -> Tuple:
    """Create optimizer. Returns (optimizer, comm_volume_fn, description)."""
    matrix_params = []
    scalar_params = []
    embed_params = []

    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "tok_emb" not in pname and "lm_head" not in pname:
            matrix_params.append(p)
        elif "tok_emb" in pname or "lm_head" in pname:
            embed_params.append(p)
        else:
            scalar_params.append(p)

    mat_count = sum(p.numel() for p in matrix_params)
    scl_count = sum(p.numel() for p in scalar_params)
    emb_count = sum(p.numel() for p in embed_params)

    desc = f"{name} lr={lr} rank={rank}"

    if name == "adamw":
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        total = sum(p.numel() for p in model.parameters())
        step_count = [0]
        def comm_fn():
            return step_count[0] * total * 4 / 1e9
        orig = opt.step
        def tracked(*a, **kw):
            step_count[0] += 1; return orig(*a, **kw)
        opt.step = tracked
        return opt, comm_fn, desc

    elif name == "adadion_v3":
        opt = AdaDionV3(
            model.parameters(), lr=lr, rank=rank,
            beta_base=0.1, damping=0.5, weight_decay=weight_decay,
            per_mode_beta=True, adaptive_rank=False,
            warmup_steps=200, R_max=2.0,
            scalar_lr=3e-4, scalar_eps=1e-8,
        )
        return opt, opt.get_comm_volume_gb, desc

    elif name == "adadion_v3_adaptive":
        opt = AdaDionV3(
            model.parameters(), lr=lr, rank=rank,
            beta_base=0.1, damping=0.5, weight_decay=weight_decay,
            per_mode_beta=True, adaptive_rank=True,
            warmup_steps=200, R_max=2.0,
            scalar_lr=3e-4, scalar_eps=1e-8,
        )
        return opt, opt.get_comm_volume_gb, desc

    elif name == "adadion_v2":
        opt = AdaDionV2Single(
            model.parameters(), lr=lr, rank=rank, mu=0.95,
            weight_decay=weight_decay, adaptive_rank=True,
            warmup_steps=200, scalar_lr=3e-4,
        )
        return opt, opt.get_comm_volume_gb, desc

    elif name == "dion":
        dion = StrippedDion(
            matrix_params, lr=lr, rank=rank, beta=1.0,
            weight_decay=weight_decay, warmup_steps=200,
        )
        adamw_groups = []
        if scalar_params:
            adamw_groups.append({"params": scalar_params, "lr": 3e-4, "weight_decay": weight_decay})
        if embed_params:
            adamw_groups.append({"params": embed_params, "lr": 3e-4, "weight_decay": weight_decay})
        adamw = AdamW(adamw_groups, lr=3e-4) if adamw_groups else None

        total_matrix = sum(p.numel() for p in matrix_params)
        m_avg = int(math.sqrt(total_matrix / max(len(matrix_params), 1)))
        step_count = [0]
        def comm_fn():
            return step_count[0] * 2 * m_avg * rank * len(matrix_params) * 4 / 1e9

        class Combo:
            def __init__(s):
                s.param_groups = list(dion.param_groups) + (list(adamw.param_groups) if adamw else [])
            def step(s, closure=None):
                step_count[0] += 1
                dion.step(closure)
                if adamw: adamw.step()
            def zero_grad(s, set_to_none=False):
                dion.zero_grad(set_to_none=set_to_none)
                if adamw: adamw.zero_grad(set_to_none=set_to_none)

        return Combo(), comm_fn, desc

    elif name == "dion_beta03":
        dion = StrippedDion(
            matrix_params, lr=lr, rank=rank, beta=0.3,
            weight_decay=weight_decay, warmup_steps=200,
        )
        adamw_groups = []
        if scalar_params:
            adamw_groups.append({"params": scalar_params, "lr": 3e-4, "weight_decay": weight_decay})
        if embed_params:
            adamw_groups.append({"params": embed_params, "lr": 3e-4, "weight_decay": weight_decay})
        adamw = AdamW(adamw_groups, lr=3e-4) if adamw_groups else None

        total_matrix = sum(p.numel() for p in matrix_params)
        m_avg = int(math.sqrt(total_matrix / max(len(matrix_params), 1)))
        step_count = [0]
        def comm_fn():
            return step_count[0] * 2 * m_avg * rank * len(matrix_params) * 4 / 1e9

        class Combo:
            def __init__(s):
                s.param_groups = list(dion.param_groups) + (list(adamw.param_groups) if adamw else [])
            def step(s, closure=None):
                step_count[0] += 1
                dion.step(closure)
                if adamw: adamw.step()
            def zero_grad(s, set_to_none=False):
                dion.zero_grad(set_to_none=set_to_none)
                if adamw: adamw.zero_grad(set_to_none=set_to_none)

        return Combo(), comm_fn, desc

    elif name == "muon":
        # Full-rank Dion as Muon approximation
        max_dim = min(max(min(p.shape) for p in matrix_params), 512)
        dion = StrippedDion(
            matrix_params, lr=lr, rank=max_dim, beta=1.0, mu=0.95,
            weight_decay=weight_decay, warmup_steps=200,
        )
        adamw_groups = []
        if scalar_params:
            adamw_groups.append({"params": scalar_params, "lr": 3e-4, "weight_decay": weight_decay})
        if embed_params:
            adamw_groups.append({"params": embed_params, "lr": 3e-4, "weight_decay": weight_decay})
        adamw = AdamW(adamw_groups, lr=3e-4) if adamw_groups else None

        total_matrix = sum(p.numel() for p in matrix_params)
        step_count = [0]
        def comm_fn():
            return step_count[0] * 2 * total_matrix * 4 / 1e9

        class Combo:
            def __init__(s):
                s.param_groups = list(dion.param_groups) + (list(adamw.param_groups) if adamw else [])
            def step(s, closure=None):
                step_count[0] += 1
                dion.step(closure)
                if adamw: adamw.step()
            def zero_grad(s, set_to_none=False):
                dion.zero_grad(set_to_none=set_to_none)
                if adamw: adamw.zero_grad(set_to_none=set_to_none)

        return Combo(), comm_fn, desc

    raise ValueError(f"Unknown optimizer: {name}")


# LR schedule

def get_lr(step: int, max_steps: int, peak_lr: float, warmup: int = 2000) -> float:
    min_lr = peak_lr * 0.1
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# Training

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
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
    avg = total_loss / max(total_tokens, 1)
    return {"val_loss": avg, "val_ppl": math.exp(min(avg, 20))}


def run_training(
    opt_name: str, lr: float, rank: int = 64,
    max_steps: int = 10000, batch_size: int = 4, seq_len: int = 1024,
    eval_interval: int = 500, log_interval: int = 100,
    device: str = "cuda", seed: int = 42,
    output_dir: str = "results/llama300m",
    tag: str = "",
) -> dict:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Model
    model = LLaMA(
        vocab_size=50257, max_seq_len=seq_len,
        dim=1024, n_layers=20, n_heads=16, ffn_dim=2816,
        tie_weights=True,
    ).to(device)
    n_params = sum(p.numel() for p in set(model.parameters()))
    print(f"\nLLaMA: {n_params/1e6:.1f}M params")

    # Data
    train_ds, val_ds = load_wikitext103(seq_len=seq_len)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=batch_size)

    # Optimizer
    opt, comm_fn, desc = create_optimizer(opt_name, model, lr=lr, rank=rank)

    name = f"{opt_name}_lr{lr}_r{rank}"
    if tag:
        name = f"{name}_{tag}"
    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"{desc} | {max_steps} steps | batch={batch_size} | seq={seq_len}")
    print(f"{'='*60}")

    model.train()
    train_iter = iter(train_loader)
    start_time = time.time()
    epoch = 0

    step_log = []
    val_log = []
    best_val_loss = float("inf")

    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # LR schedule
        lr_now = get_lr(step, max_steps, lr, warmup=min(2000, max_steps // 10))
        for group in opt.param_groups:
            group["lr"] = lr_now

        # Forward + backward
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

        opt.step()
        opt.zero_grad(set_to_none=True)

        train_loss = loss.item()
        comm_gb = comm_fn()

        step_log.append({
            "step": step, "train_loss": train_loss, "lr": lr_now,
            "grad_norm": grad_norm, "comm_gb": comm_gb,
            "elapsed_s": time.time() - start_time,
        })

        if step % log_interval == 0:
            tok_s = (step + 1) * batch_size * seq_len / (time.time() - start_time)
            print(f"step {step:6d} | loss {train_loss:.4f} | "
                  f"ppl {math.exp(min(train_loss, 20)):8.2f} | "
                  f"lr {lr_now:.6f} | gnorm {grad_norm:.3f} | "
                  f"{tok_s:.0f} tok/s | comm {comm_gb:.2f}GB")

        if step > 0 and step % eval_interval == 0:
            val = evaluate(model, val_loader, device)
            val["step"] = step
            val["comm_gb"] = comm_gb
            val_log.append(val)
            if val["val_loss"] < best_val_loss:
                best_val_loss = val["val_loss"]
            print(f"  >> EVAL: val_loss={val['val_loss']:.4f} ppl={val['val_ppl']:.2f} "
                  f"best={best_val_loss:.4f}")

    # Final eval
    final_val = evaluate(model, val_loader, device)
    final_val["step"] = max_steps
    val_log.append(final_val)
    best_val_loss = min(best_val_loss, final_val["val_loss"])

    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

    result = {
        "optimizer": opt_name, "lr": lr, "rank": rank,
        "max_steps": max_steps, "batch_size": batch_size, "seq_len": seq_len,
        "n_params": n_params,
        "final_val_loss": final_val["val_loss"],
        "final_val_ppl": final_val["val_ppl"],
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "total_comm_gb": comm_fn(),
        "total_time_s": total_time,
        "throughput_tok_s": max_steps * batch_size * seq_len / total_time,
        "peak_gpu_mem_gb": peak_mem,
    }

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "step_log.json"), "w") as f:
        json.dump(step_log, f)
    with open(os.path.join(out_dir, "val_log.json"), "w") as f:
        json.dump(val_log, f)

    print(f"\nDONE: val_loss={best_val_loss:.4f} ppl={math.exp(min(best_val_loss,20)):.2f} "
          f"comm={result['total_comm_gb']:.2f}GB time={total_time:.0f}s")
    return result


# HP sweep

# LR grids per optimizer
LR_GRIDS = {
    "adamw":              [1e-4, 3e-4, 6e-4, 1e-3],
    "adadion_v3":         [0.002, 0.005, 0.01, 0.02],
    "adadion_v3_adaptive":[0.002, 0.005, 0.01, 0.02],
    "adadion_v2":         [0.002, 0.005, 0.01, 0.02],
    "dion":               [0.005, 0.01, 0.02, 0.04],
    "dion_beta03":        [0.005, 0.01, 0.02, 0.04],
    "muon":               [0.005, 0.01, 0.02, 0.04],
}

OPTIMIZERS_ALL = ["adamw", "adadion_v3", "adadion_v3_adaptive", "adadion_v2",
                  "dion", "dion_beta03", "muon"]


def run_hp_sweep(opt_name: str, sweep_steps: int = 3000, device: str = "cuda",
                 output_dir: str = "results/llama300m/sweep") -> Tuple[float, dict]:
    """Run LR sweep for one optimizer. Returns (best_lr, results_dict)."""
    lrs = LR_GRIDS.get(opt_name, [0.005, 0.01, 0.02])
    results = {}
    best_lr = lrs[0]
    best_loss = float("inf")

    for lr_val in lrs:
        print(f"\n--- Sweep {opt_name} lr={lr_val} ({sweep_steps} steps) ---")
        try:
            r = run_training(
                opt_name, lr=lr_val, max_steps=sweep_steps,
                eval_interval=sweep_steps,  # only eval at end
                log_interval=500, device=device,
                output_dir=output_dir, tag=f"sweep",
            )
            results[lr_val] = r
            if r["best_val_loss"] < best_loss:
                best_loss = r["best_val_loss"]
                best_lr = lr_val
        except Exception as e:
            print(f"ERROR: {e}")
            results[lr_val] = {"error": str(e)}

    print(f"\n{'='*40}")
    print(f"SWEEP RESULT for {opt_name}:")
    for lr_val, r in sorted(results.items()):
        if "error" in r:
            print(f"  lr={lr_val}: ERROR")
        else:
            print(f"  lr={lr_val}: val_loss={r['best_val_loss']:.4f}")
    print(f"  BEST: lr={best_lr} (val_loss={best_loss:.4f})")
    print(f"{'='*40}")

    return best_lr, results


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["smoke", "single", "sweep", "best", "full"])
    parser.add_argument("--optimizer", default="adadion_v3")
    parser.add_argument("--optimizers", nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/llama300m")
    args = parser.parse_args()

    opts = args.optimizers or OPTIMIZERS_ALL

    if args.mode == "smoke":
        run_training(args.optimizer, lr=args.lr, max_steps=500,
                     batch_size=args.batch_size, device=args.device)

    elif args.mode == "single":
        steps = args.max_steps or 10000
        run_training(args.optimizer, lr=args.lr, rank=args.rank,
                     max_steps=steps, batch_size=args.batch_size, device=args.device,
                     output_dir=args.output_dir)

    elif args.mode == "sweep":
        sweep_steps = args.max_steps or 3000
        for opt_name in opts:
            run_hp_sweep(opt_name, sweep_steps=sweep_steps, device=args.device,
                         output_dir=os.path.join(args.output_dir, "sweep"))

    elif args.mode == "best":
        # Run full training with pre-determined best LRs
        best_lrs = json.load(open(os.path.join(args.output_dir, "best_lrs.json")))
        full_steps = args.max_steps or 15000
        for opt_name in opts:
            if opt_name in best_lrs:
                run_training(opt_name, lr=best_lrs[opt_name], rank=args.rank,
                             max_steps=full_steps, batch_size=args.batch_size,
                             device=args.device, output_dir=args.output_dir, tag="best")

    elif args.mode == "full":
        sweep_steps = args.max_steps or 3000
        best_lrs = {}
        for opt_name in opts:
            best_lr, _ = run_hp_sweep(
                opt_name, sweep_steps=sweep_steps, device=args.device,
                output_dir=os.path.join(args.output_dir, "sweep"),
            )
            best_lrs[opt_name] = best_lr

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "best_lrs.json"), "w") as f:
            json.dump(best_lrs, f, indent=2)
        print(f"\nBest LRs: {best_lrs}")

        full_steps = 15000
        all_results = {}
        for opt_name in opts:
            r = run_training(
                opt_name, lr=best_lrs[opt_name], rank=args.rank,
                max_steps=full_steps, batch_size=args.batch_size,
                device=args.device, output_dir=args.output_dir, tag="best",
            )
            all_results[opt_name] = r

        # Summary
        print(f"\n{'='*80}")
        print(f"{'Optimizer':<24} {'Val Loss':>10} {'Val PPL':>10} {'Comm (GB)':>10} {'Tok/s':>10}")
        print(f"{'-'*80}")
        for name, r in sorted(all_results.items(), key=lambda x: x[1].get("best_val_loss", 999)):
            if "error" in r:
                print(f"{name:<24} ERROR")
            else:
                print(f"{name:<24} {r['best_val_loss']:>10.4f} {r['best_val_ppl']:>10.2f} "
                      f"{r['total_comm_gb']:>10.2f} {r['throughput_tok_s']:>10.0f}")
        print(f"{'='*80}")

        with open(os.path.join(args.output_dir, "final_summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
