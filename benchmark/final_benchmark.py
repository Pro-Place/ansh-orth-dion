#!/usr/bin/env python3
"""
Final comprehensive benchmark: AdaDion V2 vs all optimizers.

Part 1: ResNet-18 (100ep, 3 seeds) - all 5 optimizers at best configs
Part 2: ViT-Small ablation (100ep, LR sweep + best config comparison)
Part 3: Detailed per-epoch metrics dump for loss curves and plots

Outputs detailed JSON with train_loss, val_loss, val_acc per epoch
for generating publication-quality comparison plots.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar10_benchmark import train_one_run
from configs import BaseConfig, AdaDionConfig, AdamWConfig, MuonConfig, DionConfig, Dion2Config


def make_base(model="resnet18", epochs=100, seed=42, grad_clip=1.0):
    return BaseConfig(
        epochs=epochs,
        warmup_epochs=5,
        batch_size=128,
        num_workers=4,
        log_interval=100,
        save_checkpoints=False,
        model_name=model,
        device="cuda",
        mixed_precision=False,
        seed=seed,
        gradient_clip=grad_clip,
        label_smoothing=0.1,
    )


# Best configs from ablation

def best_adamw():
    return AdamWConfig(lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)

def best_muon():
    return MuonConfig(
        lr=0.02, mu=0.95, nesterov=True, adjust_lr="rms_norm",
        weight_decay=0.01, flatten=True,
        scalar_lr=1e-3, scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
    )

def best_dion():
    return DionConfig(
        lr=0.02, rank_fraction=0.25, weight_decay=0.01,
        scalar_lr=1e-3, scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
    )

def best_dion2():
    return Dion2Config(
        lr=0.02, fraction=0.25, ef_decay=0.95, weight_decay=0.01,
        adjust_lr="spectral_norm", flatten=True,
        scalar_lr=1e-3, scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
    )

def best_adadion():
    return AdaDionConfig(
        lr=0.01, mu=0.95, rank_fraction_max=0.7, weight_decay=0.1,
        adaptive_rank=True, init_rank_fraction=0.5,
        erank_ema_beta=0.5, rank_scale=1.5,
        rank_min=16, rank_quantize=8, rank_step_up=16, rank_step_down=8,
        scalar_lr=0.01, scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
    )


# ViT-specific configs (lower LRs for stability)
def vit_adamw():
    return AdamWConfig(lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)

def vit_muon():
    return MuonConfig(
        lr=0.005, mu=0.95, nesterov=True, adjust_lr="rms_norm",
        weight_decay=0.01, flatten=True,
        scalar_lr=1e-3, scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
    )

def vit_dion():
    return DionConfig(
        lr=0.005, rank_fraction=0.25, weight_decay=0.01,
        scalar_lr=1e-3, scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
    )

def vit_dion2():
    return Dion2Config(
        lr=0.005, fraction=0.25, ef_decay=0.95, weight_decay=0.01,
        adjust_lr="spectral_norm", flatten=True,
        scalar_lr=1e-3, scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
    )

def vit_adadion():
    return AdaDionConfig(
        lr=0.005, mu=0.95, rank_fraction_max=0.7, weight_decay=0.1,
        adaptive_rank=True, init_rank_fraction=0.5,
        erank_ema_beta=0.5, rank_scale=1.5,
        rank_min=16, rank_quantize=8, rank_step_up=16, rank_step_down=8,
        scalar_lr=0.005, scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
    )


def run_and_record(all_results, name, base_cfg, opt_cfg, output_dir):
    logger.info(f"\n{'='*70}\n  RUN: {name}\n{'='*70}")
    try:
        summary = train_one_run(base_cfg, opt_cfg, output_dir, name)
        summary["run_name"] = name
        summary["optimizer"] = opt_cfg.name
        summary["model"] = base_cfg.model_name
        summary["seed"] = base_cfg.seed
        summary["opt_config"] = asdict(opt_cfg) if hasattr(opt_cfg, '__dataclass_fields__') else {}
        all_results.append(summary)
        logger.info(f"  RESULT: {name} -> val_acc={summary['best_val_acc']:.2f}%")
    except Exception as e:
        logger.error(f"  FAILED: {name} -> {e}")
        import traceback; traceback.print_exc()
        all_results.append({"run_name": name, "error": str(e)})

    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def run_final_benchmark(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    # PART 1: ResNet-18, 3 seeds, all 5 optimizers
    logger.info("\n" + "#"*70 + "\n  PART 1: ResNet-18 Final Benchmark (100ep, 3 seeds)\n" + "#"*70)

    opt_factories = {
        "adamw": best_adamw,
        "muon": best_muon,
        "dion": best_dion,
        "dion2": best_dion2,
        "adadion": best_adadion,
    }

    for seed in [42, 123, 456]:
        for opt_name, opt_factory in opt_factories.items():
            name = f"resnet18_{opt_name}_seed{seed}"
            run_and_record(all_results, name, make_base(seed=seed), opt_factory(), output_dir)

    # PART 2: ViT-Small LR sweep for AdaDion, then comparison
    logger.info("\n" + "#"*70 + "\n  PART 2: ViT-Small Benchmark\n" + "#"*70)

    # AdaDion LR sweep on ViT
    logger.info("\n  --- ViT-Small AdaDion LR Sweep ---")
    for lr in [0.001, 0.002, 0.005, 0.01, 0.02]:
        cfg = AdaDionConfig(
            lr=lr, mu=0.95, rank_fraction_max=0.7, weight_decay=0.1,
            adaptive_rank=True, init_rank_fraction=0.5,
            erank_ema_beta=0.5, rank_scale=1.5,
            rank_min=16, rank_quantize=8, rank_step_up=16, rank_step_down=8,
            scalar_lr=lr, scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
        )
        name = f"vit_adadion_lr{lr}"
        run_and_record(all_results, name, make_base(model="vit_small"), cfg, output_dir)

    # Find best ViT AdaDion LR
    vit_adadion_runs = [r for r in all_results
                        if r.get("run_name", "").startswith("vit_adadion_lr") and "error" not in r]
    if vit_adadion_runs:
        best_vit = max(vit_adadion_runs, key=lambda r: r.get("best_val_acc", 0))
        best_vit_lr = best_vit["opt_config"]["lr"]
        logger.info(f"  Best ViT AdaDion LR: {best_vit_lr} ({best_vit['best_val_acc']:.2f}%)")

    # ViT comparison at tuned configs
    logger.info("\n  --- ViT-Small Final Comparison ---")
    vit_factories = {
        "adamw": vit_adamw,
        "muon": vit_muon,
        "dion": vit_dion,
        "dion2": vit_dion2,
        "adadion": vit_adadion,
    }

    for opt_name, opt_factory in vit_factories.items():
        name = f"vit_{opt_name}_final"
        run_and_record(all_results, name, make_base(model="vit_small"), opt_factory(), output_dir)

    # FINAL SUMMARY TABLE
    print("\n" + "="*100)
    print(f"{'Run':<40} {'Optimizer':<12} {'Model':<12} {'Val Acc':>10} {'Val Loss':>10} {'Time':>8}")
    print("="*100)

    for r in sorted(all_results, key=lambda x: x.get("run_name", "")):
        if "error" in r:
            print(f"{r.get('run_name','?'):<40} {'':12} {'':12} {'FAILED':>10}")
        else:
            print(f"{r.get('run_name',''):<40} "
                  f"{r.get('optimizer',''):<12} "
                  f"{r.get('model',''):<12} "
                  f"{r.get('best_val_acc',0):>9.2f}% "
                  f"{r.get('final_val_loss',0):>10.4f} "
                  f"{r.get('total_train_time_sec',0):>7.0f}s")
    print("="*100)

    # Compute per-optimizer means for ResNet-18
    from collections import defaultdict
    import numpy as np
    resnet_results = defaultdict(list)
    for r in all_results:
        if r.get("model") == "resnet18" and "error" not in r:
            resnet_results[r["optimizer"]].append(r["best_val_acc"])

    if resnet_results:
        print("\n  ResNet-18 Summary (mean +/- std across seeds):")
        print(f"  {'Optimizer':<15} {'Mean Acc':>10} {'Std':>8}")
        print(f"  {'-'*35}")
        for opt in ["adadion", "dion", "muon", "dion2", "adamw"]:
            if opt in resnet_results:
                vals = resnet_results[opt]
                print(f"  {opt:<15} {np.mean(vals):>9.2f}% {np.std(vals):>7.2f}%")

    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {output_dir}/final_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results/final")
    args = parser.parse_args()
    run_final_benchmark(args.output_dir)
