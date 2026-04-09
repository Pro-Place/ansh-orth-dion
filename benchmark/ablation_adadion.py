#!/usr/bin/env python3
"""
Extensive AdaDion V2 ablation study on CIFAR-10 ResNet-18.

Ablation axes:
  1. Learning rate sweep: 0.002, 0.005, 0.01, 0.02, 0.04
  2. adaptive_rank: True vs False (False = equivalent to base Dion)
  3. Gradient clipping: 1.0 vs 0 (no clipping)
  4. Weight decay: 0.01 vs 0.1
  5. Baseline comparisons: AdamW, Muon, Dion at their best configs

Also compares against a "Dion-equivalent" mode (adaptive_rank=False)
to isolate whether the issue is the base Dion algorithm or the
adaptive rank mechanism.

All runs: 100 epochs, seed 42, ResNet-18, no AMP.
"""

import argparse
import copy
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import from benchmark
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar10_benchmark import train_one_run
from configs import BaseConfig, AdaDionConfig, AdamWConfig, MuonConfig, DionConfig, Dion2Config


def make_base(epochs=100, grad_clip=1.0):
    return BaseConfig(
        epochs=epochs,
        warmup_epochs=5,
        batch_size=128,
        num_workers=4,
        log_interval=100,
        save_checkpoints=False,
        model_name="resnet18",
        device="cuda",
        mixed_precision=False,  # No AMP for spectral optimizers
        seed=42,
        gradient_clip=grad_clip,
        label_smoothing=0.1,
    )


def run_ablation(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    def run_and_record(name, base_cfg, opt_cfg):
        logger.info(f"\n{'='*70}\n  ABLATION: {name}\n{'='*70}")
        try:
            summary = train_one_run(base_cfg, opt_cfg, output_dir, name)
            summary["ablation_name"] = name
            summary["opt_config"] = asdict(opt_cfg) if hasattr(opt_cfg, '__dataclass_fields__') else {}
            all_results.append(summary)
            logger.info(f"  RESULT: {name} -> {summary['best_val_acc']:.2f}%")
        except Exception as e:
            logger.error(f"  FAILED: {name} -> {e}")
            import traceback; traceback.print_exc()
            all_results.append({"ablation_name": name, "error": str(e)})

        # Save intermediate
        with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # PHASE 1: Baselines (AdamW, Muon, Dion at known-good configs)
    logger.info("\n" + "="*70 + "\n  PHASE 1: BASELINES\n" + "="*70)

    # AdamW baseline
    run_and_record("baseline_adamw", make_base(), AdamWConfig())

    # Muon baseline
    run_and_record("baseline_muon", make_base(), MuonConfig(
        lr=0.02, mu=0.95, nesterov=True, adjust_lr="rms_norm",
        weight_decay=0.01, scalar_lr=1e-3, flatten=True,
    ))

    # Dion baseline
    run_and_record("baseline_dion", make_base(), DionConfig(
        lr=0.02, rank_fraction=0.25, weight_decay=0.01,
        scalar_lr=1e-3,
    ))

    # PHASE 2: AdaDion LR sweep (the most critical axis)
    logger.info("\n" + "="*70 + "\n  PHASE 2: AdaDion LR SWEEP\n" + "="*70)

    for lr in [0.002, 0.005, 0.01, 0.02, 0.04]:
        cfg = AdaDionConfig(
            lr=lr,
            mu=0.95,
            rank_fraction_max=0.7,
            weight_decay=0.1,
            adaptive_rank=True,
            init_rank_fraction=0.5,
            erank_ema_beta=0.5,
            rank_scale=1.5,
            rank_min=16,
            rank_quantize=8,
            rank_step_up=16,
            rank_step_down=8,
            scalar_lr=lr,  # Match scalar LR to matrix LR
            scalar_betas=(0.95, 0.95),
            scalar_weight_decay=0.1,
            scalar_eps=1e-8,
        )
        run_and_record(f"adadion_lr{lr}", make_base(), cfg)

    # PHASE 3: adaptive_rank ON vs OFF (isolate adaptive mechanism)
    logger.info("\n" + "="*70 + "\n  PHASE 3: ADAPTIVE RANK ABLATION\n" + "="*70)

    # Find best LR from phase 2
    phase2 = [r for r in all_results if r.get("ablation_name", "").startswith("adadion_lr") and "error" not in r]
    if phase2:
        best_lr_run = max(phase2, key=lambda r: r.get("best_val_acc", 0))
        best_lr = best_lr_run["opt_config"]["lr"]
        logger.info(f"  Best LR from sweep: {best_lr} ({best_lr_run['best_val_acc']:.2f}%)")
    else:
        best_lr = 0.02

    # AdaDion with adaptive_rank=FALSE (should match Dion behavior)
    run_and_record("adadion_no_adaptive", make_base(), AdaDionConfig(
        lr=best_lr, adaptive_rank=False,
        init_rank_fraction=0.5, rank_fraction_max=0.7,
        weight_decay=0.1, scalar_lr=best_lr,
        scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
    ))

    # AdaDion with adaptive_rank=TRUE at best LR
    run_and_record("adadion_adaptive", make_base(), AdaDionConfig(
        lr=best_lr, adaptive_rank=True,
        init_rank_fraction=0.5, rank_fraction_max=0.7,
        erank_ema_beta=0.5, rank_scale=1.5,
        rank_min=16, rank_quantize=8,
        rank_step_up=16, rank_step_down=8,
        weight_decay=0.1, scalar_lr=best_lr,
        scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
    ))

    # PHASE 4: Gradient clipping ablation
    logger.info("\n" + "="*70 + "\n  PHASE 4: GRADIENT CLIPPING ABLATION\n" + "="*70)

    # No gradient clipping
    run_and_record("adadion_no_clip", make_base(grad_clip=0.0), AdaDionConfig(
        lr=best_lr, adaptive_rank=True,
        init_rank_fraction=0.5, rank_fraction_max=0.7,
        erank_ema_beta=0.5, rank_scale=1.5,
        rank_min=16, rank_quantize=8,
        weight_decay=0.1, scalar_lr=best_lr,
        scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
    ))

    # Higher clip threshold
    run_and_record("adadion_clip5", make_base(grad_clip=5.0), AdaDionConfig(
        lr=best_lr, adaptive_rank=True,
        init_rank_fraction=0.5, rank_fraction_max=0.7,
        erank_ema_beta=0.5, rank_scale=1.5,
        rank_min=16, rank_quantize=8,
        weight_decay=0.1, scalar_lr=best_lr,
        scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
    ))

    # PHASE 5: Weight decay ablation
    logger.info("\n" + "="*70 + "\n  PHASE 5: WEIGHT DECAY ABLATION\n" + "="*70)

    for wd in [0.0, 0.01, 0.05]:
        run_and_record(f"adadion_wd{wd}", make_base(), AdaDionConfig(
            lr=best_lr, adaptive_rank=True,
            init_rank_fraction=0.5, rank_fraction_max=0.7,
            erank_ema_beta=0.5, rank_scale=1.5,
            rank_min=16, rank_quantize=8,
            weight_decay=wd, scalar_lr=best_lr,
            scalar_betas=(0.95, 0.95), scalar_weight_decay=wd,
        ))

    # PHASE 6: Rank fraction ablation
    logger.info("\n" + "="*70 + "\n  PHASE 6: RANK FRACTION ABLATION\n" + "="*70)

    for rf in [0.125, 0.25, 0.5, 0.75, 1.0]:
        run_and_record(f"adadion_rf{rf}", make_base(), AdaDionConfig(
            lr=best_lr, adaptive_rank=True,
            init_rank_fraction=rf, rank_fraction_max=max(rf, 0.7),
            erank_ema_beta=0.5, rank_scale=1.5,
            rank_min=16, rank_quantize=8,
            weight_decay=0.1, scalar_lr=best_lr,
            scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
        ))

    # PHASE 7: Scalar LR decoupling
    logger.info("\n" + "="*70 + "\n  PHASE 7: SCALAR LR DECOUPLING\n" + "="*70)

    for slr in [1e-3, 3e-3, 0.01]:
        run_and_record(f"adadion_slr{slr}", make_base(), AdaDionConfig(
            lr=best_lr, adaptive_rank=True,
            init_rank_fraction=0.5, rank_fraction_max=0.7,
            erank_ema_beta=0.5, rank_scale=1.5,
            rank_min=16, rank_quantize=8,
            weight_decay=0.1, scalar_lr=slr,
            scalar_betas=(0.9, 0.999), scalar_weight_decay=0.05,
        ))

    # FINAL SUMMARY
    print("\n" + "="*90)
    print(f"{'Ablation':<35} {'Best Val Acc':>12} {'Epoch':>6} {'Time':>8}")
    print("="*90)
    for r in all_results:
        name = r.get("ablation_name", "?")
        if "error" in r:
            print(f"{name:<35} {'FAILED':>12}")
        else:
            print(f"{name:<35} {r.get('best_val_acc',0):>11.2f}% "
                  f"{r.get('best_val_epoch',0):>6} "
                  f"{r.get('total_train_time_sec',0):>7.0f}s")
    print("="*90)

    # Save final
    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {output_dir}/ablation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results/ablation")
    args = parser.parse_args()
    run_ablation(args.output_dir)
