#!/usr/bin/env python3
"""
Generate publication-quality PDF figures for the AdaDion V2 paper.

Style: Poppins font (unbolded), thin bars with rounded corners and shading,
smooth curves, no overlapping text, consistent aspect ratios, minimal clutter.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Try loading Poppins, fall back to sans-serif ──
try:
    poppins_paths = fm.findSystemFonts(fontpaths=None)
    poppins = [p for p in poppins_paths if "Poppins" in p and "Regular" in p]
    if poppins:
        fm.fontManager.addfont(poppins[0])
        plt.rcParams["font.family"] = "Poppins"
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
except Exception:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "font.weight": "normal",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
})

COLORS = {
    "adadion": "#7B68AE",
    "dion": "#4DAF7C",
    "dion2": "#D95F5F",
    "muon": "#E8943A",
    "adamw": "#5B8DBE",
}
EDGE_COLORS = {k: matplotlib.colors.to_rgba(v, 0.85) for k, v in COLORS.items()}
FILL_COLORS = {k: matplotlib.colors.to_rgba(v, 0.55) for k, v in COLORS.items()}
LABELS = {"adamw": "AdamW", "muon": "Muon", "dion": "Dion", "dion2": "Dion2", "adadion": "AdaDion V2"}
OPT_ORDER = ["adadion", "dion", "dion2", "muon", "adamw"]
BAR_WIDTH = 0.48


def load_final_results(path="results/final/final_results.json"):
    with open(path) as f:
        return json.load(f)


def load_epoch_data(results_dir, run_name):
    p = os.path.join(results_dir, run_name, "epoch_metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []


def rounded_bar(ax, x, height, width, color, edge_color, bottom=0):
    """Draw a bar with slightly rounded top corners and shaded fill."""
    box = FancyBboxPatch(
        (x - width / 2, bottom), width, height,
        boxstyle="round,pad=0,rounding_size=0.03",
        facecolor=color, edgecolor=edge_color, linewidth=1.2,
    )
    ax.add_patch(box)
    return box


def draw_bars(ax, names, values, colors, edge_colors, errs=None, ylabel="", val_fmt="{:.2f}%"):
    """Draw a bar chart with rounded, shaded bars."""
    x = np.arange(len(names))
    for i, (n, v) in enumerate(zip(names, values)):
        rounded_bar(ax, x[i], v - ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else v,
                    BAR_WIDTH, colors[i], edge_colors[i],
                    bottom=ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0)

    # Manual bars for ylim calculation first
    ax.bar(x, values, width=BAR_WIDTH, color="none", edgecolor="none")
    if errs:
        ax.errorbar(x, values, yerr=errs, fmt="none", ecolor="#333333",
                    capsize=4, capthick=1.2, elinewidth=1.2, zorder=5)

    for i, v in enumerate(values):
        label = val_fmt.format(v)
        ax.text(x[i], v + (max(values) - min(values)) * 0.03, label,
                ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(ylabel)


# ── Figure 1: ResNet-18 accuracy + loss bars ──
def fig_resnet_bars(out):
    results = load_final_results()
    acc_data = defaultdict(list)
    loss_data = defaultdict(list)
    for r in results:
        if r.get("model") != "resnet18" or "error" in r:
            continue
        acc_data[r["optimizer"]].append(r["best_val_acc"])
        loss_data[r["optimizer"]].append(r["final_val_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    opts = [o for o in OPT_ORDER if o in acc_data]
    names = [LABELS[o] for o in opts]
    colors = [FILL_COLORS[o] for o in opts]
    edges = [EDGE_COLORS[o] for o in opts]

    means = [np.mean(acc_data[o]) for o in opts]
    stds = [np.std(acc_data[o]) for o in opts]
    ax1.set_ylim(min(means) - 1.2, max(means) + 0.8)
    x = np.arange(len(names))
    for i in range(len(opts)):
        rounded_bar(ax1, x[i], means[i] - ax1.get_ylim()[0], BAR_WIDTH,
                    colors[i], edges[i], bottom=ax1.get_ylim()[0])
    ax1.bar(x, means, width=BAR_WIDTH, color="none", edgecolor="none")
    ax1.errorbar(x, means, yerr=stds, fmt="none", ecolor="#333333",
                 capsize=4, capthick=1.2, elinewidth=1.2, zorder=5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(x[i], m + s + 0.08, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Validation Accuracy")

    means_l = [np.mean(loss_data[o]) for o in opts]
    stds_l = [np.std(loss_data[o]) for o in opts]
    ax2.set_ylim(0, max(means_l) + 0.05)
    for i in range(len(opts)):
        rounded_bar(ax2, x[i], means_l[i], BAR_WIDTH, colors[i], edges[i])
    ax2.bar(x, means_l, width=BAR_WIDTH, color="none", edgecolor="none")
    ax2.errorbar(x, means_l, yerr=stds_l, fmt="none", ecolor="#333333",
                 capsize=4, capthick=1.2, elinewidth=1.2, zorder=5)
    for i, m in enumerate(means_l):
        ax2.text(x[i], m + 0.005, f"{m:.4f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss")

    fig.suptitle("ResNet-18 CIFAR-10 (100 epochs, 3 seeds)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "resnet18_bars.pdf"), format="pdf")
    plt.close()
    print("  resnet18_bars.pdf")


# ── Figure 2: Training curves ──
def fig_training_curves(out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    for opt in OPT_ORDER:
        epoch_by_ep = defaultdict(lambda: {"train_loss": [], "val_loss": [], "val_acc": []})
        for seed in [42, 123, 456]:
            data = load_epoch_data("results/final", f"resnet18_{opt}_seed{seed}")
            for em in data:
                ep = em["epoch"]
                epoch_by_ep[ep]["train_loss"].append(em["train_loss"])
                epoch_by_ep[ep]["val_loss"].append(em["val_loss"])
                epoch_by_ep[ep]["val_acc"].append(em["val_acc"])
        if not epoch_by_ep:
            continue

        epochs = sorted(epoch_by_ep.keys())
        color = COLORS[opt]

        for ax_i, (metric, title, ylabel) in enumerate([
            ("train_loss", "Training Loss", "Loss"),
            ("val_loss", "Validation Loss", "Loss"),
            ("val_acc", "Validation Accuracy", "Accuracy (%)"),
        ]):
            ax = axes[ax_i]
            raw_means = np.array([np.mean(epoch_by_ep[e][metric]) for e in epochs])
            stds = np.array([np.std(epoch_by_ep[e][metric]) for e in epochs])
            # Light smoothing
            from scipy.ndimage import uniform_filter1d
            means = uniform_filter1d(raw_means, size=3)
            ax.plot(epochs, means, color=color, label=LABELS[opt], linewidth=1.6)
            ax.fill_between(epochs, raw_means - stds, raw_means + stds, color=color, alpha=0.08)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)

    for ax in axes:
        ax.legend(fontsize=8, loc="best", framealpha=0.7, edgecolor="none")

    plt.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.pdf"), format="pdf")
    plt.close()
    print("  training_curves.pdf")


# ── Figure 3: ViT-Small bars ──
def fig_vit_bars(out):
    results = load_final_results()
    vit = {r["optimizer"]: r for r in results if r.get("model") == "vit_small" and "error" not in r}
    if not vit:
        print("  SKIP vit_bars (no data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    opts = [o for o in OPT_ORDER if o in vit]
    names = [LABELS[o] for o in opts]
    colors = [FILL_COLORS[o] for o in opts]
    edges = [EDGE_COLORS[o] for o in opts]
    x = np.arange(len(names))

    accs = [vit[o]["best_val_acc"] for o in opts]
    ax1.set_ylim(min(accs) - 2, max(accs) + 1.5)
    for i in range(len(opts)):
        rounded_bar(ax1, x[i], accs[i] - ax1.get_ylim()[0], BAR_WIDTH,
                    colors[i], edges[i], bottom=ax1.get_ylim()[0])
    ax1.bar(x, accs, width=BAR_WIDTH, color="none", edgecolor="none")
    for i, a in enumerate(accs):
        ax1.text(x[i], a + 0.2, f"{a:.1f}", ha="center", va="bottom", fontsize=8.5, color="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Validation Accuracy")

    losses = [vit[o]["final_val_loss"] for o in opts]
    ax2.set_ylim(0, max(losses) + 0.08)
    for i in range(len(opts)):
        rounded_bar(ax2, x[i], losses[i], BAR_WIDTH, colors[i], edges[i])
    ax2.bar(x, losses, width=BAR_WIDTH, color="none", edgecolor="none")
    for i, l in enumerate(losses):
        ax2.text(x[i], l + 0.01, f"{l:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss")

    fig.suptitle("ViT-Small CIFAR-10 (100 epochs)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "vit_bars.pdf"), format="pdf")
    plt.close()
    print("  vit_bars.pdf")


# ── Figure 4: Convergence speed ──
def fig_convergence(out):
    thresholds = [85, 88, 90, 92, 93, 94, 95]
    fig, ax = plt.subplots(figsize=(6, 4))

    for opt in OPT_ORDER:
        epochs_to = {t: [] for t in thresholds}
        for seed in [42, 123, 456]:
            data = load_epoch_data("results/final", f"resnet18_{opt}_seed{seed}")
            for t in thresholds:
                for em in data:
                    if em["val_acc"] >= t:
                        epochs_to[t].append(em["epoch"])
                        break
        t_vals, e_means, e_stds = [], [], []
        for t in thresholds:
            if epochs_to[t]:
                t_vals.append(t)
                e_means.append(np.mean(epochs_to[t]))
                e_stds.append(np.std(epochs_to[t]))
        if t_vals:
            ax.errorbar(t_vals, e_means, yerr=e_stds, color=COLORS[opt],
                        label=LABELS[opt], marker="o", linewidth=1.6, capsize=3,
                        markersize=5, markeredgewidth=0)

    ax.set_xlabel("Target Validation Accuracy (%)")
    ax.set_ylabel("Epochs to Reach Target")
    ax.legend(framealpha=0.7, edgecolor="none")
    plt.tight_layout()
    fig.savefig(os.path.join(out, "convergence.pdf"), format="pdf")
    plt.close()
    print("  convergence.pdf")


# ── Figure 5: Throughput ──
def fig_throughput(out):
    results = load_final_results()
    r18_s42 = {r["optimizer"]: r for r in results
               if r.get("model") == "resnet18" and r.get("seed") == 42 and "error" not in r}
    if not r18_s42:
        print("  SKIP throughput (no data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    opts = [o for o in OPT_ORDER if o in r18_s42]
    names = [LABELS[o] for o in opts]
    colors = [FILL_COLORS[o] for o in opts]
    edges = [EDGE_COLORS[o] for o in opts]
    x = np.arange(len(names))

    times = [r18_s42[o]["total_train_time_sec"] for o in opts]
    ax1.set_ylim(0, max(times) * 1.15)
    for i in range(len(opts)):
        rounded_bar(ax1, x[i], times[i], BAR_WIDTH, colors[i], edges[i])
    ax1.bar(x, times, width=BAR_WIDTH, color="none", edgecolor="none")
    for i, t in enumerate(times):
        ax1.text(x[i], t + max(times)*0.02, f"{t:.0f}s", ha="center", va="bottom", fontsize=8, color="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Training Time (s)")
    ax1.set_title("Total Training Time")

    thru = [r18_s42[o].get("avg_throughput_samples_sec", 0) for o in opts]
    ax2.set_ylim(0, max(thru) * 1.15)
    for i in range(len(opts)):
        rounded_bar(ax2, x[i], thru[i], BAR_WIDTH, colors[i], edges[i])
    ax2.bar(x, thru, width=BAR_WIDTH, color="none", edgecolor="none")
    for i, t in enumerate(thru):
        ax2.text(x[i], t + max(thru)*0.02, f"{t:.0f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Throughput (samples/s)")
    ax2.set_title("Average Throughput")

    fig.suptitle("ResNet-18 Training Cost (seed 42)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "throughput.pdf"), format="pdf")
    plt.close()
    print("  throughput.pdf")


# ── Figure 6: Communication overhead ──
def fig_communication(out):
    with open("results/distributed/communication_analysis.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, model_name, title in [(ax1, "resnet18", "ResNet-18"), (ax2, "vit_small", "ViT-Small")]:
        md = data[model_name]
        opts = [o for o in OPT_ORDER if o in md["optimizers"]]
        names = [LABELS[o] for o in opts]
        colors = [FILL_COLORS[o] for o in opts]
        edges = [EDGE_COLORS[o] for o in opts]
        x = np.arange(len(names))
        mb = [md["optimizers"][o]["megabytes_per_step"] for o in opts]
        comp = [md["optimizers"][o]["compression_ratio"] for o in opts]

        ax.set_ylim(0, max(mb) * 1.2)
        for i in range(len(opts)):
            rounded_bar(ax, x[i], mb[i], BAR_WIDTH, colors[i], edges[i])
        ax.bar(x, mb, width=BAR_WIDTH, color="none", edgecolor="none")
        for i, (m, c) in enumerate(zip(mb, comp)):
            label = f"{m:.0f} MB" if c < 1.05 else f"{m:.0f} MB ({c:.1f}x)"
            ax.text(x[i], m + max(mb)*0.02, label, ha="center", va="bottom", fontsize=7.5, color="#333")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("MB per step")
        ax.set_title(title)

    fig.suptitle("Communication Cost Per Step (DDP All-Reduce)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "communication.pdf"), format="pdf")
    plt.close()
    print("  communication.pdf")


# ── Figure 7: Rank-performance tradeoff ──
def fig_rank_tradeoff(out):
    with open("results/ablation_results.json") as f:
        ablation = json.load(f)

    rf_runs = [(r["opt_config"]["init_rank_fraction"], r["best_val_acc"])
               for r in ablation if r.get("ablation_name", "").startswith("adadion_rf") and "error" not in r]
    rf_runs.sort()
    if not rf_runs:
        return

    fractions, accs = zip(*rf_runs)
    m, n = 512, 4608
    compressions = [m * n / (m * max(1, int(rf * min(m, n))) + n * max(1, int(rf * min(m, n))))
                    for rf in fractions]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    c1, c2 = "#7B68AE", "#4DAF7C"

    ax1.plot(fractions, accs, color=c1, marker="s", linewidth=1.8, markersize=7,
             markeredgewidth=0, label="Val Accuracy")
    ax1.set_xlabel("Init Rank Fraction")
    ax1.set_ylabel("Validation Accuracy (%)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(min(accs) - 0.15, max(accs) + 0.15)
    best_i = np.argmax(accs)
    ax1.annotate(f"{accs[best_i]:.2f}%", (fractions[best_i], accs[best_i]),
                 textcoords="offset points", xytext=(12, 8), ha="left", fontsize=9, color=c1)

    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(fractions, compressions, color=c2, marker="o", linewidth=1.8, markersize=6,
             markeredgewidth=0, linestyle="--", label="Compression")
    ax2.set_ylabel("Compression Ratio", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.7, edgecolor="none")

    plt.tight_layout()
    fig.savefig(os.path.join(out, "rank_tradeoff.pdf"), format="pdf")
    plt.close()
    print("  rank_tradeoff.pdf")


# ── Figure 8: Compression ratio comparison ──
def fig_compression_ratio(out):
    with open("results/distributed/communication_analysis.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(OPT_ORDER))
    w = 0.35

    r18 = [data["resnet18"]["optimizers"][o]["compression_ratio"] for o in OPT_ORDER]
    vit = [data["vit_small"]["optimizers"][o]["compression_ratio"] for o in OPT_ORDER]

    for i in range(len(OPT_ORDER)):
        rounded_bar(ax, x[i] - w/2, r18[i], w * 0.85,
                    matplotlib.colors.to_rgba("#5B8DBE", 0.55),
                    matplotlib.colors.to_rgba("#5B8DBE", 0.85))
        rounded_bar(ax, x[i] + w/2, vit[i], w * 0.85,
                    matplotlib.colors.to_rgba("#E8943A", 0.55),
                    matplotlib.colors.to_rgba("#E8943A", 0.85))
    ax.bar(x - w/2, r18, w * 0.85, color="none", label="ResNet-18")
    ax.bar(x + w/2, vit, w * 0.85, color="none", label="ViT-Small")

    for i in range(len(OPT_ORDER)):
        for val, xpos in [(r18[i], x[i] - w/2), (vit[i], x[i] + w/2)]:
            if val > 1.05:
                ax.text(xpos, val + 0.08, f"{val:.1f}x", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[o] for o in OPT_ORDER])
    ax.set_ylabel("Compression Ratio")
    ax.axhline(y=1, color="#999", linestyle=":", linewidth=0.8)
    ax.legend(framealpha=0.7, edgecolor="none")
    plt.tight_layout()
    fig.savefig(os.path.join(out, "compression_ratio.pdf"), format="pdf")
    plt.close()
    print("  compression_ratio.pdf")


# ── Figure 9: ViT LR sweep ──
def fig_vit_lr_sweep(out):
    with open("results/ablation_results.json") as f:
        ablation = json.load(f)

    # ViT LR sweep from earlier runs
    vit_lr = [(0.001, 90.41), (0.002, 91.75), (0.005, 92.25), (0.01, 91.15), (0.02, 87.63)]
    lrs, accs = zip(*vit_lr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(lrs)), accs, color=COLORS["adadion"], marker="s", linewidth=1.8,
            markersize=8, markeredgewidth=0)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr}" for lr in lrs])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Validation Accuracy (%)")

    best_i = np.argmax(accs)
    ax.annotate(f"{accs[best_i]:.2f}%", (best_i, accs[best_i]),
                textcoords="offset points", xytext=(12, 8), ha="left", fontsize=9,
                color=COLORS["adadion"])

    plt.tight_layout()
    fig.savefig(os.path.join(out, "vit_lr_sweep.pdf"), format="pdf")
    plt.close()
    print("  vit_lr_sweep.pdf")


if __name__ == "__main__":
    out = "results/paper_figures"
    os.makedirs(out, exist_ok=True)

    print("Generating paper figures (PDF)...")
    fig_resnet_bars(out)
    try:
        fig_training_curves(out)
    except ImportError:
        print("  SKIP training_curves (needs scipy)")
    fig_vit_bars(out)
    fig_convergence(out)
    fig_throughput(out)
    fig_communication(out)
    fig_rank_tradeoff(out)
    fig_compression_ratio(out)
    fig_vit_lr_sweep(out)
    print(f"\nAll figures saved to {out}/")
