#!/usr/bin/env python3
"""
Generate all paper figures following PLOT_STYLE_GUIDE.md exactly.
Serif font (CM), no bold, no suptitle, narrow rounded bars, clean layout, PDF output.
"""

import json, os, glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Style setup (from PLOT_STYLE_GUIDE.md) ──
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5, "lines.markersize": 5,
    "axes.titleweight": "normal", "axes.labelweight": "normal", "font.weight": "normal",
})

COLORS = {"adadion": "#7B68AE", "dion": "#4DAF7C", "dion2": "#D95F5F",
          "muon": "#E8943A", "adamw": "#5B8DBE"}
LABELS = {"adadion": "AdaDion V2", "dion": "Dion", "dion2": "Dion2",
          "muon": "Muon", "adamw": "AdamW"}
ORDER = ["adadion", "dion", "dion2", "muon", "adamw"]
BW = 0.48

def fc(c, a=0.45): return mcolors.to_rgba(c, a)
def ec(c, a=0.90): return mcolors.to_rgba(c, a)

def rbar(ax, x, h, w, fcolor, ecolor, bot=0):
    ax.add_patch(FancyBboxPatch((x-w/2, bot), w, h,
        boxstyle="round,pad=0,rounding_size=0.02", facecolor=fcolor,
        edgecolor=ecolor, linewidth=1.0))

def grid(ax):
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.5, color="#cccccc")
    ax.xaxis.grid(False)
    ax.tick_params(direction="out", length=3, width=0.6)

def load_final():
    with open("results/final/final_results.json") as f: return json.load(f)

def load_epochs(run):
    p = f"results/final/{run}/epoch_metrics.json"
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return []

OUT = "results/paper_figures"

# ═══════════════════════════════════════════════════════════════
# Fig 1: ResNet-18 accuracy + loss (bars, 3 seeds)
# ═══════════════════════════════════════════════════════════════
def fig1():
    R = load_final()
    acc, loss = defaultdict(list), defaultdict(list)
    for r in R:
        if r.get("model") != "resnet18": continue
        acc[r["optimizer"]].append(r["best_val_acc"])
        loss[r["optimizer"]].append(r["final_val_loss"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.2))
    opts = [o for o in ORDER if o in acc]
    x = np.arange(len(opts))
    names = [LABELS[o] for o in opts]

    # Accuracy
    m = [np.mean(acc[o]) for o in opts]
    s = [np.std(acc[o]) for o in opts]
    a1.set_ylim(min(m)-1.0, max(m)+0.7)
    for i, o in enumerate(opts):
        rbar(a1, x[i], m[i]-a1.get_ylim()[0], BW, fc(COLORS[o]), ec(COLORS[o]), a1.get_ylim()[0])
    a1.bar(x, m, BW, color="none", edgecolor="none")
    a1.errorbar(x, m, yerr=s, fmt="none", ecolor="#444", capsize=3, capthick=1.0, elinewidth=1.0, zorder=5)
    for i in range(len(opts)):
        a1.text(x[i], m[i]+s[i]+0.06, f"{m[i]:.2f}", ha="center", va="bottom", fontsize=8, color="#333")
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel("Validation accuracy (%)"); grid(a1)

    # Loss
    ml = [np.mean(loss[o]) for o in opts]
    sl = [np.std(loss[o]) for o in opts]
    a2.set_ylim(0, max(ml)+0.04)
    for i, o in enumerate(opts):
        rbar(a2, x[i], ml[i], BW, fc(COLORS[o]), ec(COLORS[o]))
    a2.bar(x, ml, BW, color="none", edgecolor="none")
    a2.errorbar(x, ml, yerr=sl, fmt="none", ecolor="#444", capsize=3, capthick=1.0, elinewidth=1.0, zorder=5)
    for i in range(len(opts)):
        a2.text(x[i], ml[i]+sl[i]+0.003, f"{ml[i]:.4f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel("Validation loss"); grid(a2)

    plt.tight_layout()
    fig.savefig(f"{OUT}/resnet18_bars.pdf", format="pdf"); plt.close()
    print("  resnet18_bars.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 2: Training curves (3 subplots, confidence bands)
# ═══════════════════════════════════════════════════════════════
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    from scipy.ndimage import uniform_filter1d

    for opt in ORDER:
        by_ep = defaultdict(lambda: {"train_loss":[], "val_loss":[], "val_acc":[]})
        for seed in [42,123,456]:
            for em in load_epochs(f"resnet18_{opt}_seed{seed}"):
                by_ep[em["epoch"]]["train_loss"].append(em["train_loss"])
                by_ep[em["epoch"]]["val_loss"].append(em["val_loss"])
                by_ep[em["epoch"]]["val_acc"].append(em["val_acc"])
        if not by_ep: continue
        eps = sorted(by_ep.keys())
        c = COLORS[opt]
        for ai, (k, yl) in enumerate([("train_loss","Loss"),("val_loss","Loss"),("val_acc","Accuracy (%)")]):
            ax = axes[ai]
            raw = np.array([np.mean(by_ep[e][k]) for e in eps])
            std = np.array([np.std(by_ep[e][k]) for e in eps])
            sm = uniform_filter1d(raw, size=5)
            ax.plot(eps, sm, color=c, linewidth=1.5, label=LABELS[opt])
            ax.fill_between(eps, raw-std, raw+std, color=c, alpha=0.10)
            ax.set_xlabel("Epoch"); ax.set_ylabel(yl)

    titles = ["Training loss", "Validation loss", "Validation accuracy"]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i]); grid(ax)
        ax.legend(framealpha=0.7, edgecolor="none", fontsize=7, loc="best")

    plt.tight_layout()
    fig.savefig(f"{OUT}/training_curves.pdf", format="pdf"); plt.close()
    print("  training_curves.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 3: ViT-Small bars
# ═══════════════════════════════════════════════════════════════
def fig3():
    R = load_final()
    vit = {r["optimizer"]: r for r in R if r.get("model")=="vit_small"}
    if not vit: print("  SKIP vit"); return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.2))
    opts = [o for o in ORDER if o in vit]
    x = np.arange(len(opts))
    names = [LABELS[o] for o in opts]

    ac = [vit[o]["best_val_acc"] for o in opts]
    a1.set_ylim(min(ac)-2, max(ac)+1.5)
    for i, o in enumerate(opts):
        rbar(a1, x[i], ac[i]-a1.get_ylim()[0], BW, fc(COLORS[o]), ec(COLORS[o]), a1.get_ylim()[0])
    a1.bar(x, ac, BW, color="none", edgecolor="none")
    for i in range(len(opts)):
        a1.text(x[i], ac[i]+0.2, f"{ac[i]:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel("Validation accuracy (%)"); grid(a1)

    lo = [vit[o]["final_val_loss"] for o in opts]
    a2.set_ylim(0, max(lo)+0.08)
    for i, o in enumerate(opts):
        rbar(a2, x[i], lo[i], BW, fc(COLORS[o]), ec(COLORS[o]))
    a2.bar(x, lo, BW, color="none", edgecolor="none")
    for i in range(len(opts)):
        a2.text(x[i], lo[i]+0.01, f"{lo[i]:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel("Validation loss"); grid(a2)

    plt.tight_layout()
    fig.savefig(f"{OUT}/vit_bars.pdf", format="pdf"); plt.close()
    print("  vit_bars.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 4: Convergence speed
# ═══════════════════════════════════════════════════════════════
def fig4():
    thresholds = [85, 88, 90, 92, 93, 94, 95]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for opt in ORDER:
        ep_to = {t:[] for t in thresholds}
        for seed in [42,123,456]:
            for em in load_epochs(f"resnet18_{opt}_seed{seed}"):
                for t in thresholds:
                    if em["val_acc"] >= t and not ep_to[t]:
                        ep_to[t].append(em["epoch"]); break
                    elif em["val_acc"] >= t and len([x for x in ep_to[t]]) < seed:
                        ep_to[t].append(em["epoch"]); break
        # Rebuild properly
        ep_to2 = {t:[] for t in thresholds}
        for seed in [42,123,456]:
            data = load_epochs(f"resnet18_{opt}_seed{seed}")
            for t in thresholds:
                for em in data:
                    if em["val_acc"] >= t:
                        ep_to2[t].append(em["epoch"]); break

        tv, em_, es = [], [], []
        for t in thresholds:
            if ep_to2[t]:
                tv.append(t); em_.append(np.mean(ep_to2[t])); es.append(np.std(ep_to2[t]))
        if tv:
            ax.errorbar(tv, em_, yerr=es, color=COLORS[opt], label=LABELS[opt],
                        marker="o", linewidth=1.5, capsize=3, markersize=5, markeredgewidth=0)

    ax.set_xlabel("Target validation accuracy (%)")
    ax.set_ylabel("Epochs to reach target")
    ax.legend(framealpha=0.7, edgecolor="none"); grid(ax)
    plt.tight_layout()
    fig.savefig(f"{OUT}/convergence.pdf", format="pdf"); plt.close()
    print("  convergence.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 5: Throughput + training time
# ═══════════════════════════════════════════════════════════════
def fig5():
    R = load_final()
    r18 = {r["optimizer"]: r for r in R if r.get("model")=="resnet18" and r.get("seed")==42}
    if not r18: return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.2))
    opts = [o for o in ORDER if o in r18]
    x = np.arange(len(opts))
    names = [LABELS[o] for o in opts]

    times = [r18[o]["total_train_time_sec"] for o in opts]
    a1.set_ylim(0, max(times)*1.15)
    for i, o in enumerate(opts):
        rbar(a1, x[i], times[i], BW, fc(COLORS[o]), ec(COLORS[o]))
    a1.bar(x, times, BW, color="none", edgecolor="none")
    for i in range(len(opts)):
        a1.text(x[i], times[i]+max(times)*0.02, f"{times[i]:.0f}s", ha="center", va="bottom", fontsize=7.5, color="#333")
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel("Training time (s)"); grid(a1)

    thru = [r18[o].get("avg_throughput_samples_sec",0) for o in opts]
    a2.set_ylim(0, max(thru)*1.15)
    for i, o in enumerate(opts):
        rbar(a2, x[i], thru[i], BW, fc(COLORS[o]), ec(COLORS[o]))
    a2.bar(x, thru, BW, color="none", edgecolor="none")
    for i in range(len(opts)):
        a2.text(x[i], thru[i]+max(thru)*0.02, f"{thru[i]:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel("Throughput (samples/s)"); grid(a2)

    plt.tight_layout()
    fig.savefig(f"{OUT}/throughput.pdf", format="pdf"); plt.close()
    print("  throughput.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 6: Communication overhead
# ═══════════════════════════════════════════════════════════════
def fig6():
    with open("results/distributed/communication_analysis.json") as f: data = json.load(f)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, mn, title in [(a1,"resnet18","ResNet-18"),(a2,"vit_small","ViT-Small")]:
        md = data[mn]
        opts = [o for o in ORDER if o in md["optimizers"]]
        x = np.arange(len(opts))
        names = [LABELS[o] for o in opts]
        mb = [md["optimizers"][o]["megabytes_per_step"] for o in opts]
        comp = [md["optimizers"][o]["compression_ratio"] for o in opts]

        ax.set_ylim(0, max(mb)*1.2)
        for i, o in enumerate(opts):
            rbar(ax, x[i], mb[i], BW, fc(COLORS[o]), ec(COLORS[o]))
        ax.bar(x, mb, BW, color="none", edgecolor="none")
        for i in range(len(opts)):
            label = f"{mb[i]:.0f}" if comp[i] < 1.05 else f"{mb[i]:.0f} ({comp[i]:.1f}x)"
            ax.text(x[i], mb[i]+max(mb)*0.02, label, ha="center", va="bottom", fontsize=7, color="#333")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("MB per step"); ax.set_title(title); grid(ax)

    plt.tight_layout()
    fig.savefig(f"{OUT}/communication.pdf", format="pdf"); plt.close()
    print("  communication.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 7: Rank-performance tradeoff (dual axis)
# ═══════════════════════════════════════════════════════════════
def fig7():
    with open("results/ablation_results.json") as f: abl = json.load(f)
    rf = sorted([(r["opt_config"]["init_rank_fraction"], r["best_val_acc"])
                 for r in abl if r.get("ablation_name","").startswith("adadion_rf") and "error" not in r])
    if not rf: return
    fracs, accs = zip(*rf)
    m, n = 512, 4608
    comps = [m*n/(m*max(1,int(f*min(m,n)))+n*max(1,int(f*min(m,n)))) for f in fracs]

    fig, a1 = plt.subplots(figsize=(5, 3.5))
    c1, c2 = "#7B68AE", "#4DAF7C"
    a1.plot(fracs, accs, color=c1, marker="s", linewidth=1.8, markersize=7, markeredgewidth=0, label="Accuracy")
    a1.set_xlabel("Init rank fraction")
    a1.set_ylabel("Validation accuracy (%)", color=c1)
    a1.tick_params(axis="y", labelcolor=c1)
    a1.set_ylim(min(accs)-0.15, max(accs)+0.2)
    bi = np.argmax(accs)
    a1.annotate(f"{accs[bi]:.2f}", (fracs[bi], accs[bi]),
                textcoords="offset points", xytext=(12,8), fontsize=8, color=c1)

    a2 = a1.twinx()
    a2.spines["right"].set_visible(True)
    a2.plot(fracs, comps, color=c2, marker="o", linewidth=1.8, markersize=6,
            markeredgewidth=0, linestyle="--", label="Compression")
    a2.set_ylabel("Compression ratio", color=c2)
    a2.tick_params(axis="y", labelcolor=c2)

    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1+h2, l1+l2, loc="lower left", framealpha=0.7, edgecolor="none")
    grid(a1)
    plt.tight_layout()
    fig.savefig(f"{OUT}/rank_tradeoff.pdf", format="pdf"); plt.close()
    print("  rank_tradeoff.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 8: Compression ratio grouped bars
# ═══════════════════════════════════════════════════════════════
def fig8():
    with open("results/distributed/communication_analysis.json") as f: data = json.load(f)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(ORDER)); w = 0.32
    r18 = [data["resnet18"]["optimizers"][o]["compression_ratio"] for o in ORDER]
    vit = [data["vit_small"]["optimizers"][o]["compression_ratio"] for o in ORDER]

    c1, c2 = "#5B8DBE", "#E8943A"
    for i in range(len(ORDER)):
        rbar(ax, x[i]-w/2, r18[i], w*0.85, fc(c1), ec(c1))
        rbar(ax, x[i]+w/2, vit[i], w*0.85, fc(c2), ec(c2))
    ax.bar(x-w/2, r18, w*0.85, color="none", label="ResNet-18")
    ax.bar(x+w/2, vit, w*0.85, color="none", label="ViT-Small")
    for i in range(len(ORDER)):
        for v, xp in [(r18[i],x[i]-w/2),(vit[i],x[i]+w/2)]:
            if v > 1.05:
                ax.text(xp, v+0.06, f"{v:.1f}x", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[o] for o in ORDER])
    ax.set_ylabel("Compression ratio"); ax.axhline(1, color="#999", ls=":", lw=0.8)
    ax.legend(framealpha=0.7, edgecolor="none"); grid(ax)
    plt.tight_layout()
    fig.savefig(f"{OUT}/compression_ratio.pdf", format="pdf"); plt.close()
    print("  compression_ratio.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 9: ViT LR sweep
# ═══════════════════════════════════════════════════════════════
def fig9():
    pts = [(0.001,90.41),(0.002,91.75),(0.005,92.25),(0.01,91.15),(0.02,87.63)]
    lrs, accs = zip(*pts)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(len(lrs)), accs, color=COLORS["adadion"], marker="s", linewidth=1.8,
            markersize=8, markeredgewidth=0)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([str(lr) for lr in lrs])
    ax.set_xlabel("Learning rate"); ax.set_ylabel("Validation accuracy (%)")
    bi = np.argmax(accs)
    ax.annotate(f"{accs[bi]:.2f}", (bi, accs[bi]),
                textcoords="offset points", xytext=(12,8), fontsize=8, color=COLORS["adadion"])
    grid(ax)
    plt.tight_layout()
    fig.savefig(f"{OUT}/vit_lr_sweep.pdf", format="pdf"); plt.close()
    print("  vit_lr_sweep.pdf")

# ═══════════════════════════════════════════════════════════════
# Fig 10: Wide ResNet scaling
# ═══════════════════════════════════════════════════════════════
def fig10():
    sfiles = sorted(glob.glob("results/wide_resnet/**/summary.json", recursive=True))
    if not sfiles: print("  SKIP wrn_scaling"); return
    runs = {}
    for s in sfiles:
        with open(s) as f: r = json.load(f)
        runs[(r["model"], r["optimizer"])] = r

    widths = ["wrn-28-2", "wrn-28-4", "wrn-28-10"]
    params = {"wrn-28-2": "1.5M", "wrn-28-4": "5.9M", "wrn-28-10": "36.5M"}
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    opts_here = ["adadion", "dion", "muon", "adamw"]
    for opt in opts_here:
        vals = []
        for w in widths:
            key = (w, opt)
            if key in runs:
                vals.append(runs[key]["best_val_acc"])
            else:
                vals.append(None)
        valid_x = [i for i,v in enumerate(vals) if v is not None]
        valid_y = [v for v in vals if v is not None]
        ax.plot(valid_x, valid_y, color=COLORS[opt], marker="o", linewidth=1.8,
                markersize=7, markeredgewidth=0, label=LABELS[opt])

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([f"{w}\n({params[w]})" for w in widths], fontsize=8)
    ax.set_xlabel("Model (parameters)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend(framealpha=0.7, edgecolor="none"); grid(ax)
    plt.tight_layout()
    fig.savefig(f"{OUT}/wrn_scaling.pdf", format="pdf"); plt.close()
    print("  wrn_scaling.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Generating paper figures...")
    fig1()
    try: fig2()
    except ImportError: print("  SKIP fig2 (needs scipy)")
    fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); fig9(); fig10()
    print(f"\nDone. All PDFs in {OUT}/")
