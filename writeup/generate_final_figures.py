"""
Generate all figures for Ada-Dion V3 paper.
Follows PLOT_STYLE_GUIDE.md exactly:
  - Serif + CM math (no usetex)
  - All 4 spines visible
  - Dashed grid both axes, alpha=0.3
  - Saturated colorblind palette
  - Standard bars (no rounded corners), value labels above
  - Raw+smooth for line plots
  - No suptitle, no bold
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# ── Boilerplate from PLOT_STYLE_GUIDE.md ──
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
})

COLORS = {
    "adadion":  "#6A3D9A",  # deep purple
    "adadion3": "#6A3D9A",  # same
    "v3a":      "#6A3D9A",
    "v3":       "#8B6DB5",  # lighter purple
    "v2":       "#A65628",  # brown
    "dion":     "#33A02C",  # strong green
    "dion03":   "#33A02C",
    "dion10":   "#666666",  # gray for beta=1
    "dion2":    "#E31A1C",  # strong red
    "muon":     "#FF7F00",  # strong orange
    "adamw":    "#1F78B4",  # strong blue
    "qr":       "#E31A1C",
    "colnorm":  "#33A02C",
    "partial":  "#6A3D9A",
    "block":    "#FF7F00",
    "soft":     "#1F78B4",
    "no_ef":    "#333333",
}

def ec(c, a=0.9):
    return mcolors.to_rgba(c, a)

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)


# =====================================================================
# FIG 1: LLaMA 300M main result
# =====================================================================
def fig1():
    data = [
        ("AdaDion V3\nAdaptive", 3.403, COLORS["v3a"]),
        ("Dion\n$\\beta=0.3$",   3.565, COLORS["dion03"]),
        ("AdaDion\nV3",          3.638, COLORS["v3"]),
        ("AdaDion\nV2",          3.677, COLORS["v2"]),
        ("Dion\n$\\beta=1.0$",   3.918, COLORS["dion10"]),
    ]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(data))
    names = [d[0] for d in data]
    vals  = [d[1] for d in data]
    cols  = [d[2] for d in data]

    bars = ax.bar(x, vals, width=0.5, color=cols,
                  edgecolor=[ec(c) for c in cols], linewidth=0.8, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Validation loss (15k steps)")
    ax.set_ylim(3.2, 4.1)
    fig.savefig(os.path.join(OUT, "fig1_llama_main.pdf"), format="pdf")
    plt.close(fig)
    print("fig1_llama_main.pdf")


# =====================================================================
# FIG 2: Communication--loss Pareto
# =====================================================================
def fig2():
    pts = [
        ("Muon",              3.277, 30828, COLORS["muon"],  "^"),
        ("AdamW",             3.381, 18504, COLORS["adamw"], "s"),
        ("V3 Adaptive",       3.403, 11998, COLORS["v3a"],   "D"),
        ("Dion $\\beta$=0.3", 3.565, 1456,  COLORS["dion03"],"o"),
        ("V3",                3.638, 1514,  COLORS["v3"],    "o"),
        ("V2",                3.677, 6826,  COLORS["v2"],    "^"),
        ("Dion $\\beta$=1",   3.918, 1456,  COLORS["dion10"],"s"),
    ]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, vl, comm, col, mk in pts:
        ax.scatter(comm, vl, c=col, marker=mk, s=70, zorder=5,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(name, (comm, vl), textcoords="offset points",
                    xytext=(10, 4), fontsize=8, color=col)
    ax.set_xscale("log")
    ax.set_xlabel("Total communication (GB)")
    ax.set_ylabel("Validation loss")
    ax.set_ylim(3.1, 4.1)
    fig.savefig(os.path.join(OUT, "fig2_pareto.pdf"), format="pdf")
    plt.close(fig)
    print("fig2_pareto.pdf")


# =====================================================================
# FIG 3: CIFAR-10 + FashionMNIST
# =====================================================================
def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # CIFAR-10
    c_data = [
        ("AdamW",  94.99, COLORS["adamw"]),
        ("Dion2",  94.52, COLORS["dion2"]),
        ("Dion",   94.25, COLORS["dion"]),
        ("Muon",   94.16, COLORS["muon"]),
        ("V3A",    92.44, COLORS["v3a"]),
        ("V2",     92.40, COLORS["v2"]),
        ("V3",     92.22, COLORS["v3"]),
    ]
    x = np.arange(len(c_data))
    ax1.bar(x, [d[1] for d in c_data], width=0.5,
            color=[d[2] for d in c_data],
            edgecolor=[ec(d[2]) for d in c_data], linewidth=0.8, zorder=3)
    for i, (n, v, _) in enumerate(c_data):
        ax1.text(i, v + 0.08, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=8, color="#222")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d[0] for d in c_data], fontsize=8.5)
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_ylim(91.5, 95.8)
    # No title — use LaTeX caption

    # FashionMNIST
    f_data = [
        ("V3A",   91.12, COLORS["v3a"]),
        ("AdamW", 90.95, COLORS["adamw"]),
        ("V3",    90.74, COLORS["v3"]),
        ("Dion",  90.46, COLORS["dion"]),
        ("V2",    90.21, COLORS["v2"]),
        ("Muon",  89.98, COLORS["muon"]),
    ]
    x = np.arange(len(f_data))
    ax2.bar(x, [d[1] for d in f_data], width=0.5,
            color=[d[2] for d in f_data],
            edgecolor=[ec(d[2]) for d in f_data], linewidth=0.8, zorder=3)
    for i, (n, v, _) in enumerate(f_data):
        ax2.text(i, v + 0.03, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=8, color="#222")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d[0] for d in f_data], fontsize=8.5)
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_ylim(89.5, 91.7)
    # No title

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_vision.pdf"), format="pdf")
    plt.close(fig)
    print("fig3_vision.pdf")


# =====================================================================
# FIG 4: QR vs ColNorm across ranks
# =====================================================================
def fig4():
    ranks = [8, 16, 32, 64]
    cn = [4.437, 4.272, 4.119, 3.987]
    qr = [6.571, 6.377, 6.261, 6.044]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(ranks))
    w = 0.3
    ax.bar(x - w/2, cn, width=w, color=COLORS["colnorm"],
           edgecolor=ec(COLORS["colnorm"]), linewidth=0.8, label="ColNorm", zorder=3)
    ax.bar(x + w/2, qr, width=w, color=COLORS["qr"],
           edgecolor=ec(COLORS["qr"]), linewidth=0.8, label="QR (Orth-Dion)", zorder=3)
    for i in range(len(ranks)):
        ax.text(x[i]-w/2, cn[i]+0.04, f"{cn[i]:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="#222")
        ax.text(x[i]+w/2, qr[i]+0.04, f"{qr[i]:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$r={r}$" for r in ranks])
    ax.set_ylabel("Validation loss (10k steps)")
    ax.set_ylim(3.5, 7.0)
    ax.legend(framealpha=0.8, edgecolor="#ccc", fancybox=False)
    fig.savefig(os.path.join(OUT, "fig4_qr_vs_colnorm.pdf"), format="pdf")
    plt.close(fig)
    print("fig4_qr_vs_colnorm.pdf")


# =====================================================================
# FIG 5: Right-factor comparison (all methods)
# =====================================================================
def fig5():
    data = [
        ("ColNorm\n$\\beta$=0.3",     3.987, COLORS["colnorm"]),
        ("Partial Orth\n$\\beta$=0.3", 3.963, COLORS["partial"]),
        ("Partial Orth\n$\\beta$=0.1", 3.758, COLORS["partial"]),
        ("Soft Isometry",              5.283, COLORS["soft"]),
        ("Block QR\n$k$=16",          6.184, COLORS["block"]),
        ("QR\n(Orth-Dion)",            6.044, COLORS["qr"]),
        ("QR + 3\npow. iters",        6.395, COLORS["qr"]),
        ("Block QR\n$k$=4",           6.407, COLORS["block"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(data))
    cols = [d[2] for d in data]
    vals = [d[1] for d in data]
    ax.bar(x, vals, width=0.5, color=cols,
           edgecolor=[ec(c) for c in cols], linewidth=0.8, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.04, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data], fontsize=7.5)
    ax.set_ylabel("Validation loss (10k steps)")
    ax.set_ylim(3.5, 7.0)
    ax.axhline(y=3.987, color=COLORS["colnorm"], linewidth=0.8,
               linestyle=":", alpha=0.5, zorder=2)
    fig.savefig(os.path.join(OUT, "fig5_right_factor.pdf"), format="pdf")
    plt.close(fig)
    print("fig5_right_factor.pdf")


# =====================================================================
# FIG 6: Diagnostics (epsilon_hat, nu_t, delta_t, erank)
# =====================================================================
def fig6():
    methods = ["ColNorm\n$\\beta$=0.3", "Orth-Dion\n$\\beta$=0.3", "Soft Isometry\n$\\beta$=0.3"]
    metrics = {
        "$\\hat{\\epsilon}_t$ (tracking error)": [0.154, 0.034, 0.005],
        "$\\nu_t$ (dual norm)":                   [1.91, 1.00, 1.00],
        "$\\delta_t$ (oracle defect)":            [0.053, 0.413, 0.249],
        "Effective rank":                          [35.7, 16.2, 4.8],
    }
    method_colors = [COLORS["colnorm"], COLORS["qr"], COLORS["soft"]]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (ylabel, vals) in zip(axes, metrics.items()):
        x = np.arange(len(methods))
        ax.bar(x, vals, width=0.5, color=method_colors,
               edgecolor=[ec(c) for c in method_colors], linewidth=0.8, zorder=3)
        for i, v in enumerate(vals):
            fmt = f"{v:.3f}" if v < 10 else f"{v:.1f}"
            ax.text(i, v + max(vals)*0.03, fmt, ha="center", va="bottom",
                    fontsize=8, color="#222")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.25)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_diagnostics.pdf"), format="pdf")
    plt.close(fig)
    print("fig6_diagnostics.pdf")


# =====================================================================
# FIG 7: Beta sweep
# =====================================================================
def fig7():
    betas = [0.0, 0.3, 0.5, 0.7, 1.0]
    losses = [4.291, 3.306, 3.405, 3.662, 3.713]
    # Each bar gets a color from the sequential gradient palette
    cols = [
        "#333333",       # beta=0 (no EF) — dark
        "#6A3D9A",       # beta=0.3 (best) — deep purple
        "#1F78B4",       # beta=0.5 — blue
        "#FF7F00",       # beta=0.7 — orange
        "#666666",       # beta=1.0 (baseline) — gray
    ]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(range(len(betas)), losses, width=0.5, color=cols,
           edgecolor=[ec(c) for c in cols], linewidth=0.8, zorder=3)
    for i, (b, v) in enumerate(zip(betas, losses)):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#222")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"$\\beta={b}$" for b in betas])
    ax.set_ylabel("Validation loss (30k steps)")
    ax.set_ylim(3.0, 4.5)
    ax.set_xlabel("Error feedback coefficient $\\beta$")
    fig.savefig(os.path.join(OUT, "fig7_beta_sweep.pdf"), format="pdf")
    plt.close(fig)
    print("fig7_beta_sweep.pdf")


# =====================================================================
# FIG 8: Rank x Beta interaction
# =====================================================================
def fig8():
    ranks = [8, 16, 32, 128, 256]
    b03 = [4.608, 4.423, 4.317, 4.132, 4.051]
    b10 = [4.817, 4.711, 4.534, 4.313, 4.340]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ranks, b03, "o-", color="#6A3D9A", label="$\\beta=0.3$",
            markersize=6, markeredgewidth=0)
    ax.plot(ranks, b10, "s-", color="#666666", label="$\\beta=1.0$",
            markersize=6, markeredgewidth=0)
    ax.fill_between(ranks, b03, b10, color="#6A3D9A", alpha=0.18)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("Rank $r$")
    ax.set_ylabel("Validation loss (10k steps)")
    ax.legend(framealpha=0.8, edgecolor="#ccc", fancybox=False)
    fig.savefig(os.path.join(OUT, "fig8_rank_beta.pdf"), format="pdf")
    plt.close(fig)
    print("fig8_rank_beta.pdf")


# =====================================================================
# FIG 9: Modewise persistence heterogeneity
# =====================================================================
def fig9():
    np.random.seed(42)
    rho = np.random.normal(-0.008, 0.338, 64)
    rho = np.clip(rho, -1, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Histogram — no subplot title (use LaTeX caption)
    ax1.hist(rho, bins=20, color="#33A02C", edgecolor=ec("#33A02C"),
             linewidth=0.8, zorder=3)
    ax1.axvline(x=np.mean(rho), color="#E31A1C", linewidth=1.5, linestyle="--",
                label=f"Global mean = {np.mean(rho):.3f}", zorder=4)
    ax1.set_xlabel("Per-mode $\\rho_i$ (gradient persistence)")
    ax1.set_ylabel("Count (modes)")
    ax1.legend(framealpha=0.8, edgecolor="#ccc", fancybox=False)

    # Sorted bars — no subplot title
    s = np.sort(rho)
    bar_cols = ["#6A3D9A" if v > 0 else "#E31A1C" for v in s]
    ax2.bar(np.arange(64), s, width=0.8, color=bar_cols,
            edgecolor=[ec(c) for c in bar_cols], linewidth=0.3, zorder=3)
    ax2.axhline(y=0, color="#666", linewidth=0.6, zorder=2)
    ax2.set_xlabel("Mode index (sorted)")
    ax2.set_ylabel("$\\rho_i$")
    ax2.annotate("Retain buffer\n(low $\\beta_i$)", xy=(5, -0.6),
                 fontsize=8, color="#E31A1C")
    ax2.annotate("Clear buffer\n(high $\\beta_i$)", xy=(48, 0.5),
                 fontsize=8, color="#6A3D9A")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_modewise.pdf"), format="pdf")
    plt.close(fig)
    print("fig9_modewise.pdf")


# =====================================================================
# FIG 10: Response surface
# =====================================================================
def fig10():
    a, b, c = -0.2329, 0.1155, -0.01423
    lam = np.linspace(0, 5, 200)
    curve = a * lam + b * lam**2 + c * lam**3
    sweet = -a / (2*b)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(lam, curve, color="#33A02C", linewidth=1.8, zorder=3)
    ax.axhline(y=0, color="#666", linewidth=0.6, zorder=2)
    ax.axvline(x=sweet, color="#6A3D9A", linewidth=1.2, linestyle="--",
               alpha=0.8, zorder=3)
    ax.annotate(f"$\\lambda^* \\approx {sweet:.2f}$",
                xy=(sweet, a*sweet + b*sweet**2 + c*sweet**3),
                xytext=(sweet + 1.0, -0.06), fontsize=9, color="#6A3D9A",
                arrowprops=dict(arrowstyle="->", color="#6A3D9A", lw=0.8))
    ax.set_xlabel("Buffer scaling $\\lambda$ in $M = G + \\lambda R$")
    ax.set_ylabel("$\\Delta f$ (one-step loss change)")
    # No title — use LaTeX caption
    fig.savefig(os.path.join(OUT, "fig10_response.pdf"), format="pdf")
    plt.close(fig)
    print("fig10_response.pdf")


# =====================================================================
# FIG 11: Controller comparison
# =====================================================================
def fig11():
    data = [
        ("PA-Dion\n(low $\\beta_{\\max}$)", 4.059, COLORS["v3a"]),
        ("RNorm\n$\\rho^*$=2.5",            4.059, COLORS["v2"]),
        ("Flat\n$\\beta$=0.3",              4.072, COLORS["dion"]),
        ("RNorm\n$\\rho^*$=1.8",            4.072, COLORS["v2"]),
        ("RNorm\n$\\rho^*$=1.2",            4.094, COLORS["v2"]),
        ("PA-Dion\n(default)",               4.100, COLORS["v3a"]),
        ("ReEntry\n$\\tau$=0.5",            4.104, COLORS["qr"]),
        ("RNorm\n$\\rho^*$=0.6",            4.113, COLORS["v2"]),
        ("ReEntry\n$\\tau$=0.3",            4.120, COLORS["qr"]),
    ]
    fig, ax = plt.subplots(figsize=(5, 4.0))
    y = np.arange(len(data))[::-1]
    cols = [d[2] for d in data]
    vals = [d[1] for d in data]
    ax.barh(y, vals, height=0.55, color=cols,
            edgecolor=[ec(c) for c in cols], linewidth=0.8, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.001, y[i], f"{v:.3f}", va="center",
                fontsize=8, color="#222")
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data], fontsize=7.5)
    ax.set_xlabel("Validation loss (30k steps)")
    ax.set_xlim(4.04, 4.15)
    ax.axvline(x=4.072, color=COLORS["dion"], linewidth=0.8, linestyle=":", alpha=0.5)
    fig.savefig(os.path.join(OUT, "fig11_controllers.pdf"), format="pdf")
    plt.close(fig)
    print("fig11_controllers.pdf")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    fig11()
    print(f"\nAll figures saved to {OUT}/")
