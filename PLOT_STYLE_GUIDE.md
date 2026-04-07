# Plot Style Guide for Academic Papers

This document specifies the exact matplotlib style to use when generating figures for our papers. Copy this entire file into your prompt when asking an LLM to generate plotting code.

---

## General Principles

- Every figure must be a self-contained PDF exported at 300 DPI.
- Figures must be readable at the size they appear in a two-column LaTeX document (column width ~3.3 inches, full width ~7 inches).
- Do not add decorative elements. No background color, no watermarks, no unnecessary borders.
- Prefer showing data over describing it. If something can be a plot, do not make it a table. If something is a single number, do not make it a plot.

---

## Typography

- **Font family**: Use `serif` with LaTeX rendering enabled.
- **How to set**:
```python
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
```
- If LaTeX is unavailable, fall back to `"font.family": "serif"` with `"mathtext.fontset": "cm"`.
- **Font sizes** (these are for a single-column figure at ~3.3 inches wide; scale proportionally for full-width):
  - Axis labels: 9pt
  - Tick labels: 8pt
  - Legend: 8pt
  - Subplot titles (if needed): 10pt
  - Figure suptitle: avoid when possible; use the LaTeX caption instead
- **Weight**: Everything is regular weight. Never use bold for axis labels, tick labels, or legend entries. The only exception is if you need to highlight one specific entry in a legend.

---

## Colors

Use a muted, colorblind-friendly palette. Do not use matplotlib defaults. Assign one color per method and keep it consistent across all figures in the paper.

```python
COLORS = {
    "AdaDion V2": "#7B68AE",   # muted purple
    "Dion":       "#4DAF7C",   # muted green
    "Dion2":      "#D95F5F",   # muted red
    "Muon":       "#E8943A",   # muted orange
    "AdamW":      "#5B8DBE",   # muted blue
}
```

For fills (e.g., shaded regions, bar interiors), use the same color at 40-50% opacity. For edges and lines, use the same color at 85-100% opacity.

```python
import matplotlib.colors as mcolors
fill = mcolors.to_rgba(color, alpha=0.45)
edge = mcolors.to_rgba(color, alpha=0.90)
```

---

## Line Plots (Loss Curves, Accuracy Curves, Convergence)

- **Line width**: 1.5 for primary lines. 0.8 for secondary/reference lines.
- **Markers**: Use only when there are fewer than 10 data points per line. Marker size 5, no edge. Preferred markers: `"o"`, `"s"`, `"^"`, `"D"`, `"v"`.
- **Smoothing**: Apply light smoothing for noisy training curves:
```python
from scipy.ndimage import uniform_filter1d
smoothed = uniform_filter1d(raw_values, size=5)
```
  Plot the smoothed line and optionally show the raw data as a faint line (alpha=0.15) or shaded region.
- **Confidence bands** (multiple seeds): Use `fill_between` with alpha=0.12. Do not use error bars on line plots.
```python
ax.plot(x, mean, color=c, linewidth=1.5, label=name)
ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.12)
```
- **Log scale**: Use log scale on the y-axis for loss plots when the range spans more than one order of magnitude. Use `ax.set_yscale("log")`.

---

## Bar Charts

- **Bar width**: 0.45 to 0.55 (never wider than 0.6; wide bars create too much empty colored area).
- **Rounded corners**: Apply slight rounding to the top corners only.
```python
from matplotlib.patches import FancyBboxPatch

def rounded_bar(ax, x, height, width, facecolor, edgecolor, bottom=0):
    box = FancyBboxPatch(
        (x - width/2, bottom), width, height,
        boxstyle="round,pad=0,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
    )
    ax.add_patch(box)
```
- **Fill**: Use the method color at 45% opacity for the fill and 90% opacity for the edge (1.0pt line).
- **Value labels**: Place the numeric value above each bar, centered, in 8pt font, color `#333333`. Format: accuracy as `"95.38"` (no % sign on the label if the axis already says %), loss as `"0.2072"`.
- **Error bars**: Thin lines (linewidth=1.0), small caps (capsize=3), color `#444444`. Do not use thick error bars.
- **Grouped bars**: When comparing two groups (e.g., ResNet vs ViT), use side-by-side bars with width 0.35 each, separated by a small gap. Add a legend to distinguish groups.
- **Axis**: Always start the y-axis slightly below the minimum value to avoid bars appearing to float. Use `ax.set_ylim(bottom=min_val - margin)`.

---

## Axes and Grid

- **Spines**: Remove top and right spines.
```python
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```
- **Grid**: Light horizontal grid lines only (no vertical). Alpha=0.2, linewidth=0.5, color `#cccccc`.
```python
ax.yaxis.grid(True, alpha=0.2, linewidth=0.5, color="#cccccc")
ax.xaxis.grid(False)
```
- **Tick marks**: Small ticks pointing outward. Remove minor ticks unless using log scale.
```python
ax.tick_params(direction="out", length=3, width=0.6)
```
- **Axis labels**: Sentence case ("Validation accuracy (%)"), not title case. Use LaTeX math mode for symbols: `r"Learning rate $\alpha$"`.

---

## Legends

- Place the legend where it does not overlap any data. Preferred locations: `"upper right"`, `"lower right"`, or outside the plot area.
- Use a semi-transparent background with no border:
```python
ax.legend(framealpha=0.7, edgecolor="none", fontsize=8)
```
- If the legend is shared across subplots, place it once below the figure row using `fig.legend()`.
- Order legend entries to match the visual ordering in the plot (e.g., top line first if lines are stacked).

---

## Subplots and Multi-Panel Figures

- Use `plt.subplots(1, N)` for horizontal arrangements. Never mix different aspect ratios within the same figure.
- All subplots in a row must have the same height. Use `figsize=(width, height)` where width scales with N (e.g., 5 inches per subplot) and height is fixed (3.5-4.0 inches).
- Add spacing with `plt.tight_layout()` or `fig.subplots_adjust()`.
- Do not put a `suptitle` on the figure. Use the LaTeX `\caption{}` for the title.
- If subplots share an x-axis or y-axis, use `sharex=True` or `sharey=True` and remove redundant labels.

---

## Annotations

- Use `ax.annotate()` sparingly. Only annotate the best result or a specific callout.
- Annotation font size: 8-9pt. Color: same as the data point being annotated.
- Use `textcoords="offset points"` with a small offset (8-12 points) so the label does not sit on top of the marker.
- Never let annotations overlap with each other or with data. Manually adjust positions if needed.

---

## Dual-Axis Plots

- Use dual y-axes only when showing two related quantities with different scales (e.g., accuracy and compression ratio).
- Left axis: primary metric, solid lines.
- Right axis: secondary metric, dashed lines.
- Color-code the axis labels and tick labels to match the corresponding line.
- Make the right spine visible:
```python
ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
```

---

## Export

- Always export as PDF for LaTeX inclusion:
```python
fig.savefig("figure_name.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)
```
- DPI setting does not affect vector PDF, but set `figure.dpi=300` for any rasterized elements.
- Close figures after saving: `plt.close(fig)`.

---

## Boilerplate

Paste this at the top of every plotting script:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

COLORS = {
    "AdaDion V2": "#7B68AE",
    "Dion":       "#4DAF7C",
    "Dion2":      "#D95F5F",
    "Muon":       "#E8943A",
    "AdamW":      "#5B8DBE",
}

def fill_color(c, alpha=0.45):
    return mcolors.to_rgba(c, alpha)

def edge_color(c, alpha=0.90):
    return mcolors.to_rgba(c, alpha)

def rounded_bar(ax, x, height, width, fc, ec, bottom=0):
    box = FancyBboxPatch(
        (x - width / 2, bottom), width, height,
        boxstyle="round,pad=0,rounding_size=0.025",
        facecolor=fc, edgecolor=ec, linewidth=1.0,
    )
    ax.add_patch(box)

def setup_grid(ax):
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.5, color="#cccccc")
    ax.xaxis.grid(False)
    ax.tick_params(direction="out", length=3, width=0.6)
```

---

## Checklist Before Submitting Figures

1. All text is readable at column width without zooming.
2. No overlapping labels, annotations, or legends.
3. Colors are consistent across all figures in the paper.
4. No bold text anywhere except possibly one highlighted legend entry.
5. Exported as PDF.
6. No figure has a suptitle (use LaTeX caption).
7. Bars are narrow with rounded corners and shaded fill.
8. Line plots use confidence bands, not error bars.
9. Axis labels use sentence case and LaTeX math for symbols.
10. Top and right spines are removed.
