# Plot Style Guide for Academic Papers

This document specifies the exact matplotlib style used in our AdaDion paper. Copy this entire file into your prompt when asking an LLM to generate plotting code.

Style references: Balles et al. (ICML 2018), Daxberger et al. (NeurIPS 2021), Jordan (Muon, 2024).

---

## General Principles

- Every figure must be a self-contained PDF.
- Figures must be readable at the size they appear in a two-column LaTeX document (column width ~3.3 inches, full width ~7 inches).
- No decorative elements. No background color, no watermarks, no unnecessary borders.
- No figure-level titles (suptitle). Use the LaTeX caption instead.

---

## Typography

- **Font**: Serif with Computer Modern math rendering.
- **Do not use `text.usetex: True`** unless you are certain LaTeX is installed. Use the fallback:
```python
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})
```
- **Font sizes**:
  - Base font: 10pt
  - Axis labels: 10pt
  - Tick labels: 9pt
  - Legend: 8pt
  - Subplot titles: 11pt
  - Annotations: 8-9pt
- **Weight**: Everything regular. No bold anywhere.

---

## Colors

Saturated, colorblind-distinguishable palette. One color per method, consistent across all figures.

```python
COLORS = {
    "adadion": "#6A3D9A",  # deep purple
    "dion":    "#33A02C",  # strong green
    "dion2":   "#E31A1C",  # strong red
    "muon":    "#FF7F00",  # strong orange
    "adamw":   "#1F78B4",  # strong blue
}
```

For multi-gamma or multi-run comparisons, use a sequential gradient:
```python
GAMMA_COLORS = {
    "0.5": "#1F78B4",   # blue
    "1.0": "#33A02C",   # green
    "1.5": "#FF7F00",   # orange
    "2.0": "#E31A1C",   # red
    "2.5": "#6A3D9A",   # purple
    "3.0": "#A65628",   # brown
}
```

---

## Axes, Spines, and Grid

All four spines visible. Dashed grid on both axes.

```python
plt.rcParams.update({
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
})
```

- Tick marks: default direction and length.
- Axis labels: sentence case ("Validation accuracy (%)"). Use mathtext for symbols: `r"Rank scale factor $\gamma$"`.

---

## Line Plots (Loss Curves, Accuracy Curves, Rank Dynamics)

- **Line width**: 1.8 for bold smooth lines, 0.5-0.8 for raw faint lines.
- **Raw + smooth technique**: Plot the raw data as a faint line (alpha=0.1-0.15), then plot a smoothed version on top as a bold line (alpha=1.0). This shows both the noise and the trend.
```python
from scipy.ndimage import uniform_filter1d
raw = np.array(values)
smooth = uniform_filter1d(raw, size=5)  # 5-point moving average
ax.plot(x, raw, color=c, alpha=0.15, linewidth=0.8)    # faint raw
ax.plot(x, smooth, color=c, linewidth=1.8, label=name)  # bold smooth
```
- **Confidence bands** (multiple seeds): Use `fill_between` with alpha=0.18.
```python
ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18)
```
- **Log scale**: Use log scale on y-axis for loss plots when range spans more than one order of magnitude.
- **Markers**: Use only when fewer than 10 data points per line. Marker size 5-7, `markeredgewidth=0`. Preferred: `"o"`, `"s"`.

---

## Bar Charts

- **Bar width**: 0.5 (standard matplotlib bars, no rounded corners).
- **Fill**: Full color from the palette.
- **Edge**: Same color at 90% opacity, linewidth 0.8.
```python
ax.bar(x, values, width=0.5, color=colors,
       edgecolor=[mcolors.to_rgba(c, 0.9) for c in colors],
       linewidth=0.8, zorder=3)
```
- **Error bars**: Black, thin, small caps.
```python
ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
            capsize=3.5, capthick=1.0, elinewidth=1.0, zorder=4)
```
- **Value labels**: Above each bar, centered, 8-8.5pt, color `#222222`.
```python
ax.text(x[i], val + offset, f"{val:.2f}", ha="center", va="bottom",
        fontsize=8.5, color="#222")
```
- **Axis**: Start y-axis slightly below minimum value so bars do not float.

---

## Legends

- Place where it does not overlap data. Preferred: `"best"`, `"upper right"`, `"lower right"`.
- Semi-transparent background, thin gray border, no rounded box:
```python
ax.legend(framealpha=0.8, edgecolor="#ccc", fancybox=False, fontsize=8)
```
- For shared legends across subplots, place in the first subplot only or use `fig.legend()` below the row.

---

## Subplots

- Use `plt.subplots(1, N)` for horizontal arrangements.
- All subplots in a row must have the same height.
- Sizing: approximately 5 inches per subplot width, 3.2-4.0 inches height.
  - 2-panel: `figsize=(8, 3.2)` or `figsize=(10, 3.8)`
  - 3-panel: `figsize=(14, 3.8)`
- No suptitle. Use LaTeX caption.
- Use `plt.tight_layout()` for spacing.

---

## Annotations

- Sparingly. Only annotate the best result or a specific callout.
- Font size 8-9pt. Color matches the data point.
- Use `textcoords="offset points"` with offset (10, 8) so label does not sit on the marker.
```python
ax.annotate(f"{val:.2f}", (x, y), textcoords="offset points",
            xytext=(10, 8), fontsize=9, color=color)
```

---

## Dual-Axis Plots

- Left axis: primary metric, solid lines.
- Right axis: secondary metric, dashed lines.
- Color-code axis labels and ticks to match the lines.
- Right spine must be visible:
```python
ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
```

---

## Export

```python
fig.savefig("figure_name.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
```

---

## Full Boilerplate

Paste this at the top of every plotting script:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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
    "adadion": "#6A3D9A",
    "dion":    "#33A02C",
    "dion2":   "#E31A1C",
    "muon":    "#FF7F00",
    "adamw":   "#1F78B4",
}

LABELS = {
    "adadion": "AdaDion V2",
    "dion":    "Dion",
    "dion2":   "Dion2",
    "muon":    "Muon",
    "adamw":   "AdamW",
}
```

---

## Checklist

1. Serif font, Computer Modern math.
2. All four spines visible.
3. Dashed grid on both axes (alpha=0.3).
4. Training curves: raw faint + smooth bold + confidence bands.
5. Loss on log scale.
6. Bar charts: standard fill, black error bars, value labels above.
7. No bold text. No suptitle.
8. Saturated color palette, consistent across all figures.
9. Legends: framealpha=0.8, edgecolor="#ccc", fancybox=False.
10. Exported as PDF.
