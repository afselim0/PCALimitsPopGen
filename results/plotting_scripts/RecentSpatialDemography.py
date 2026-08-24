#%%
"""
Recent-split demography schematic and the spatial PC pair, at three migration rates
(4 panels). Figure 12 in CLAUDE.md.

A  schematic: a K x K grid of circles (deme B_pq, nearest-neighbour lines) in a box,
   plus one large circle for panmictic A, both descending from a single ancestor --
   not a demesdraw tube plot, which has no notion of the grid's spatial layout and would
   just draw 1+K^2 disconnected tubes. Each deme circle is filled with the same
   grid_color() used for its samples in panels B-D, so the schematic doubles as a legend
   for which color is which grid position.
B,C,D  PC2 vs PC3 (the theoretical spatial pair, located by overlap) at weak,
   intermediate and strong migration, B's samples colored by grid position (a 2D
   colormap over (row, col)) and A's samples in light grey.

Reads results/analysis/RecentSpatialDemography, writes
results/figures/RecentSpatialDemography.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RecentSpatialDemography")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

COLOR_A = "0.75"
COLOR_EDGE = "0.2"
COLOR_OTHER = "0.82"
LABELS = ("weak", "intermediate", "strong")


# Bivariate colormap corners (row, col) = (0,0), (0,K-1), (K-1,0), (K-1,K-1). Bilinear
# interpolation between four perceptually distinct colors, rather than mapping either
# grid axis onto hue: hue is cyclic (0 and 360 degrees are the same color), so a naive
# hue = col / (K-1) makes the two opposite edges of the grid falsely look similar while
# spatially adjacent columns near the middle can jump between unrelated hues. Bilinear
# RGB blending has no such wraparound -- color distance is monotonic in grid distance
# along both axes, which is the point of coloring by position at all.
_CORNER_00 = np.array(colors.to_rgb("#4A6FE3"))   # blue
_CORNER_0K = np.array(colors.to_rgb("#E45756"))   # red
_CORNER_K0 = np.array(colors.to_rgb("#54A24B"))   # green
_CORNER_KK = np.array(colors.to_rgb("#EECA3B"))   # yellow


def grid_color(row, col, K):
    """Bilinear-interpolated color for grid position (row, col) in a K x K grid --
    shared by the schematic's demes and the PC panels' sample colors, so the two can be
    compared directly."""
    u = col / max(K - 1, 1)
    v = row / max(K - 1, 1)
    top = _CORNER_00 * (1 - u) + _CORNER_0K * u
    bottom = _CORNER_K0 * (1 - u) + _CORNER_KK * u
    return tuple(top * (1 - v) + bottom * v)


def panel_schematic(ax, K):
    """K x K grid of small circles (B), each colored by its grid position to match the
    PC panels, plus one large circle for A, both under a common ancestor drawn as a
    single point above."""
    grid_span = K - 1
    x0 = 2.6  # grid's left edge, leaving room for A on the left
    for r in range(K):
        for c in range(K):
            if c + 1 < K:
                ax.plot([x0 + c, x0 + c + 1], [r, r], "-", color=COLOR_EDGE,
                        linewidth=0.7, zorder=1)
            if r + 1 < K:
                ax.plot([x0 + c, x0 + c], [r, r + 1], "-", color=COLOR_EDGE,
                        linewidth=0.7, zorder=1)
    for r in range(K):
        for c in range(K):
            ax.add_patch(Circle((x0 + c, r), 0.28, facecolor=grid_color(r, c, K),
                                edgecolor=COLOR_EDGE, linewidth=0.6, zorder=2))

    a_x, a_y = 0.9, grid_span / 2
    ax.add_patch(Circle((a_x, a_y), 0.85, facecolor=COLOR_A, edgecolor=COLOR_EDGE,
                        linewidth=0.9, zorder=2))
    ax.text(a_x, a_y, "A", ha="center", va="center", fontsize=10)
    ax.text(x0 + grid_span / 2, -0.9, "B (K x K)", ha="center", va="center", fontsize=9)

    anc_x, anc_y = (a_x + x0 + grid_span / 2) / 2, grid_span + 1.6
    ax.plot(anc_x, anc_y, "o", color=COLOR_EDGE, markersize=5, zorder=3)
    ax.text(anc_x, anc_y + 0.35, "ANC", ha="center", va="bottom", fontsize=9)
    for x, y in [(a_x, a_y + 0.85), (x0 + grid_span / 2, grid_span + 0.3)]:
        ax.add_patch(FancyArrowPatch((anc_x, anc_y - 0.1), (x, y), color=COLOR_EDGE,
                                     linewidth=0.8, arrowstyle="-", zorder=1))

    ax.set_xlim(-0.3, x0 + grid_span + 0.5)
    ax.set_ylim(-1.4, grid_span + 2.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("demography")


def panel_pc(ax, d, K, title):
    is_A = d.deme_row.isna().values
    ax.scatter(d.pc2[is_A], d.pc3[is_A], s=4, color=COLOR_OTHER, alpha=0.6,
              edgecolors="none", zorder=1)
    dB = d[~is_A]
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(dB.deme_row, dB.deme_col)]
    ax.scatter(dB.pc2, dB.pc3, s=6, c=c, edgecolors="none", zorder=2)
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    ax.set_title(title)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    pc = pd.read_csv(os.path.join(analysis_dir, "pc.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("label")
    K = int(pc.deme_row.max() + 1)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
    panel_schematic(axes[0], K)

    for j, label in enumerate(LABELS, start=1):
        m = float(meta.loc[label, "m"])
        M = float(meta.loc[label, "M"])
        d = pc[pc.label == label]
        panel_pc(axes[j], d, K, rf"{label}  ($M={M:.2g}$)")

    for ax, letter in zip(axes, "ABCD"):
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "RecentSpatialDemography.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
