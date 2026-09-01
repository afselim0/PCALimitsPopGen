#%%
"""
Recent-split demography schematic and the spatial PC pair, at three migration rates
(4 panels). Figure 12 in CLAUDE.md.

A  schematic: a K x K grid of circles (deme B_pq, nearest-neighbour lines) in a box,
   plus one large circle for panmictic A, both descending from a grey ancestral circle
   drawn above them; the split is drawn as a horizontal bar spanning A and the grid with
   a downward arrow at each end, annotated with the split time -- not a demesdraw tube
   plot, which has no notion of the grid's spatial layout and would just draw 1+K^2
   disconnected tubes. Each deme circle is filled with the same
   grid_color() used for its samples in panels B-D, so the schematic doubles as a legend
   for which color is which grid position.
B,C,D  the leading three PCs by eigenvalue rank at weak, intermediate and strong
   migration: a short one-axis strip of PC1 (the A/B split, so A is drawn black there)
   above a PC2-vs-PC3 scatter (the grid's spatial pair, A in light grey). B's samples
   are colored by grid position throughout (a 2D colormap over (row, col)).

Reads results/analysis/RecentSpatialDemography, writes
results/figures/RecentSpatialDemography.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch

from crossing_panel import save_figure
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


def panel_schematic(ax, K, t_split):
    """K x K grid of small circles (B), each colored by its grid position to match the
    PC panels, plus one large circle for A, both descending from a grey ancestral circle
    via a horizontal bar with a downward arrow over each descendant."""
    grid_span = K - 1
    x0 = 3.4  # grid's left edge, leaving room for A on the left
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

    # A sits level with the grid's top row so both descendants hang the same distance
    # below the split bar and the two arrows read as one simultaneous event
    a_r = 1.15
    a_x, a_y = 1.0, grid_span - a_r
    ax.add_patch(Circle((a_x, a_y), a_r, facecolor=COLOR_A, edgecolor=COLOR_EDGE,
                        linewidth=0.9, zorder=2))
    ax.text(a_x, a_y, "A", ha="center", va="center", fontsize=10)
    grid_x = x0 + grid_span / 2
    ax.text(grid_x, -0.9, "B (9 x 9)", ha="center", va="center", fontsize=9)

    bar_y = grid_span + 2.0
    anc_x, anc_y = (a_x + grid_x) / 2, bar_y + 2.6
    anc_r = 1.35
    ax.add_patch(Circle((anc_x, anc_y), anc_r, facecolor=COLOR_A, edgecolor=COLOR_EDGE,
                        linewidth=0.9, zorder=2))
    ax.text(anc_x, anc_y, "Ancestral", ha="center", va="center", fontsize=7.5)

    # split: a horizontal bar spanning A and the grid, fed by the ancestor, with one
    # downward arrow at each end onto the two descendants
    ax.plot([a_x, grid_x], [bar_y, bar_y], "-", color=COLOR_EDGE, linewidth=1.0,
            zorder=1)
    ax.plot([anc_x, anc_x], [anc_y - anc_r, bar_y], "-", color=COLOR_EDGE,
            linewidth=1.0, zorder=1)
    for x, y in [(a_x, a_y + a_r), (grid_x, grid_span + 0.75)]:
        ax.add_patch(FancyArrowPatch((x, bar_y), (x, y), color=COLOR_EDGE,
                                     linewidth=1.0, arrowstyle="-|>",
                                     mutation_scale=9, shrinkA=0, shrinkB=0, zorder=1))
    ax.text(anc_x, bar_y - 0.3, f"{t_split:g} generations ago", ha="center", va="top",
            fontsize=8)

    ax.set_xlim(-0.3, x0 + grid_span + 0.5)
    ax.set_ylim(-1.4, anc_y + anc_r + 0.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ax.set_title("demography")


def panel_pc1(ax, d, K, title):
    """PC1 alone on a horizontal strip: one axis, no y axis. A is black here (not the
    light grey of the PC2/PC3 panels) because PC1 is the A/B split itself, so which
    samples are A is the whole content of the strip."""
    is_A = d.deme_row.isna().values
    y = np.zeros(len(d))
    dB = d[~is_A]
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(dB.deme_row, dB.deme_col)]
    ax.scatter(dB.pc1, y[~is_A], s=6, c=c, edgecolors="none", alpha=0.7, zorder=2)
    ax.scatter(d.pc1[is_A], y[is_A], s=6, color="black", edgecolors="none",
               alpha=0.7, zorder=3)
    ax.set_xlabel("PC1", labelpad=1)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, pad=6)


def panel_pc(ax, d, K, title=None):
    is_A = d.deme_row.isna().values
    ax.scatter(d.pc2[is_A], d.pc3[is_A], s=4, color=COLOR_OTHER, alpha=0.6,
              edgecolors="none", zorder=1)
    dB = d[~is_A]
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(dB.deme_row, dB.deme_col)]
    ax.scatter(dB.pc2, dB.pc3, s=6, c=c, edgecolors="none", zorder=2)
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    if title:
        ax.set_title(title, pad=6)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    pc = pd.read_csv(os.path.join(analysis_dir, "pc.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("label")
    K = int(pc.deme_row.max() + 1)
    t_split = float(meta["t_split"].iloc[0]) if "t_split" in meta else 500.0

    # Each PC panel is a short PC1 strip stacked over the PC2-vs-PC3 scatter, so the
    # three columns need two rows while the schematic spans both.
    fig = plt.figure(figsize=(15.5, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 5], hspace=0.55, wspace=0.32)
    ax_schematic = fig.add_subplot(gs[:, 0])
    panel_schematic(ax_schematic, K, t_split)

    letter_axes = [ax_schematic]
    for j, label in enumerate(LABELS, start=1):
        M = float(meta.loc[label, "M"])
        cos = float(meta.loc[label, "cos"])
        d = pc[pc.label == label]
        ax_top = fig.add_subplot(gs[0, j])
        panel_pc1(ax_top, d, K, rf"$M={M:.2g}$")
        panel_pc(fig.add_subplot(gs[1, j]), d, K,
                 rf"cosine similarity $={cos:.2f}$")
        letter_axes.append(ax_top)

    for ax, letter in zip(letter_axes, "ABCD"):
        ax.text(-0.15, 1.06 if ax is ax_schematic else 1.35, letter,
                transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    out = save_figure(fig, figure_dir, "RecentSpatialDemography")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
