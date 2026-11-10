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
B  the PC2-vs-PC3 scatter (the grid's spatial pair, A in light grey) at weak,
   intermediate and strong migration, titled with M and the cosine similarity. B's
   samples are colored by grid position (a 2D colormap over (row, col)). PC1 is not
   shown: it is the A/B split, which panel A already states.

Reads results/analysis/RecentSpatialDemography, writes
results/figures/DemoRecentSS.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.transforms import ScaledTranslation

from crossing_panel import (ANNOT_PT, LABEL_PT, PANEL_LETTER_SIZE, TICK_PT, TITLE_PT,
                            figsize, save_figure)
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RecentSpatialDemography")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

COLOR_A = "0.75"
COLOR_EDGE = "0.2"
COLOR_OTHER = "0.82"
LABELS = ("weak", "intermediate", "strong")
# Half-height of the PC1 strips' y jitter, in axis units. Deliberately shallow: enough to
# show where the samples pile up, not enough to read as a second PC.
JITTER_PC1 = 1.0
JITTER_SEED = 20240517
# This figure is laid out WIDER than the manuscript's 6.5 in page: see make_figure.
FIGURE_W = 11.0
# Axis labels and tick labels on this figure only, 1.5 pt under the manuscript table: its
# PC panels are three-to-a-row beside a schematic, so they are narrower than a normal panel
# and the shared sizes crowd them.
# +1.5 pt over the earlier wide draft, which put these 1.5 pt UNDER the manuscript table:
# at 11 in the panels are wide enough to carry the table's own sizes.
LABEL_PT_LOCAL = LABEL_PT
TICK_PT_LOCAL = TICK_PT
# Scatter marker area (pt^2). Small: each panel carries several hundred samples, and at the
# previous 1.8 the B cloud read as a solid block rather than as points with a density.
MARKER_S = 2.2


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
    via a horizontal bar with a downward arrow over each descendant. The ancestral circle
    is unlabelled: its position at the head of the split says what it is, and the word did
    not fit inside it or sit clear beside it at any size the schematic can afford."""
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
            # 0.30, not 0.42: demes sit one unit apart, so at 0.42 neighbouring circles
            # nearly touch and the migration edges between them are all but hidden.
            ax.add_patch(Circle((x0 + c, r), 0.30, facecolor=grid_color(r, c, K),
                                edgecolor=COLOR_EDGE, linewidth=0.6, zorder=2))

    # A sits level with the grid's top row so both descendants hang the same distance
    # below the split bar and the two arrows read as one simultaneous event
    a_r = 1.15
    a_x, a_y = -0.6, grid_span - a_r
    ax.add_patch(Circle((a_x, a_y), a_r, facecolor=COLOR_A, edgecolor=COLOR_EDGE,
                        linewidth=0.9, zorder=2))
    ax.text(a_x, a_y, "A", ha="center", va="center", fontsize=LABEL_PT + 1.5)
    grid_x = x0 + grid_span / 2
    ax.text(grid_x, -1.3, rf"B (${K}\times{K}$ demes)", ha="center", va="center",
            fontsize=ANNOT_PT + 1.5)

    # Between the two extremes tried: grid_span + 3.4 stretched the schematic's vertical
    # span for no gain, + 1.7 left the one-line split label with no clearance under the bar.
    bar_y = grid_span + 2.4
    # The bar's right end, and so the grid's descent arrow, sits over the grid's LEFT
    # portion rather than its centre. With A pushed left too, the wedge between the two
    # arrows is wide enough to hold the split-time label without either arrow crossing it.
    split_x = x0 + grid_span * 0.62
    anc_x, anc_y = (a_x + split_x) / 2, bar_y + 2.6
    anc_r = 1.35
    ax.add_patch(Circle((anc_x, anc_y), anc_r, facecolor=COLOR_A, edgecolor=COLOR_EDGE,
                        linewidth=0.9, zorder=2))

    # split: a horizontal bar spanning A and the grid, fed by the ancestor, with one
    # downward arrow at each end onto the two descendants
    ax.plot([a_x, split_x], [bar_y, bar_y], "-", color=COLOR_EDGE, linewidth=1.0,
            zorder=1)
    ax.plot([anc_x, anc_x], [anc_y - anc_r, bar_y], "-", color=COLOR_EDGE,
            linewidth=1.0, zorder=1)
    for x, y in [(a_x, a_y + a_r), (split_x, grid_span + 0.75)]:
        ax.add_patch(FancyArrowPatch((x, bar_y), (x, y), color=COLOR_EDGE,
                                     linewidth=1.0, arrowstyle="-|>",
                                     mutation_scale=9, shrinkA=0, shrinkB=0, zorder=1))
    # In the wedge BETWEEN the two descent arrows, under the bar.
    ax.text((a_x + split_x) / 2, bar_y - 0.45, f"{t_split:g} generations ago",
            ha="center", va="top", fontsize=ANNOT_PT + 1.5)

    ax.set_xlim(a_x - a_r - 0.3, x0 + grid_span + 0.5)
    ax.set_ylim(-1.4, anc_y + anc_r + 0.4)
    # The grid's own vertical extent, in data units, so make_figure can line it up with the
    # PC panels beside it.
    ax._grid_ylim = (-0.5, grid_span + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_pc1(ax, d, K, title, rng):
    """PC1 alone on a horizontal strip, with the points jittered in y.

    A is black here (not the light grey of the PC2/PC3 panels) because PC1 is the A/B split
    itself, so which samples are A is the whole content of the strip.

    The y jitter carries NO information -- it exists so the strip shows density. Drawn at
    y=0 exactly, several hundred samples piled into one row of overlapping markers and the
    two clusters read as two solid bars of equal weight whatever their sizes. The spread is
    shallow relative to the axis so the strip still reads as one axis of variation.
    """
    is_A = d.deme_row.isna().values
    y = rng.uniform(-JITTER_PC1, JITTER_PC1, len(d))
    dB = d[~is_A]
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(dB.deme_row, dB.deme_col)]
    ax.scatter(dB.pc1, y[~is_A], s=MARKER_S, c=c, edgecolors="none", alpha=0.7, zorder=2)
    ax.scatter(d.pc1[is_A], y[is_A], s=MARKER_S, color="black", edgecolors="none",
               alpha=0.7, zorder=3)
    ax.set_xlabel("PC1", labelpad=1, fontsize=LABEL_PT_LOCAL)
    ax.tick_params(labelsize=TICK_PT_LOCAL)
    # Three ticks at full size run together on a strip this narrow.
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_yticks([])
    # Just wider than the jitter band: a taller axis leaves the band as a sliver in the
    # middle of an otherwise empty strip.
    ax.set_ylim(-2.6 * JITTER_PC1, 2.6 * JITTER_PC1)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, pad=6)


def panel_pc(ax, d, K, title=None):
    is_A = d.deme_row.isna().values
    ax.scatter(d.pc2[is_A], d.pc3[is_A], s=MARKER_S * 0.7, color=COLOR_OTHER, alpha=0.6,
              edgecolors="none", zorder=1)
    dB = d[~is_A]
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(dB.deme_row, dB.deme_col)]
    ax.scatter(dB.pc2, dB.pc3, s=MARKER_S, c=c, edgecolors="none", zorder=2)
    ax.set_xlabel("PC2", fontsize=LABEL_PT_LOCAL)
    ax.set_ylabel("PC3", fontsize=LABEL_PT_LOCAL)
    ax.tick_params(labelsize=TICK_PT_LOCAL)
    if title:
        ax.set_title(title, pad=6, fontsize=TITLE_PT)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    pc = pd.read_csv(os.path.join(analysis_dir, "pc.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("label")
    K = int(pc.deme_row.max() + 1)
    t_split = float(meta["t_split"].iloc[0]) if "t_split" in meta else 500.0
    rng = np.random.default_rng(JITTER_SEED)

    # ONE row of four: the schematic beside the three PC2-vs-PC3 scatters.
    #
    # This figure OPTS OUT of the manuscript's 6.5 in page cap (max_width_in=FIGURE_W).
    # Four panels in a row is what this comparison is -- one demography and the same plot
    # at three migration rates -- and at 6.5 in each scatter gets ~1.2 in, which is too
    # small for a 9x9 lattice of coloured points to read as a lattice. Every other figure
    # keeps the cap; the exception is declared here rather than by raising it globally.
    #
    # The schematic's column is ~1.35x a scatter's: it has to hold the 9x9 grid, a separate
    # population and the split above them.
    fig = plt.figure(figsize=figsize(FIGURE_W, 2.6, max_width_in=FIGURE_W))
    # 0.24 put panel B's "PC3" label on top of the schematic's deme grid: the gap has to
    # hold the next panel's y label and its tick numbers.
    gs = fig.add_gridspec(1, 4, wspace=0.30, width_ratios=[1.35, 1, 1, 1])
    ax_schematic = fig.add_subplot(gs[0, 0])
    panel_schematic(ax_schematic, K, t_split)

    letter_axes = [ax_schematic]
    for j, label in enumerate(LABELS, start=1):
        M = float(meta.loc[label, "M"])
        cos = float(meta.loc[label, "cos"])
        d = pc[pc.label == label]
        ax_pc = fig.add_subplot(gs[0, j])
        # Name both vectors: the OBSERVED axis of a grid sampled t_split generations after
        # the split, against the EQUILIBRIUM stepping-stone axis that stands in for
        # geographic position. The generic <u_hat, u>^2 left it ambiguous which pair of
        # vectors this panel's number refers to. t_split comes from the data, not a literal.
        panel_pc(ax_pc, d, K,
                 rf"$M={M:.2g}$" "\n"
                 rf"$\langle\hat u_{{t={t_split:g}}}, u_\infty\rangle^2 = {cos:.2f}$")
        ax_pc.set_box_aspect(1.0)      # PC2 and PC3 span a degenerate pair: equal scales
        letter_axes.append(ax_pc)

    # Line the deme grid up with the PC panels. The schematic's axes are much taller than
    # a PC panel (they also hold the ancestor and the split above the grid), and the grid
    # sits in the LOWER part of that range, so left alone it rendered well below the
    # scatters. Scale and offset the schematic's axes box so the data band the grid
    # occupies lands exactly on the PC panels' box.
    fig.canvas.draw()
    pc_box = letter_axes[1].get_position()
    sch = ax_schematic.get_position()
    y0d, y1d = ax_schematic._grid_ylim
    lo, hi = ax_schematic.get_ylim()
    f0, f1 = (y0d - lo) / (hi - lo), (y1d - lo) / (hi - lo)
    # Matching the grid to the PC box EXACTLY would make the schematic's axes tall enough
    # to hold the split above it as well, which inflates the whole row. Cap the height at
    # the row's own height and accept a grid slightly shorter than the scatters.
    height = min(pc_box.height / (f1 - f0), sch.height)
    ax_schematic.set_position([sch.x0, pc_box.y0 - f0 * height, sch.width, height])

    # One letter per PANEL: four panels in one row, so A..D.
    #
    # All four sit at ONE height, taken from the PC panels' top edge. An axes-relative
    # offset put each letter above its own axes, and the schematic's axes are taller than a
    # PC panel's (they hold the split above the grid), so A floated well above B, C and D.
    fig.canvas.draw()
    letter_y = letter_axes[1].get_position().y1
    for ax, letter in zip(letter_axes, "ABCD"):
        fig.text(ax.get_position().x0, letter_y, letter,
                 transform=fig.transFigure + ScaledTranslation(
                     -22 / 72, 16 / 72, fig.dpi_scale_trans),
                 fontsize=PANEL_LETTER_SIZE + 0.5, fontweight="bold", va="top")

    out = save_figure(fig, figure_dir, "DemoRecentSS")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
