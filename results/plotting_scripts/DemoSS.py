#%%
"""
Sample PC shapes at three regimes -- no structure, weak, strong -- 1D and 2D (2x4).

Column 1 (schematic): the deme geometry itself -- a single-line chain of K circles (1D)
and a K x K grid of circles (2D), each circle filled with the same color used for its
samples in the PC panels, so the schematic doubles as the legend for which color is which
deme. The analysis script fixes each PC's sign (1D) and in-plane rotation (2D) so that the
PCs increase with deme index / row / column, i.e. the colors run the same way here as in
the scatters.
Row 1 (1D): PC1 on the x axis against uniform random jitter on the y axis. The y axis
carries no information; plotting PC1 against deme index instead would draw the chain
position twice (once as the coordinate, once as the color) and hide how the samples pile
up, whereas jitter shows the marginal distribution of PC1 directly.
Row 2 (2D): PC1 vs PC2 scatter (procrustes-aligned to the degenerate (1,0)/(0,1)
closed-form pair beforehand, in the analysis script, so PC1/PC2 here are oriented
consistently with the theoretical axes rather than by an arbitrary in-plane rotation).

Columns 2-4 are the three regimes, selected in the analysis script by the MEASURED
overlap: no structure (overlap ~ 0), weak (~ 0.5), strong (the largest M first reaching
0.9).

Colors: 1D uses viridis along the chain; 2D uses the bivariate (row, col) colormap shared
with DemoRecentSS, so grid position maps to color the same way in both figures.

Reads results/analysis/SteppingStoneEigenvectorShapes, writes
results/figures/DemoSS.pdf
"""
import os

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.patches import Circle

from crossing_panel import ANNOT_PT, PANEL_LETTER_SIZE, TITLE_PT, figsize, save_figure
from plotting_style import setup_plotting_style
from DemoRecentSS import grid_color

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis",
                            "SteppingStoneEigenvectorShapes")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

REGIMES = ["no_structure", "weak", "strong"]
REGIME_TITLES = {"no_structure": "no structure", "weak": "weak structure",
                 "strong": "strong structure"}
CMAP_1D = cm.viridis
COLOR_EDGE = "0.2"
JITTER_SEED = 0


# Half-height of the PC1 jitter band, in units of the (unlabelled) y axis, whose limits
# are +/-1. Small on purpose -- see panel_1d.
JITTER_1D = 0.16


def _fmt_M(v):
    """M as a plain integer when it is one, else LaTeX 'm x 10^e' -- never '3.2e+03'."""
    v = float(v)
    if v >= 1000:
        e = int(np.floor(np.log10(v)))
        return rf"{v / 10.0 ** e:.1f} \times 10^{{{e}}}"
    return f"{v:.0f}" if v == int(v) else f"{v:.3g}"


def chain_color(deme, K):
    """viridis along the 1D chain -- the counterpart of grid_color for one dimension."""
    return CMAP_1D(deme / max(K - 1, 1))


def panel_schematic_1d(ax, K, n):
    """The K-deme chain on a single line. The circles are sized to the available width
    (radius ~ half the deme spacing) rather than to a fixed radius, so the whole chain
    fits one row at the same panel width as the 2D grid."""
    radius = 0.42
    ax.plot([0, K - 1], [0, 0], "-", color=COLOR_EDGE, linewidth=0.7, zorder=1)
    for j in range(K):
        ax.add_patch(Circle((j, 0), radius, facecolor=chain_color(j, K),
                            edgecolor=COLOR_EDGE, linewidth=0.4, zorder=2))
    ax.set_xlim(-1.0, K)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.text((K - 1) / 2, -1.1, f"$K = {K}$ demes\n$n = {n}$ samples per deme",
            ha="center", va="top", fontsize=ANNOT_PT)
    _blank_axes(ax)
    ax.set_title("1D stepping stones")


def panel_schematic_2d(ax, K, n):
    """K x K circles on a nearest-neighbour grid, filled by the bivariate grid color."""
    for r in range(K):
        for c in range(K):
            if c + 1 < K:
                ax.plot([c, c + 1], [r, r], "-", color=COLOR_EDGE, linewidth=0.8, zorder=1)
            if r + 1 < K:
                ax.plot([c, c], [r, r + 1], "-", color=COLOR_EDGE, linewidth=0.8, zorder=1)
    for r in range(K):
        for c in range(K):
            ax.add_patch(Circle((c, r), 0.30, facecolor=grid_color(r, c, K),
                                edgecolor=COLOR_EDGE, linewidth=0.5, zorder=2))
    ax.set_xlim(-0.7, K - 0.3)
    ax.set_ylim(-1.6, K - 0.3)
    ax.set_aspect("equal")
    ax.text((K - 1) / 2, -0.9, rf"$K \times K = {K} \times {K}$ demes" "\n" rf"$n = {n}$ samples per deme",
            ha="center", va="top", fontsize=ANNOT_PT)
    _blank_axes(ax)
    ax.set_title("2D stepping stones")


def _blank_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_1d(ax, d, meta_row, K, rng):
    """
    PC1 on x, a thin band of jitter on y -- the y axis carries no information.

    The jitter is deliberately shallow (JITTER_1D of the panel height): spread over a full
    axis it reads as a second PC, which is exactly the wrong thing for a panel showing a
    single one-dimensional axis. Enough to separate overlapping points, not enough to look
    like a dimension.
    """
    y = rng.uniform(-JITTER_1D, JITTER_1D, len(d))
    c = [chain_color(j, K) for j in d.deme]
    ax.scatter(d.pc1, y, c=c, s=2.5, edgecolors="none", alpha=0.85)
    ax.set_xlabel("PC1")
    # Just wider than the jitter band, not +/-1: the points span only JITTER_1D of the
    # axis, so a full-height axis left ~80% of the row blank and stranded the title far
    # above the data it describes.
    ax.set_ylim(-3.4 * JITTER_1D, 3.4 * JITTER_1D)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(rf"$M={_fmt_M(meta_row.M)}$" "\n"
                 rf"$\langle\hat u,u\rangle^2 = {meta_row.median_cos:.2f}$")


def panel_2d(ax, d, meta_row, K):
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(d.row, d.col)]
    ax.scatter(d.pc1, d.pc2, c=c, s=2.5, edgecolors="none", alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # equal DATA scale (PC1 and PC2 are the same kind of quantity, and the 2D lattice only
    # reads as a lattice when one unit of PC1 is one unit of PC2), in a SQUARE box. Without
    # the box aspect the panel took whatever shape the gridspec cell had -- 1.0 x 1.4 in,
    # so the square lattice sat in a portrait frame with empty margins above and below.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)
    ax.set_title(rf"$M={_fmt_M(meta_row.M)}$" "\n"
                 rf"$\langle\hat u,u\rangle^2 = {meta_row.median_cos:.2f}$")


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index(["dimension", "regime"])
    rng = np.random.default_rng(JITTER_SEED)

    K_1d = int(meta.loc[(1, REGIMES[0])].K)
    K_2d = int(meta.loc[(2, REGIMES[0])].K)
    n_1d = int(meta.loc[(1, REGIMES[0])].n_per_deme)
    n_2d = int(meta.loc[(2, REGIMES[0])].n_per_deme)

    # All four columns share a width. The schematic column carries no axes, so it was
    # previously narrowed to 0.75, but that left the 2D grid visibly smaller than the PC
    # panels beside it and the two rows' schematics on different scales.
    # hspace was 1.45, which left a band of empty page as tall as the 1D row between the
    # two blocks: the 1D PC strips fill only the middle of their cell (their y limits are
    # just wider than the jitter band), so most of that generous gap was already blank.
    # Height set so the 2D row's cell is as tall as a panel is wide: those panels are
    # square (set_box_aspect(1) in panel_2d), and any excess cell height would only open
    # blank page above and below them rather than making them bigger.
    fig, axes = plt.subplots(2, 4, figsize=figsize(6.5, 3.35),
                             gridspec_kw=dict(height_ratios=[0.80, 1.6],
                                              wspace=0.62, hspace=0.95))
    panel_schematic_1d(axes[0, 0], K_1d, n_1d)
    panel_schematic_2d(axes[1, 0], K_2d, n_2d)

    for j, regime in enumerate(REGIMES, start=1):
        d1 = pcs[(pcs.dimension == 1) & (pcs.regime == regime)]
        panel_1d(axes[0, j], d1, meta.loc[(1, regime)], K_1d, rng)

        d2 = pcs[(pcs.dimension == 2) & (pcs.regime == regime)]
        panel_2d(axes[1, j], d2, meta.loc[(2, regime)], K_2d)

    # The row name goes on the SCHEMATIC panel (which has no axes of its own), not on the
    # first PC panel: as a ylabel there it was drawn across the scatter, and in row 2 it
    # collided with that panel's real "PC2" label.
    for r in (0, 1):
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])
    axes[1, 1].set_ylabel("PC2")

    # The row heading stays CENTRED over its schematic, and the panel letter is placed
    # beside it in the left title slot. Both slots sit at the same height, so the letter
    # lines up with the heading without measuring either.
    #
    # Both are then raised to the height of the M titles beside them. Those are TWO lines
    # ("M = ...", then the cosine similarity), so their first line sits a line higher than
    # an ordinary one-line title; without this the letter and heading read as belonging
    # below the row they label. The pad is expressed in points off the type table, so it
    # tracks the title size rather than being a fixed nudge.
    pad = plt.rcParams["axes.titlepad"] + TITLE_PT * 1.35
    for i, letter in enumerate("AB"):
        ax = axes[i, 0]
        ax.set_title(ax.get_title(), pad=pad)
        # The letter is placed by OFFSET from the axes' top-left corner, not in the left
        # title slot: both slots share one baseline, and the centred heading is wide enough
        # that its left edge runs back into the left slot and the two overprint.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -20 / 72, (pad + 2) / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="baseline")

    # Explicit margins, not tight_layout: the schematic panels have a fixed aspect and
    # equal-scaled axes, which tight_layout cannot lay out (it warns and mis-places the
    # row), and the gridspec already sets the inter-panel spacing.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.10)
    out = save_figure(fig, figure_dir, "DemoSS")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
