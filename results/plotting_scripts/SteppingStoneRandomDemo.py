#%%
"""
Sampling density decides whether PCA sees spatial structure, 1D and 2D (4x3).

Migration is held FIXED for each dimension (one M per dimension, chosen so the transition
happens within the phi range), so the only thing varying across a row is how densely the
demes are sampled. Sparse sampling misses the structure that dense sampling recovers.

Columns are the three sampling fractions phi in [0.2, 0.4, 0.6]; rows are

    row 1  the 1D sampling scheme: a chain of K circles connected by the migration graph,
           colored by chain position (viridis), sampled demes at full alpha and unsampled
           faded
    row 2  1D PC1 for that scheme, colored the same way
    row 3  the 2D sampling scheme: a K x K grid, colored by grid position
    row 4  2D PC1 vs PC2 for that scheme, on a square panel

Colors: 1D uses viridis along the chain; 2D uses the bivariate (row, col) colormap shared
with SteppingStoneEigenvectorShapes and RecentSpatialDemography, so grid position maps to
color the same way across all three figures. The analysis script fixes each PC's sign (1D)
and in-plane rotation (2D) so PC1 increases with chain position / column and PC2 with row,
i.e. the scatter colors run the same way as the schematic above them.

Reads results/analysis/SteppingStoneRandomDemo, writes
results/figures/SteppingStoneRandomDemo.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Circle

from plotting_style import setup_plotting_style
from RecentSpatialDemography import grid_color

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SteppingStoneRandomDemo")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

COLOR_EDGE = "0.2"
CMAP_1D = cm.viridis
ALPHA_SAMPLED = 0.95
ALPHA_UNSAMPLED = 0.15


def chain_color(deme, K):
    """viridis along the 1D chain -- the counterpart of grid_color for one dimension."""
    return CMAP_1D(deme / max(K - 1, 1))


def _blank_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scheme_1d(ax, demes, K):
    """A chain of K circles, connected by lines, colored by deme index, faded if unsampled."""
    ax.plot([0, K - 1], [0, 0], "-", color=COLOR_EDGE, linewidth=1.0, zorder=1)
    for x in range(K):
        alpha = ALPHA_SAMPLED if bool(demes[demes.deme == x].sampled.iloc[0]) else ALPHA_UNSAMPLED
        ax.add_patch(Circle((x, 0), 0.42, facecolor=chain_color(x, K), edgecolor=COLOR_EDGE,
                            linewidth=0.9, alpha=alpha, zorder=2))
    ax.set_xlim(-0.7, K - 0.3)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    _blank_axes(ax)


def scheme_2d(ax, demes, K):
    """A K x K grid of circles, nearest-neighbour lines, colored by grid position, faded
    if unsampled."""
    for r in range(K):
        for c in range(K):
            if c + 1 < K:
                ax.plot([c, c + 1], [r, r], "-", color=COLOR_EDGE, linewidth=0.8, zorder=1)
            if r + 1 < K:
                ax.plot([c, c], [r, r + 1], "-", color=COLOR_EDGE, linewidth=0.8, zorder=1)
    for r in range(K):
        for c in range(K):
            sampled = bool(demes[(demes.row == r) & (demes.col == c)].sampled.iloc[0])
            alpha = ALPHA_SAMPLED if sampled else ALPHA_UNSAMPLED
            ax.add_patch(Circle((c, r), 0.40, facecolor=grid_color(r, c, K),
                                edgecolor=COLOR_EDGE, linewidth=0.7, alpha=alpha, zorder=2))
    pad = 0.45
    ax.set_xlim(-pad, K - 1 + pad)
    ax.set_ylim(-pad, K - 1 + pad)
    ax.set_aspect("equal")
    _blank_axes(ax)


def pc_panel_1d(ax, d, K):
    """PC1 on x only -- the y axis (and its spine) carries no information."""
    c = [chain_color(int(j), K) for j in d.deme]
    ax.scatter(d.pc1, np.zeros(len(d)), c=c, s=10, edgecolors="none", alpha=0.85)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("PC1")


def pc_panel_2d(ax, d, K):
    """PC1 vs PC2 on a square panel: the two PCs span a degenerate eigenvalue pair, so
    equal data ranges on both axes keep the recovered grid from looking stretched."""
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(d.row, d.col)]
    ax.scatter(d.pc1, d.pc2, c=c, s=10, edgecolors="none", alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    lim = 1.08 * max(np.abs(d.pc1).max(), np.abs(d.pc2).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)


def _section_title(fig, row_axes, text, pad):
    """A title spanning one section, placed `pad` (figure fraction) above that section's
    column titles."""
    boxes = [ax.get_position() for ax in row_axes]
    x = 0.5 * (min(b.x0 for b in boxes) + max(b.x1 for b in boxes))
    y = max(b.y1 for b in boxes) + pad
    fig.text(x, y, text, ha="center", va="bottom", fontsize=12)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    demes = pd.read_csv(os.path.join(analysis_dir, "sampled_demes.csv"))
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("dimension")

    phis = sorted(pcs.phi.unique())
    K_1d = int(demes[demes.dimension == 1].deme.max()) + 1
    K_2d = int(np.sqrt(int(demes[demes.dimension == 2].deme.max()) + 1))

    # Two independent sub-gridspecs rather than one 4-row grid, so the 1D and 2D sections
    # can be separated by a real gap that leaves room for the 2D section title.
    # The schematic rows are aspect-locked, so their height ratios are set to roughly the
    # geometry's own aspect at this column width: a 25-deme chain is very wide and flat, a
    # 9x9 grid is square. Anything taller just adds dead space around them.
    fig = plt.figure(figsize=(11.0, 10.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.44, 2.0], hspace=0.32)
    gs_1d = outer[0].subgridspec(2, len(phis), height_ratios=[0.14, 0.30], hspace=0.05)
    gs_2d = outer[1].subgridspec(2, len(phis), height_ratios=[1.0, 1.0], hspace=0.20)
    axes = np.array([[fig.add_subplot(gs_1d[0, j]) for j in range(len(phis))],
                     [fig.add_subplot(gs_1d[1, j]) for j in range(len(phis))],
                     [fig.add_subplot(gs_2d[0, j]) for j in range(len(phis))],
                     [fig.add_subplot(gs_2d[1, j]) for j in range(len(phis))]])

    for j, phi in enumerate(phis):
        d1 = demes[(demes.dimension == 1) & (demes.phi == phi)]
        d2 = demes[(demes.dimension == 2) & (demes.phi == phi)]
        p1 = pcs[(pcs.dimension == 1) & (pcs.phi == phi)]
        p2 = pcs[(pcs.dimension == 2) & (pcs.phi == phi)]

        scheme_1d(axes[0, j], d1, K_1d)
        pc_panel_1d(axes[1, j], p1, K_1d)
        scheme_2d(axes[2, j], d2, K_2d)
        pc_panel_2d(axes[3, j], p2, K_2d)

        # Each section gets its own column titles: the two geometries have different deme
        # counts, so the same phi means a different number of sampled demes in each.
        axes[0, j].set_title(rf"$\phi={phi:g}$  ({int(d1.sampled.sum())}/{K_1d} demes)",
                             fontsize=10)
        axes[2, j].set_title(rf"$\phi={phi:g}$  ({int(d2.sampled.sum())}/{K_2d * K_2d} demes)",
                             fontsize=10)

    axes[0, 0].set_ylabel("sampling", fontsize=10)
    axes[1, 0].set_ylabel("PC1", fontsize=10)
    axes[2, 0].set_ylabel("sampling", fontsize=10)

    M_1d, M_2d = meta.loc[1, "M"], meta.loc[2, "M"]
    _section_title(fig, axes[0, :],
                   rf"1D stepping-stones (${K_1d}$ demes, $M={M_1d:.0f}$)", pad=0.025)
    _section_title(fig, axes[2, :],
                   rf"2D stepping-stones (${K_2d} \times {K_2d}$ demes, "
                   rf"$M={M_2d:.0f}$)", pad=0.025)

    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneRandomDemo.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
