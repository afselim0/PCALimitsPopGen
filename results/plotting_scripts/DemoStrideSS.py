#%%
"""
Uniform deme dilution and what PCA still sees, 1D and 2D (4x3).

The counterpart of DemoRandomSS. There the deme subset is a random draw; here
it is every p-th deme of the chain (1D) or every p-th row AND column of the grid (2D), at
a total sample size held FIXED across p. Migration is held fixed too (one M per
dimension), so the only thing varying across a row is how far apart the sampled demes sit
-- the samples freed by dropping demes go to the demes that remain.

Columns are the three strides p; rows are

    row 1  the 1D sampling scheme: a chain of K circles connected by the migration graph,
           colored by chain position (viridis), sampled demes at full alpha and unsampled
           faded
    row 2  1D PC1 for that scheme, colored the same way, titled with its cosine
           similarity to the closed-form spatial axis restricted to the sampled demes
    row 3  the 2D sampling scheme: a K x K grid, colored by grid position
    row 4  2D PC1 vs PC2 for that scheme, on a square panel, titled with the mean
           principal cosine to the restricted closed-form degenerate pair

The schematic and PC drawing are shared verbatim with DemoRandomSS (imported,
not copied), so the two figures are read the same way: 1D uses viridis along the chain, 2D
the bivariate (row, col) colormap shared with DemoSS and
DemoRecentSS.

Reads results/analysis/SteppingStoneStrideDemo, writes
results/figures/DemoStrideSS.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crossing_panel import TITLE_PT, figsize, save_figure
from plotting_style import setup_plotting_style
import numpy as np
from matplotlib import cm
from matplotlib.patches import Circle
from crossing_panel import PANEL_LETTER_SIZE
from DemoRecentSS import grid_color

# Helpers inlined from the pre-GENETICS DemoRandomSS, which no longer provides them.
COLOR_EDGE = "0.2"
CMAP_1D = cm.viridis
ALPHA_SAMPLED = 0.95
ALPHA_UNSAMPLED = 0.15


# Half-height of the PC1 jitter band on a +/-1 y axis. Small on purpose -- see pc_panel_1d.
JITTER_1D = 0.16


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


def pc_panel_1d(ax, d, K, rng=None):
    """
    PC1 on x, with a shallow jitter on y -- the y axis carries no information.

    Without the jitter the samples of one deme land on a single point and the density along
    PC1 is unreadable. The band is deliberately thin (JITTER_1D of a +/-1 axis) so it
    separates points without reading as a second PC.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    c = [chain_color(int(j), K) for j in d.deme]
    y = rng.uniform(-JITTER_1D, JITTER_1D, len(d))
    ax.scatter(d.pc1, y, c=c, s=2.5, edgecolors="none", alpha=0.85)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("PC1")


def pc_panel_2d(ax, d, K):
    """PC1 vs PC2 on a square panel: the two PCs span a degenerate eigenvalue pair, so
    equal data ranges on both axes keep the recovered grid from looking stretched."""
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(d.row, d.col)]
    ax.scatter(d.pc1, d.pc2, c=c, s=2.5, edgecolors="none", alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    lim = 1.08 * max(np.abs(d.pc1).max(), np.abs(d.pc2).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)


def _section_title(fig, row_axes, text, pad, letter=None):
    """
    A title spanning one section, placed clear of that section's column titles.

    `letter` labels the whole block (its two rows), drawn bold at the LEFT edge of the
    section while the title itself stays centred and NOT bold -- the panel letter is the
    only type in the figure set that is bold, so a bold section title competed with it.

    The clearance is MEASURED from the rendered column titles, not assumed: `pad` is only
    the extra gap left above them. A fixed figure-fraction offset is not a fixed amount of
    page -- the same 0.025 that cleared the column titles on the old 10.5 in-tall canvas is
    a third as much ink on the 6.5 in-wide page, and the section title landed on top of
    them.
    """
    fig.canvas.draw()                      # titles have no extent until the figure renders
    inv = fig.transFigure.inverted()
    boxes = [ax.get_position() for ax in row_axes]
    x = 0.5 * (min(b.x0 for b in boxes) + max(b.x1 for b in boxes))
    top = max(b.y1 for b in boxes)
    for ax in row_axes:
        if ax.get_visible() and ax.get_title():
            bb = inv.transform_bbox(ax.title.get_window_extent(fig.canvas.get_renderer()))
            top = max(top, bb.y1)
    fig.text(x, top + pad, text, ha="center", va="bottom", fontsize=TITLE_PT)
    if letter:
        fig.text(min(b.x0 for b in boxes) - 0.035, top + pad, letter, ha="left",
                 va="bottom", fontsize=PANEL_LETTER_SIZE, fontweight="bold")



_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SteppingStoneStrideDemo")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    demes = pd.read_csv(os.path.join(analysis_dir, "sampled_demes.csv"))
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv"))

    # The two dimensions run their own stride grids (2D skips p=3, see the analysis
    # script), so the columns are indexed by position rather than by a shared p value.
    p_1d = sorted(pcs[pcs.dimension == 1].p.unique())
    p_2d = sorted(pcs[pcs.dimension == 2].p.unique())
    n_col = max(len(p_1d), len(p_2d))
    K_1d = int(demes[demes.dimension == 1].deme.max()) + 1
    K_2d = int(np.sqrt(int(demes[demes.dimension == 2].deme.max()) + 1))

    # Two independent sub-gridspecs rather than one 4-row grid, so the 1D and 2D sections
    # can be separated by a real gap that leaves room for the 2D section title. The
    # schematic rows are aspect-locked, so their height ratios follow the geometry's own
    # aspect at this column width: a 25-deme chain is wide and flat, a 9x9 grid is square.
    fig = plt.figure(figsize=figsize(11.0, 10.5))
    # The 1D section is given real height (0.70, not 0.44): its chain schematic is
    # aspect-locked and takes only a sliver, so a short section starves the PC row below
    # it, which is the panel carrying the data.
    outer = fig.add_gridspec(2, 1, height_ratios=[0.70, 2.0], hspace=0.62)
    gs_1d = outer[0].subgridspec(2, n_col, height_ratios=[0.12, 0.52], hspace=0.55)
    gs_2d = outer[1].subgridspec(2, n_col, height_ratios=[1.0, 1.0], hspace=0.22)
    axes = np.array([[fig.add_subplot(gs_1d[0, j]) for j in range(n_col)],
                     [fig.add_subplot(gs_1d[1, j]) for j in range(n_col)],
                     [fig.add_subplot(gs_2d[0, j]) for j in range(n_col)],
                     [fig.add_subplot(gs_2d[1, j]) for j in range(n_col)]])

    for j in range(n_col):
        for dimension, p_values, scheme, pc_panel, K, rows in (
                (1, p_1d, scheme_1d, pc_panel_1d, K_1d, (0, 1)),
                (2, p_2d, scheme_2d, pc_panel_2d, K_2d, (2, 3))):
            r_scheme, r_pc = rows
            if j >= len(p_values):
                axes[r_scheme, j].set_visible(False)
                axes[r_pc, j].set_visible(False)
                continue
            p = p_values[j]
            d = demes[(demes.dimension == dimension) & (demes.p == p)]
            q = pcs[(pcs.dimension == dimension) & (pcs.p == p)]
            m = meta[(meta.dimension == dimension) & (meta.p == p)].iloc[0]

            scheme(axes[r_scheme, j], d, K)
            pc_panel(axes[r_pc, j], q, K)
            # How well the PC panel recovers the spatial axis, on the panel itself.
            axes[r_pc, j].set_title(
                rf"$\langle\hat u,u\rangle^2 = {q['cos'].iloc[0]:.2f}$")
            # n-per-deme is derived from the fixed total N, so it is stated per column:
            # it is the thing that grows as coverage thins.
            axes[r_scheme, j].set_title(
                rf"$p={p:g}$  ({int(m.n_demes_sampled)} demes)" "\n"
                rf"$n={int(m.n_per_deme)}$ each, $N={int(m.N)}$")

    M_1d = float(meta[meta.dimension == 1].M.iloc[0])
    M_2d = float(meta[meta.dimension == 2].M.iloc[0])
    # n-per-deme is NOT in the block title here, unlike DemoRandomSS: total N is held
    # fixed across p, so n grows as coverage thins and is stated per column instead.
    _section_title(fig, axes[0, :],
                   rf"1D stepping stones (${K_1d}$ demes, $M={M_1d:.0f}$)",
                   pad=0.025, letter="A")
    _section_title(fig, axes[2, :],
                   rf"2D stepping stones (${K_2d} \times {K_2d}$ demes, "
                   rf"$M={M_2d:.0f}$)", pad=0.025, letter="B")

    out = save_figure(fig, figure_dir, "DemoStrideSS")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
