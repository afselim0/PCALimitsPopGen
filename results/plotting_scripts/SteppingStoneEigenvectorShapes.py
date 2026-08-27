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
with RecentSpatialDemography, so grid position maps to color the same way in both figures.

Reads results/analysis/SteppingStoneEigenvectorShapes, writes
results/figures/SteppingStoneEigenvectorShapes.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.patches import Circle

from plotting_style import setup_plotting_style
from RecentSpatialDemography import grid_color

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


def chain_color(deme, K):
    """viridis along the 1D chain -- the counterpart of grid_color for one dimension."""
    return CMAP_1D(deme / max(K - 1, 1))


def panel_schematic_1d(ax, K):
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
    ax.text((K - 1) / 2, -1.1, f"$K = {K}$ demes", ha="center", va="top", fontsize=9)
    _blank_axes(ax)
    ax.set_title("demes")


def panel_schematic_2d(ax, K):
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
    ax.text((K - 1) / 2, -0.9, rf"$K \times K = {K} \times {K}$ demes",
            ha="center", va="top", fontsize=9)
    _blank_axes(ax)
    ax.set_title("demes")


def _blank_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_1d(ax, d, meta_row, K, rng):
    """PC1 on x, uniform jitter on y -- the y axis carries no information."""
    y = rng.uniform(0, 1, len(d))
    c = [chain_color(j, K) for j in d.deme]
    ax.scatter(d.pc1, y, c=c, s=8, edgecolors="none", alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(rf"$M={meta_row.M:.3g}$, cosine similarity$={meta_row.median_cos:.2f}$")


def panel_2d(ax, d, meta_row, K):
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(d.row, d.col)]
    ax.scatter(d.pc1, d.pc2, c=c, s=8, edgecolors="none", alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(rf"$M={meta_row.M:.3g}$, cosine similarity$={meta_row.median_cos:.2f}$")


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index(["dimension", "regime"])
    rng = np.random.default_rng(JITTER_SEED)

    K_1d = int(meta.loc[(1, REGIMES[0])].K)
    K_2d = int(meta.loc[(2, REGIMES[0])].K)
    n_1d = int(meta.loc[(1, REGIMES[0])].n_per_deme)
    n_2d = int(meta.loc[(2, REGIMES[0])].n_per_deme)

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.2),
                             gridspec_kw=dict(height_ratios=[1.0, 1.5]))
    panel_schematic_1d(axes[0, 0], K_1d)
    panel_schematic_2d(axes[1, 0], K_2d)

    for j, regime in enumerate(REGIMES, start=1):
        d1 = pcs[(pcs.dimension == 1) & (pcs.regime == regime)]
        panel_1d(axes[0, j], d1, meta.loc[(1, regime)], K_1d, rng)
        axes[0, j].set_title(f"{REGIME_TITLES[regime]}\n" + axes[0, j].get_title())

        d2 = pcs[(pcs.dimension == 2) & (pcs.regime == regime)]
        panel_2d(axes[1, j], d2, meta.loc[(2, regime)], K_2d)

    axes[0, 1].set_ylabel(rf"1D ($K={K_1d}$, $n={n_1d}$)" + "\njitter")
    axes[1, 1].set_ylabel(rf"2D ($K={K_2d}$, $n={n_2d}$)" + "\nPC2")

    for i, letter in enumerate("AB"):
        axes[i, 0].text(-0.05, 1.05, letter, transform=axes[i, 0].transAxes,
                        fontsize=13, fontweight="bold", va="bottom")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneEigenvectorShapes.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
