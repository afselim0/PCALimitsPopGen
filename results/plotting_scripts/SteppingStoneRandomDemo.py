#%%
"""
Random deme sampling demonstrated, 1D and 2D (2 blocks of 2x3, total 2x6).

Each block (1D top, 2D bottom) has 3 columns, one per sampling fraction phi in
[0.2, 0.4, 0.6]:

    column 1  the sampling scheme: demes as circles (chain for 1D, grid for 2D),
              connected by lines showing the migration graph, sampled demes filled grey
    column 2  PC1 (1D) or PC1 vs PC2 (2D) for the samples actually drawn at that phi,
              n_per_deme = 8, colored by deme index

Reads results/analysis/SteppingStoneRandomDemo, writes
results/figures/SteppingStoneRandomDemo.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.patches import Circle

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SteppingStoneRandomDemo")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

COLOR_SAMPLED = "0.55"
COLOR_UNSAMPLED = "white"
COLOR_EDGE = "0.2"
CMAP = cm.viridis


def scheme_1d(ax, demes, K, phi):
    """A chain of K circles, connected by lines, sampled ones filled grey."""
    xs = np.arange(K)
    for x in xs[:-1]:
        ax.plot([x, x + 1], [0, 0], "-", color=COLOR_EDGE, linewidth=1.0, zorder=1)
    for x in xs:
        sampled = bool(demes[demes.deme == x].sampled.iloc[0])
        color = COLOR_SAMPLED if sampled else COLOR_UNSAMPLED
        ax.add_patch(Circle((x, 0), 0.32, facecolor=color, edgecolor=COLOR_EDGE,
                            linewidth=0.9, zorder=2))
    ax.set_xlim(-0.7, K - 0.3)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(rf"$\phi={phi:g}$")


def scheme_2d(ax, demes, K, phi):
    """A K x K grid of circles, nearest-neighbour lines, sampled ones filled grey."""
    for r in range(K):
        for c in range(K):
            if c + 1 < K:
                ax.plot([c, c + 1], [r, r], "-", color=COLOR_EDGE, linewidth=0.8, zorder=1)
            if r + 1 < K:
                ax.plot([c, c], [r, r + 1], "-", color=COLOR_EDGE, linewidth=0.8, zorder=1)
    for r in range(K):
        for c in range(K):
            sampled = bool(demes[(demes.row == r) & (demes.col == c)].sampled.iloc[0])
            color = COLOR_SAMPLED if sampled else COLOR_UNSAMPLED
            ax.add_patch(Circle((c, r), 0.32, facecolor=color, edgecolor=COLOR_EDGE,
                                linewidth=0.7, zorder=2))
    ax.set_xlim(-0.7, K - 0.3)
    ax.set_ylim(-0.7, K - 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(rf"$\phi={phi:g}$")


def pc_panel_1d(ax, d, K, norm):
    ax.scatter(d.pc1, np.zeros(len(d)), c=d.deme, cmap=CMAP, norm=norm, s=10,
              edgecolors='none')
    ax.set_yticks([])
    ax.set_xlabel("PC1")


def pc_panel_2d(ax, d, K_demes, norm):
    ax.scatter(d.pc1, d.pc2, c=d.deme, cmap=CMAP, norm=norm, s=10, edgecolors='none')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    demes = pd.read_csv(os.path.join(analysis_dir, "sampled_demes.csv"))
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("dimension")

    phis = sorted(pcs.phi.unique())
    K_1d = int(meta.loc[1, "K"])
    K_2d = int(meta.loc[2, "K"])
    norm_1d = colors.Normalize(vmin=0, vmax=K_1d - 1)
    norm_2d = colors.Normalize(vmin=0, vmax=K_2d * K_2d - 1)

    fig, axes = plt.subplots(2, 6, figsize=(17.0, 5.6),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 1]))

    for j, phi in enumerate(phis):
        d1 = demes[(demes.dimension == 1) & (demes.phi == phi)]
        scheme_1d(axes[0, j], d1, K_1d, phi)
        p1 = pcs[(pcs.dimension == 1) & (pcs.phi == phi)]
        pc_panel_1d(axes[0, j + 3], p1, K_1d, norm_1d)

        d2 = demes[(demes.dimension == 2) & (demes.phi == phi)]
        scheme_2d(axes[1, j], d2, K_2d, phi)
        p2 = pcs[(pcs.dimension == 2) & (pcs.phi == phi)]
        pc_panel_2d(axes[1, j + 3], p2, K_2d * K_2d, norm_2d)

    axes[0, 0].set_ylabel(f"1D  (K={K_1d})\nsampling scheme", fontsize=10)
    axes[1, 0].set_ylabel(f"2D  (K={K_2d})\nsampling scheme", fontsize=10)

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneRandomDemo.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
