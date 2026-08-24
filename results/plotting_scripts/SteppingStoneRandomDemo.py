#%%
"""
Random deme sampling demonstrated, 1D and 2D (6x3).

Six rows, one per (dimension, phi): 1D phi in [0.2, 0.4, 0.6], then 2D phi in
[0.2, 0.4, 0.6]. Three columns:

    column 1  the sampling scheme: demes as circles (chain for 1D, grid for 2D),
              connected by lines showing the migration graph, colored by deme index
              (viridis), sampled demes at full alpha and unsampled demes faded
    column 2  PC1 (1D) or PC1 vs PC2 (2D) at M near M_c (structure visible but not
              saturated), colored by deme index
    column 3  same, at M well above M_c (effectively no structure)

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

COLOR_EDGE = "0.2"
CMAP = cm.viridis
ALPHA_SAMPLED = 0.95
ALPHA_UNSAMPLED = 0.15
M_LEVELS = ["near_Mc", "high"]
M_LEVEL_TITLES = {"near_Mc": r"$M \approx M_c$", "high": r"$M \gg M_c$"}


def scheme_1d(ax, demes, K, phi, norm):
    """A chain of K circles, connected by lines, colored by deme index, faded if unsampled."""
    xs = np.arange(K)
    for x in xs[:-1]:
        ax.plot([x, x + 1], [0, 0], "-", color=COLOR_EDGE, linewidth=1.0, zorder=1)
    for x in xs:
        sampled = bool(demes[demes.deme == x].sampled.iloc[0])
        alpha = ALPHA_SAMPLED if sampled else ALPHA_UNSAMPLED
        ax.add_patch(Circle((x, 0), 0.32, facecolor=CMAP(norm(x)), edgecolor=COLOR_EDGE,
                            linewidth=0.9, alpha=alpha, zorder=2))
    ax.set_xlim(-0.7, K - 0.3)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scheme_2d(ax, demes, K, phi, norm):
    """A K x K grid of circles, nearest-neighbour lines, colored by deme index, faded if unsampled."""
    for r in range(K):
        for c in range(K):
            if c + 1 < K:
                ax.plot([c, c + 1], [r, r], "-", color=COLOR_EDGE, linewidth=0.8, zorder=1)
            if r + 1 < K:
                ax.plot([c, c], [r, r + 1], "-", color=COLOR_EDGE, linewidth=0.8, zorder=1)
    for r in range(K):
        for c in range(K):
            deme_idx = r * K + c
            sampled = bool(demes[(demes.row == r) & (demes.col == c)].sampled.iloc[0])
            alpha = ALPHA_SAMPLED if sampled else ALPHA_UNSAMPLED
            ax.add_patch(Circle((c, r), 0.32, facecolor=CMAP(norm(deme_idx)),
                                edgecolor=COLOR_EDGE, linewidth=0.7, alpha=alpha, zorder=2))
    ax.set_xlim(-0.7, K - 0.3)
    ax.set_ylim(-0.7, K - 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def pc_panel_1d(ax, d, norm):
    ax.scatter(d.pc1, np.zeros(len(d)), c=d.deme, cmap=CMAP, norm=norm, s=10,
              edgecolors='none')
    ax.set_yticks([])
    ax.set_xlabel("PC1")


def pc_panel_2d(ax, d, norm):
    ax.scatter(d.pc1, d.pc2, c=d.deme, cmap=CMAP, norm=norm, s=10, edgecolors='none')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    demes = pd.read_csv(os.path.join(analysis_dir, "sampled_demes.csv"))
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index(["dimension", "m_level"])

    phis = sorted(pcs.phi.unique())
    K_1d = int(demes[demes.dimension == 1].deme.max()) + 1
    K_2d_side = int(np.sqrt(int(demes[demes.dimension == 2].deme.max()) + 1))
    norm_1d = colors.Normalize(vmin=0, vmax=K_1d - 1)
    norm_2d = colors.Normalize(vmin=0, vmax=K_2d_side * K_2d_side - 1)

    rows = [(1, phi) for phi in phis] + [(2, phi) for phi in phis]
    fig, axes = plt.subplots(6, 3, figsize=(9.0, 16.5))

    for i, (dimension, phi) in enumerate(rows):
        d_demes = demes[(demes.dimension == dimension) & (demes.phi == phi)]
        K = K_1d if dimension == 1 else K_2d_side
        norm = norm_1d if dimension == 1 else norm_2d
        scheme_func = scheme_1d if dimension == 1 else scheme_2d
        pc_func = pc_panel_1d if dimension == 1 else pc_panel_2d

        scheme_func(axes[i, 0], d_demes, K, phi, norm)
        axes[i, 0].set_ylabel(f"{dimension}D  " + rf"$\phi={phi:g}$", fontsize=10)

        for j, level in enumerate(M_LEVELS):
            p = pcs[(pcs.dimension == dimension) & (pcs.phi == phi) & (pcs.m_level == level)]
            pc_func(axes[i, j + 1], p, norm)
            if phi == phis[0]:
                ratio = meta.loc[(dimension, level), "M_over_Mc"]
                axes[i, j + 1].set_title(M_LEVEL_TITLES[level] + rf"  ($M/M_c={ratio:.2g}$)")

    axes[0, 0].set_title("sampling scheme")
    axes[len(phis), 0].set_title("sampling scheme")

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
