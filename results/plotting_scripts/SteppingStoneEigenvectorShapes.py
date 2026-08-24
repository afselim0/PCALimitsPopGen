#%%
"""
Sample PC shapes at three regimes -- no structure, weak, strong -- 1D and 2D (2x3).

Row 1 (1D): PC1 against deme position, colored by deme index along the chain.
Row 2 (2D): PC1 vs PC2 scatter, colored by deme index (procrustes-aligned to the
degenerate (1,0)/(0,1) closed-form pair beforehand, in the analysis script, so PC1/PC2
here are oriented consistently with the theoretical axes rather than by an arbitrary
in-plane rotation).

Columns: no structure (M >> M_c), weak (M ~ M_c), strong (M << M_c).

Reads results/analysis/SteppingStoneEigenvectorShapes, writes
results/figures/SteppingStoneEigenvectorShapes.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis",
                            "SteppingStoneEigenvectorShapes")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

REGIMES = ["no_structure", "weak", "strong"]
REGIME_TITLES = {"no_structure": "no structure", "weak": "weak structure",
                 "strong": "strong structure"}
CMAP = cm.viridis


def panel_1d(ax, d, meta_row, norm):
    ax.scatter(d.deme, d.pc1, c=d.deme, cmap=CMAP, norm=norm, s=8, edgecolors='none')
    ax.set_xlabel('deme index')
    ax.set_ylabel('PC1')
    ax.set_title(rf'$M/M_c={meta_row.M_over_Mc:.2f}$')


def panel_2d(ax, d, meta_row, norm):
    ax.scatter(d.pc1, d.pc2, c=d.deme, cmap=CMAP, norm=norm, s=8, edgecolors='none')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(rf'$M/M_c={meta_row.M_over_Mc:.2f}$')
    ax.set_aspect('equal', adjustable='datalim')


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index(["dimension", "regime"])

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.4))

    K_1d = int(meta.loc[(1, REGIMES[0])].K)
    K_2d = int(meta.loc[(2, REGIMES[0])].K)
    n_1d = int(meta.loc[(1, REGIMES[0])].n_per_deme)
    n_2d = int(meta.loc[(2, REGIMES[0])].n_per_deme)
    norm_1d = colors.Normalize(vmin=0, vmax=K_1d - 1)
    norm_2d = colors.Normalize(vmin=0, vmax=K_2d * K_2d - 1)

    for j, regime in enumerate(REGIMES):
        d1 = pcs[(pcs.dimension == 1) & (pcs.regime == regime)]
        panel_1d(axes[0, j], d1, meta.loc[(1, regime)], norm_1d)
        axes[0, j].set_title(f'{REGIME_TITLES[regime]}\n' + axes[0, j].get_title())

        d2 = pcs[(pcs.dimension == 2) & (pcs.regime == regime)]
        panel_2d(axes[1, j], d2, meta.loc[(2, regime)], norm_2d)

    axes[0, 0].set_ylabel(rf'1D ($K={K_1d}$, $n={n_1d}$)' + '\nPC1')
    axes[1, 0].set_ylabel(rf'2D ($K={K_2d}$, $n={n_2d}$)' + '\nPC2')

    for i, letter in enumerate("AB"):
        axes[i, 0].text(-0.22, 1.05, letter, transform=axes[i, 0].transAxes,
                        fontsize=13, fontweight='bold', va='bottom')

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneEigenvectorShapes.pdf")
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
