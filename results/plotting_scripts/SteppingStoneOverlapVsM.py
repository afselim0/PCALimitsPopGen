#%%
"""
PC recovery vs M, even sampling, 1D and 2D -- one curve per n_per_deme.

A  1D: |cos| between PC1 and the leading closed-form eigenvector.
B  2D: procrustes-aligned subspace cosine between the top two PCs and the degenerate
   (1,0)/(0,1) pair.

Solid lines are the measured median over replicates; dashed lines are the BBP prediction
(analytical_cos, built from the measured L_eff and the closed-form nu -- no free
parameters). Colour encodes n_per_deme.

Reads results/analysis/SteppingStoneEvenSampling, writes
results/figures/SteppingStoneOverlapVsM.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SteppingStoneEvenSampling")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis


def panel(ax, df, title):
    n_values = np.sort(df.n_per_deme.unique())
    norm = colors.Normalize(vmin=n_values.min(), vmax=n_values.max())

    for n_per in n_values:
        d = df[df.n_per_deme == n_per].sort_values("M")
        color = CMAP_N(norm(n_per))
        ax.plot(d.M, d.median_cos, "-", color=color, linewidth=1.5, zorder=3)
        ax.plot(d.M, d.analytical_cos, "--", color=color, linewidth=1.1, alpha=0.85,
                zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$M$")
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title(title)

    sm = cm.ScalarMappable(norm=norm, cmap=CMAP_N)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$n_{\mathrm{per\,deme}}$")
    cbar.outline.set_visible(False)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    df_1d = pd.read_csv(os.path.join(analysis_dir, "even_1D.csv"))
    df_2d = pd.read_csv(os.path.join(analysis_dir, "even_2D.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    panel(axes[0], df_1d, rf"1D  ($K={int(df_1d.K.iloc[0])}$)")
    panel(axes[1], df_2d, rf"2D  ($K={int(df_2d.K.iloc[0])}$)")

    axes[0].plot([], [], "-", color="0.35", linewidth=1.5, label="simulation (median)")
    axes[0].plot([], [], "--", color="0.35", linewidth=1.1, label="BBP prediction")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneOverlapVsM.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
