#%%
"""
PC recovery of the true spatial axis vs M, random deme subsets, 1D and 2D (2x2).

Row 1: 1D.  Row 2: 2D.
Column A: fixed phi = 0.4, varying n_per_deme.
Column B: fixed n_per_deme = 8, varying phi.

Solid lines are the measured median over replicates; dashed lines are the BBP prediction
(analytical_cos, from the measured L_eff and the closed-form nu at the reduced N -- no
free parameters). Color encodes the swept sampling parameter.

Reads results/analysis/SteppingStoneRandomSampling, writes
results/figures/SteppingStoneRandomOverlapVsM.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis",
                            "SteppingStoneRandomSampling")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis
CMAP_PHI = cm.plasma


def panel(ax, df, key, cmap, label, title):
    levels = np.sort(df[key].unique())
    norm = colors.Normalize(vmin=levels.min(), vmax=levels.max())

    for level in levels:
        d = df[df[key] == level].sort_values("M")
        color = cmap(norm(level))
        ax.plot(d.M, d.median_cos, "-", color=color, linewidth=1.5, zorder=3)
        ax.plot(d.M, d.analytical_cos, "--", color=color, linewidth=1.1, alpha=0.85,
                zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$M$")
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title(title)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(label)
    cbar.outline.set_visible(False)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    df_1d = pd.read_csv(os.path.join(analysis_dir, "random_1D.csv"))
    df_2d = pd.read_csv(os.path.join(analysis_dir, "random_2D.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))

    for i, (df, dim_label) in enumerate([(df_1d, "1D"), (df_2d, "2D")]):
        K = int(df.K.iloc[0])
        d_n = df[df.sweep == "vary_n"]
        d_phi = df[df.sweep == "vary_phi"]
        phi_n = d_n.phi.iloc[0]
        n_phi = int(d_phi.n_per_deme.iloc[0])

        panel(axes[i, 0], d_n, "n_per_deme", CMAP_N, r"$n_{\mathrm{per\,deme}}$",
              rf"{dim_label}  ($K={K}$), varying $n$  ($\phi={phi_n:g}$)")
        panel(axes[i, 1], d_phi, "phi", CMAP_PHI, r"$\phi$",
              rf"{dim_label}  ($K={K}$), varying $\phi$  ($n={n_phi}$)")

    axes[0, 0].plot([], [], "-", color="0.35", linewidth=1.5, label="simulation (median)")
    axes[0, 0].plot([], [], "--", color="0.35", linewidth=1.1, label="BBP prediction")
    axes[0, 0].legend(loc="lower right", fontsize=9, frameon=False)

    for (i, j), letter in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
        axes[i, j].text(-0.16, 1.06, letter, transform=axes[i, j].transAxes,
                        fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneRandomOverlapVsM.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
