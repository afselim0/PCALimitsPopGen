#%%
"""
Verify the closed-form stepping-stone eigenvalue against the measured one, 1D and 2D.

A single K per dimension (no K sweep), so the swept quantity is n_per_deme: the median
measured eigenvalue (from src/analysis/SteppingStoneEvenSampling.py) against the
closed-form nu_theory, at every simulated m and every n_per_deme in the even-sampling
sweep.

A  1D
B  2D (nu_theory is the degenerate pair's shared value; nu_meas is the mean eigenvalue
   of the two sample eigenvectors matched to it by overlap)

Reads results/analysis/SteppingStoneEvenSampling, writes
results/figures/SteppingStoneEigenvalueVerification.pdf
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
    """nu_meas (median over replicates) vs nu_theory, colored by n_per_deme, y=x dotted."""
    n_values = np.sort(df.n_per_deme.unique())
    norm = colors.Normalize(vmin=n_values.min(), vmax=n_values.max())

    lim = [min(df.nu_theory.min(), df.nu_meas.min()) * 0.8,
           max(df.nu_theory.max(), df.nu_meas.max()) * 1.2]
    ax.plot(lim, lim, ':', color='0.35', linewidth=1.4, zorder=1, label=r'$y=x$')

    for n_per in n_values:
        d = df[df.n_per_deme == n_per]
        ax.scatter(d.nu_theory, d.nu_meas, s=20, color=CMAP_N(norm(n_per)),
                  edgecolors='none', alpha=0.85, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r'closed-form $\nu_1$')
    ax.set_ylabel(r'measured $\nu_1$ (median)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8.5, frameon=False)

    sm = cm.ScalarMappable(norm=norm, cmap=CMAP_N)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$n_{\mathrm{per\,deme}}$')
    cbar.outline.set_visible(False)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    df_1d = pd.read_csv(os.path.join(analysis_dir, "even_1D.csv"))
    df_2d = pd.read_csv(os.path.join(analysis_dir, "even_2D.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    panel(axes[0], df_1d, rf'1D  ($K={int(df_1d.K.iloc[0])}$)')
    panel(axes[1], df_2d, rf'2D  ($K={int(df_2d.K.iloc[0])}$)')

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneEigenvalueVerification.pdf")
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
