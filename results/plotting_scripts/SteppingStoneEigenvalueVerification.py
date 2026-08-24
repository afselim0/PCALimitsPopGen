#%%
"""
Verify the closed-form stepping-stone eigenvalue against the measured one, 1D and 2D.

A single K per dimension (no K sweep), so the swept quantities are n_per_deme (color) and
M (marker shape): the median measured eigenvalue (from
src/analysis/SteppingStoneEvenSampling.py) against the closed-form nu_theory, at a handful
of representative M per n_per_deme in the even-sampling sweep.

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

from plotting_style import setup_plotting_style, okabe_ito

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SteppingStoneEvenSampling")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

NU1_LABEL = r'$\nu_1 = N / (M K \sin^2(\pi / 2K))$'
NUM_M_LEVELS = 5
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


def representative_M(df, num_levels=NUM_M_LEVELS):
    """num_levels simulated M values, log-spaced targets snapped to the nearest available M."""
    m_available = np.sort(df.M.unique())
    targets = np.geomspace(m_available.min(), m_available.max(), num_levels)
    idx = np.unique([np.argmin(np.abs(m_available - t)) for t in targets])
    return m_available[idx]


def panel(ax, df, title):
    """nu_meas (median over replicates) vs nu_theory: color = n_per_deme, marker = M, y=x dotted."""
    n_values = np.sort(df.n_per_deme.unique())
    m_values = representative_M(df)
    color_of = {n: okabe_ito[i % len(okabe_ito)] for i, n in enumerate(n_values)}
    marker_of = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(m_values)}

    d = df[df.M.isin(m_values)]
    lim = [min(d.nu_theory.min(), d.nu_meas.min()) * 0.8,
           max(d.nu_theory.max(), d.nu_meas.max()) * 1.2]
    ax.plot(lim, lim, ':', color='0.35', linewidth=1.4, zorder=1, label=r'$y=x$')

    for n_per in n_values:
        for m in m_values:
            cell = d[(d.n_per_deme == n_per) & (d.M == m)]
            ax.scatter(cell.nu_theory, cell.nu_meas, s=28, color=color_of[n_per],
                      marker=marker_of[m], edgecolors='none', alpha=0.85, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(NU1_LABEL)
    ax.set_ylabel(r'measured $\nu_1$ (median)')
    ax.set_title(title)

    n_handles = [plt.Line2D([], [], marker='o', linestyle='none', color=color_of[n],
                            markersize=6, label=f'{n:g}') for n in n_values]
    m_handles = [plt.Line2D([], [], marker=marker_of[m], linestyle='none', color='0.3',
                            markersize=6, label=f'{m:.3g}') for m in m_values]
    leg1 = ax.legend(handles=[plt.Line2D([], [], linestyle=':', color='0.35', label=r'$y=x$'),
                              *n_handles],
                     title=r'$n_{\mathrm{per\,deme}}$', loc='upper left', fontsize=7.5,
                     title_fontsize=8, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=m_handles, title=r'$M$', loc='lower right', fontsize=7.5,
             title_fontsize=8, frameon=False)


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
