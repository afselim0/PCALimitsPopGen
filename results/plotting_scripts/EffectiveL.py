#%%
"""
The variance-effective SNP count L_eff, on panmictic genotypes.

A  L_eff against the raw count L, one line per sample size N, with the L_eff = L identity
   for reference. L_eff falls well short of L at every N -- the GRM is built from
   unnormalized genotypes, so locus l enters with weight v_l = p_l(1-p_l), and a coalescent
   frequency spectrum is dominated by rare variants whose weight is near zero. L_eff is
   linear in L with a slope below 1, so it counts how many loci effectively contribute.

B  the same quantity against N at the largest L, where the slope of A is read directly.
   L_eff decreases with N and flattens towards a plateau: a larger sample resolves more
   rare variants, so loci that were monomorphic (and discarded) in a small subsample become
   polymorphic-but-rare in a larger one, adding to L while adding almost nothing to L_eff.
   The decline is decelerating rather than finished -- successive steps fall 11.3% at
   N = 200 down to 1.6% at N = 1000 -- so the curve is approaching its asymptote from above
   rather than having reached it.

Reads results/analysis/EffectiveL, writes results/figures/EffectiveL.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from crossing_panel import save_figure
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "EffectiveL")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis


def panel_A(ax, fig, summary):
    """L_eff vs L, one line per N, against the L_eff = L identity."""
    N_values = np.sort(summary.N.unique())
    norm = colors.Normalize(vmin=N_values.min(), vmax=N_values.max())

    # Identity: the ceiling L_eff would reach if every locus carried equal weight.
    L_grid = np.array([summary.L_requested.min(), summary.L_requested.max()])
    ax.plot(L_grid, L_grid, color='0.55', linestyle='--', linewidth=1.2, zorder=1,
            label=r'$L_{\mathrm{eff}} = L$')

    for N in N_values:
        d = summary[summary.N == N].sort_values("L_requested")
        ax.plot(d.L_requested, d.L_eff, '-', color=CMAP_N(norm(N)), linewidth=1.6,
                zorder=2)

    ax.set_xlabel(r'$L$  (independent SNPs)')
    ax.set_ylabel(r'$L_{\mathrm{eff}} = \dfrac{(\sum_l \ h_l)^2}{\sum_l \ h_l^2}$')
    ax.set_title(r'$L_{\mathrm{eff}}$ vs $L$')
    ax.legend(loc='upper left', fontsize=9, frameon=False)

    sm = cm.ScalarMappable(norm=norm, cmap=CMAP_N)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$N$')
    cbar.outline.set_visible(False)


def panel_B(ax, summary):
    """L_eff vs N at the largest L: the slope of panel A, read directly."""
    L_max = summary.L_requested.max()
    d = summary[summary.L_requested == L_max].sort_values("N")

    ax.plot(d.N, d.L_eff, 'o-', color=CMAP_N(0.35), markersize=4.5, linewidth=1.6)
    ax.set_xlabel(r'$N$ (total number of samples)')
    ax.set_title(rf'$L_{{\mathrm{{eff}}}}$ vs $N$  ($L={L_max:g}$)')
    # Zero-anchored, so the plateau is read against the full scale rather than against a
    # zoomed axis that would exaggerate the remaining decline.
    ax.set_ylim(0, d.L_eff.max() * 1.08)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    summary = pd.read_csv(os.path.join(analysis_dir, "summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    panel_A(axes[0], fig, summary)
    panel_B(axes[1], summary)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

    fig.tight_layout()
    out = save_figure(fig, figure_dir, "EffectiveL")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
