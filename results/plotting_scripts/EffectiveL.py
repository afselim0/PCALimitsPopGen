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
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from crossing_panel import PANEL_LETTER_SIZE, figsize, save_figure
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
                zorder=2, label=f'{N:g}')

    ax.set_xlabel(r'$L$  (independent SNPs)')
    # dfrac, not the inline /: at LABEL_PT the inline form set the two sums on one line
    # and the ratio read as a product.
    ax.set_ylabel(r'$L_{\text{eff}}$',
                  labelpad=2)
    ax.set_title(r'$L_{\mathrm{eff}}$ vs raw $L$')
    # The identity line is keyed inside the panel; the ten N curves get their own discrete
    # legend under the row (see make_figure). A continuous colourbar was the wrong encoding
    # here -- N takes ten chosen values, not a continuum, and the bar asked the reader to
    # interpolate a ramp to recover them.
    ax.legend([Line2D([], [], color='0.55', linestyle='--', linewidth=1.2)],
              [r'$L_{\mathrm{eff}} = L$'], loc='upper left', frameon=False)
    handles = [Line2D([], [], color=CMAP_N(norm(N)), linewidth=1.6) for N in N_values]
    return handles, [f'{N:g}' for N in N_values]


def panel_B(ax, summary):
    """L_eff vs N at the largest L: the slope of panel A, read directly."""
    L_max = summary.L_requested.max()
    d = summary[summary.L_requested == L_max].sort_values("N")

    ax.plot(d.N, d.L_eff, 'o-', color=CMAP_N(0.35), markersize=3.5, linewidth=1.6)
    ax.set_xlabel(r'$N$ (total number of samples)')
    # Panel B plots L_eff, like panel A. Without this the axis was left unlabelled and
    # picked up the colorbar's "N", which named the wrong quantity.
    ax.set_ylabel(r'$L_{\mathrm{eff}}$')
    ax.set_title(rf'$L_{{\mathrm{{eff}}}}$ vs $N$  ($L={L_max:g}$)')
    # Zero-anchored, so the plateau is read against the full scale rather than against a
    # zoomed axis that would exaggerate the remaining decline.
    ax.set_ylim(0, d.L_eff.max() * 1.08)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    summary = pd.read_csv(os.path.join(analysis_dir, "summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=figsize(6.5, 3.4))
    handles, labels = panel_A(axes[0], fig, summary)
    panel_B(axes[1], summary)

    # Square axes boxes, set directly rather than by guessing a figure height: the
    # colourbar and the shared y axis take horizontal space that the figure aspect alone
    # cannot account for.
    for ax in axes:
        ax.set_box_aspect(1.0)

    for ax, letter in zip(axes, "AB"):
        # Offset in POINTS, not axes fraction: a fraction of a 2.6 in panel is much
        # less page than a fraction of the old wide one, so the fractional offset put
        # panel B's letter inside panel A.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -22 / 72, 12 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight='bold', va='top')

    # Leave a band at the foot of the canvas for the N key, so tight_layout does not
    # spread the axes over it and drop the legend on the x labels.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    # The ten N curves are keyed BELOW the panel row, in five columns. Ten entries beside
    # a 2.6 in panel would be taller than the panel itself; under the row they sit in the
    # clear band tight_layout leaves beneath the x labels.
    fig.legend(handles, labels, title=r'$N$ (total number of samples)',
               loc='upper center', bbox_to_anchor=(0.5, 0.115), ncol=5,
               frameon=False, handlelength=1.4, columnspacing=1.4)
    out = save_figure(fig, figure_dir, "EffectiveL")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
