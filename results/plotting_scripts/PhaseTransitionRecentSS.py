#%%
"""
Spatial-axis recovery vs migration for the recent-split demography: a panmictic population
A plus a K x K grid B that split off t_split generations ago.

A  n-per-deme-B varied at a fixed 50/50 split between the structured and panmictic halves.
B  phi (B's share of the total sample) varied at n-per-deme-B = 4.

phi here is the *total sample fraction* drawn from the structured demes -- not the
fraction-of-demes-sampled phi in PhaseTransitionRandomSS, which is a different sweep
entirely. Every deme is always sampled here; phi only trades n_A against B's total.

The target axis is the (1,0)/(0,1) spatial pair, located by overlap rather than by rank:
it is not reliably the leading eigenvalue of the full A+B system once A carries a large
share of the sample.

x is 1/M, applied in the analysis rather than here. The fit and the band come from
src/sweeps_analysis/summarise.py -- this script draws only.

Unlike the other overlap figures, this one reports a PEAK rather than a detection crossing.
cos^2 is not monotone in 1/M here: it rises to ~0.93 near 1/M ~ 0.3 and then decays, because
at very weak migration the demes drift into independent clusters and a single (1,0)/(0,1)
pair stops describing the structure. The summariser therefore fits these sweeps with an
*unconstrained* penalized smooth (mgcv's "tp") rather than the shape-constrained one used
elsewhere -- only smoothness is imposed, so the maximum is read off wherever the data puts
it -- and marks each curve at that maximum.
The target line is kept as a reference level, not as something claimed to be crossed.

Reads results/sweep_summaries/split_stepping_stone/{curves,crossings,points}.csv
    (sweeps split_ss_vary_N and split_ss_vary_phi)
Writes results/figures/PhaseTransitionRecentSS.pdf
"""
import os

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import pandas as pd
from matplotlib import cm

from crossing_panel import (reserve_side_legend, PANEL_LETTER_SIZE, figsize, DEFAULT_TARGET, L_title, figure_encoding_legend,
                            load_summary, peak_handles, peak_panel_from_summary,
                            peak_table, save_figure, style_swept_axis, variant_note)
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries",
                           "split_stepping_stone")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

# Room under the panels for the shared encoding key.
BOTTOM_MARGIN = 0.20

CMAP = cm.viridis
CMAP_PHI = cm.plasma
TARGET = DEFAULT_TARGET


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET,
                standardized=False):
    setup_plotting_style()
    # This demography publishes no theory, so the theory frame is deliberately
    # unused: these panels show the measurement alone.
    curves, crossings, points, _ = load_summary(summary_dir, with_theory=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize(6.5, 4.35))
    res_n = peak_panel_from_summary(
        axes[0], curves, crossings, points, sweep="split_ss_vary_N",
        cmap=CMAP, cbar_label=r"$n_{B,\mathrm{per\,deme}}$", target=target, standardized=standardized)
    res_phi = peak_panel_from_summary(
        axes[1], curves, crossings, points, sweep="split_ss_vary_phi",
        cmap=CMAP_PHI, cbar_label=r"$\phi$", target=target, standardized=standardized)

    for ax in axes:
        style_swept_axis(ax, r"$1/M$", tick_row=False)
    axes[0].set_title(L_title("split_ss_vary_N", r"fixed $\phi=0.5$"))
    axes[1].set_title(L_title("split_ss_vary_phi",
                              r"fixed $n_{B,\mathrm{per\,deme}}=4$"))

    for ax in axes:
        variant_note(ax, standardized)

    for ax, letter in zip(axes, "AB"):
        # Offset in POINTS, not axes fraction: at 12 pt the fractional offset put the
        # letter on the "using L_eff" row and the panel title.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -34 / 72, 26 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="top")

    # tight_layout knows nothing about the figure-level key, so the bottom margin is
    # opened for it after the fact; wspace holds panel B's y-axis label clear of A.
    # reserve_side_legend, not a bare wspace: the per-panel keys are anchored OUTSIDE
    # the axes, so tight_layout cannot see them and lays the axes across the full
    # width -- the key then lands on the neighbouring panel. This reserves the
    # legend's width in inches so each panel keeps its own square.
    reserve_side_legend(fig, n_panels=2, bottom=BOTTOM_MARGIN)
    figure_encoding_legend(fig, peak_handles(target=target), ncol=2)
    out = save_figure(fig, figure_dir, "PhaseTransitionRecentSS", standardized)
    print(f"wrote {out}")
    print(peak_table(res_n, "n_B", "1/M"))
    print(peak_table(res_phi, "phi", "1/M"))
    return fig


def make_all(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    """Both versions: the ordinary GRM and the standardized one, same theory marks."""
    return [make_figure(summary_dir, figure_dir, target, standardized=s)
            for s in (False, True)]


if __name__ == "__main__":
    make_all()

# %%
