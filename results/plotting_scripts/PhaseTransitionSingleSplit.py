#%%
"""
PC1 recovery across the two-population split sweep, as a detection threshold in Fst.

A  PC1 cosine similarity vs Fst at ten sample sizes N (phi = 0.5).
B  PC1 cosine similarity vs Fst at ten sampling fractions phi (N = 200).

Each level is a flat-then-increasing threshold fit through every replicate, with its
95% confidence band. The comparison against theory is made at two points rather than
along the whole curve: a filled marker on the curve where it reaches the target, a square on
the x axis at the fitted detection threshold, and a dashed / dotted vertical line where the
theory reaches each. Ticks under the x axis restate both theoretical positions using L_eff.

The fit, the band and the theoretical crossing are all computed upstream by
src/sweeps_analysis/summarise.py and read here as columns -- this script draws, it does not
fit, and it holds no theory of its own. The theoretical crossing inverts the exact Ding &
Yang overlap (src/threshold_and_overlap) for nu = 2 N phi (1-phi) Fst using each level's own
measured L_eff and locus moments, so nothing in it is fitted: the horizontal gap between
the marker and the dashed line at each level is the whole result.

Reads results/sweep_summaries/single_split/{curves,crossings,points}.csv
    (sweeps split_vary_N and split_vary_phi)
Writes results/figures/PhaseTransitionSingleSplit.pdf
"""
import os

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import pandas as pd
from matplotlib import cm

from crossing_panel import (reserve_side_legend, PANEL_LETTER_SIZE, figsize, DEFAULT_TARGET, L_title, crossing_panel_from_summary,
                            crossing_table, encoding_handles, figure_encoding_legend,
                            encoding_bottom_margin, load_summary, save_figure,
                            style_swept_axis, variant_note)
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "single_split")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis
CMAP_PHI = cm.plasma
TARGET = DEFAULT_TARGET


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET,
                standardized=False):
    setup_plotting_style()
    curves, crossings, points, theory = load_summary(summary_dir)

    fig, axes = plt.subplots(1, 2, figsize=figsize(6.5, 4.35))

    res_N = crossing_panel_from_summary(
        axes[0], curves, crossings, points, sweep="split_vary_N",
        cmap=CMAP_N, cbar_label=r"$N$", target=target, theory_table=theory,
        standardized=standardized)
    res_phi = crossing_panel_from_summary(
        axes[1], curves, crossings, points, sweep="split_vary_phi",
        cmap=CMAP_PHI, cbar_label=r"$\phi$", target=target, theory_table=theory,
        standardized=standardized)

    for ax in axes:
        style_swept_axis(ax, r"$F_{ST}$")
    axes[0].set_title(L_title("split_vary_N", r"varying $N$  ($\phi=0.5$)"))
    axes[1].set_title(L_title("split_vary_phi", r"varying $\phi$  ($N=200$)"))
    for ax in axes:
        variant_note(ax, standardized)

    for ax, letter in zip(axes, "AB"):
        # Offset in POINTS, not axes fraction: at 12 pt the fractional offset put the
        # letter on the "using L_eff" row and the panel title.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -34 / 72, 26 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="top")

    # The encoding key goes under both panels, so tight_layout runs first (it knows nothing
    # about a figure-level legend) and the bottom margin is then opened to hold it.
    handles = encoding_handles(target=target, show_band=False)
    # reserve_side_legend, not a bare wspace: the per-panel keys are anchored OUTSIDE
    # the axes, so tight_layout cannot see them and lays the axes across the full
    # width -- the key then lands on the neighbouring panel. This reserves the
    # legend's width in inches so each panel keeps its own square.
    reserve_side_legend(fig, n_panels=2, bottom=encoding_bottom_margin(handles, fig=fig))
    figure_encoding_legend(fig, handles)
    out = save_figure(fig, figure_dir, "PhaseTransitionSingleSplit", standardized)
    print(f"wrote {out}")
    print(crossing_table(res_N, "N", "Fst"))
    print(crossing_table(res_phi, "phi", "Fst"))
    return fig


def make_all(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    """Both versions: the ordinary GRM and the standardized one, same theory marks."""
    return [make_figure(summary_dir, figure_dir, target, standardized=s)
            for s in (False, True)]


if __name__ == "__main__":
    make_all()

# %%
