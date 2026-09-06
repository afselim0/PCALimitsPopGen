#%%
"""
Spatial-axis recovery vs migration, sampling every p-th deme at a fixed total N.

A  1D stepping stone, N = 100 throughout, curves are the sampling dilution p.
B  2D stepping stone, N = 324 throughout, curves are the sampling dilution p (every p-th
   row and column).

The counterpart of PhaseTransitionRandomSS. There the curves are a random fraction
of demes at fixed n-per-deme, so both coverage and N move together; here the total sample
size is held fixed and only the coverage thins, so the curves isolate what spatial
coverage alone buys.

x is 1/M rather than M: overlap grows as migration weakens, so the reciprocal makes the
transition curve increasing. That reciprocal is applied in the analysis, not here.

The fit (flat below the fitted transition, monotone increasing above), its confidence band and the
theoretical crossing are computed upstream by src/sweeps_analysis/summarise.py and read
here as columns -- this script draws, it does not fit, and it holds no theory of its own.
A filled marker sits on the curve where it reaches the target and a square on the x axis
at the fitted detection threshold; dashed / dotted vertical lines mark where theory reaches
each.

2D compares the degenerate (1,0)/(0,1) pair as a subspace (procrustes-aligned principal
angles). At K=9 that pair is exactly degenerate only when the retained rows sit
symmetrically about the lattice centre, which p=3 ({0,3,6}) does not -- see the config.

Reads results/sweep_summaries/stepping_stone_stride/{curves,crossings,points}.csv
Writes results/figures/PhaseTransitionStrideSS.pdf
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
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "stepping_stone_stride")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP = cm.plasma
TARGET = DEFAULT_TARGET
SWEEPS = [("ss1d_vary_deme_stride", "1D"), ("ss2d_vary_deme_stride", "2D")]
LEGEND_LABEL = "sampling\ndilution ($p$)"


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET,
                standardized=False):
    setup_plotting_style()
    curves, crossings, points, theory = load_summary(summary_dir)

    # The eigenvalue threshold marker is off here: K ~ N in the stepping stones, so
    # many structure axes are detached at once and the single-spike fit behind that
    # marker does not apply (sub-threshold lambda/lambda_bulk is 1.16-1.28, not 1).
    fig, axes = plt.subplots(1, 2, figsize=figsize(6.5, 4.35))
    results = []
    for ax, (sweep, label) in zip(axes, SWEEPS):
        res = crossing_panel_from_summary(ax, curves, crossings, points, sweep=sweep,
                                          cmap=CMAP, cbar_label=LEGEND_LABEL,
                                          target=target, theory_table=theory, standardized=standardized, show_eig_threshold=False)
        style_swept_axis(ax, r"$1/M$")
        ax.set_title(L_title(sweep, label))
        results.append(res)

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
    handles = encoding_handles(target=target)
    # reserve_side_legend, not a bare wspace: the per-panel keys are anchored
    # OUTSIDE the axes, so tight_layout cannot see them and lays the axes across the
    # full width -- the key then lands on the neighbouring panel. This reserves the
    # legend's width in inches so each panel keeps its own square.
    reserve_side_legend(fig, n_panels=2,
                        bottom=encoding_bottom_margin(handles, fig=fig))
    figure_encoding_legend(fig, handles)
    out = save_figure(fig, figure_dir, "PhaseTransitionStrideSS", standardized)
    print(f"wrote {out}")
    for (_, label), res in zip(SWEEPS, results):
        print(f"\n{label}")
        print(crossing_table(res, "p", "1/M"))
    return fig


def make_all(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    """Both versions: the ordinary GRM and the standardized one, same theory marks."""
    return [make_figure(summary_dir, figure_dir, target, standardized=s)
            for s in (False, True)]


if __name__ == "__main__":
    make_all()

# %%
