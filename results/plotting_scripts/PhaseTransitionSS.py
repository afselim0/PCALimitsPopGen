#%%
"""
Spatial-axis recovery vs migration, sampling every deme.

A  1D stepping stone, every deme sampled, curves are n-per-deme.
B  2D stepping stone, every deme sampled, curves are n-per-deme (degenerate leading pair).

x is 1/M rather than M: overlap grows as migration weakens, so the reciprocal makes the
transition curve increasing. That reciprocal is applied in the analysis, not here.

The fit (flat below the fitted transition, monotone increasing above), its 95% band and the theoretical
crossing are computed upstream by src/sweeps_analysis/summarise.py and read here as columns
-- this script draws, it does not fit, and it holds no theory of its own. A filled marker
sits on the curve where it reaches the target squared overlap, a dashed vertical line where theory
reaches it.

2D compares the degenerate (1,0)/(0,1) pair as a subspace (procrustes-aligned principal
angles), since no canonical basis exists inside a degenerate eigenspace.

Reads results/sweep_summaries/stepping_stone_even/{curves,crossings,points}.csv
Writes results/figures/SteppingStoneOverlapVsM.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from crossing_panel import (DEFAULT_TARGET, crossing_panel_from_summary,
                            crossing_table, encoding_legend, load_summary, save_figure,
                            variant_note)
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "stepping_stone_even")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP = cm.viridis
TARGET = DEFAULT_TARGET
SWEEPS = [("ss1d_vary_n", "1D"), ("ss2d_vary_n", "2D")]


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET,
                standardized=False):
    setup_plotting_style()
    curves, crossings, points, theory = load_summary(summary_dir)

    # The eigenvalue threshold marker is off here: K ~ N in the stepping stones, so
    # many structure axes are detached at once and the single-spike fit behind that
    # marker does not apply (sub-threshold lambda/lambda_bulk is 1.16-1.28, not 1).
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    results = []
    for ax, (sweep, label) in zip(axes, SWEEPS):
        res = crossing_panel_from_summary(ax, curves, crossings, points, sweep=sweep,
                                          cmap=CMAP, cbar_label=r"$n_{\mathrm{per\,deme}}$", target=target, theory_table=theory, standardized=standardized, show_eig_threshold=False)
        ax.set_xlabel(r"$1/M$")
        ax.set_title(label)
        results.append(res)

    encoding_legend(axes[0], target=target, loc="lower right", show_eig_threshold=False)
    for ax in axes:
        variant_note(ax, standardized)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    # tight_layout does not reserve room for a legend anchored outside the axes,
    # so the inter-panel gap is set explicitly to hold panel A's legend clear of
    # panel B's y-axis label.
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    out = save_figure(fig, figure_dir, "SteppingStoneOverlapVsM", standardized)
    print(f"wrote {out}")
    for (_, label), res in zip(SWEEPS, results):
        print(f"\n{label}")
        print(crossing_table(res, "n", "1/M"))
    return fig


def make_all(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    """Both versions: the ordinary GRM and the standardized one, same theory marks."""
    return [make_figure(summary_dir, figure_dir, target, standardized=s)
            for s in (False, True)]


if __name__ == "__main__":
    make_all()

# %%
