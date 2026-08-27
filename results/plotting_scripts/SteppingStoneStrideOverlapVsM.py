#%%
"""
Spatial-axis recovery vs migration, sampling every p-th deme at a fixed total N.

A  1D stepping stone, N = 100 throughout, curves are the sampling dilution p.
B  2D stepping stone, N = 324 throughout, curves are the sampling dilution p (every p-th
   row and column).

The counterpart of SteppingStoneRandomOverlapVsM. There the curves are a random fraction
of demes at fixed n-per-deme, so both coverage and N move together; here the total sample
size is held fixed and only the coverage thins, so the curves isolate what spatial
coverage alone buys.

x is 1/M rather than M: overlap grows as migration weakens, so the reciprocal makes the
transition curve increasing. That reciprocal is applied in the analysis, not here.

The fit (a monotone shape-constrained scam smooth), its confidence band and the
theoretical crossing are computed upstream by src/sweeps_analysis/summarise.py and read
here as columns -- this script draws, it does not fit, and it holds no theory of its own.
A filled marker sits on the curve where it reaches squared overlap 0.9; a dashed vertical
line marks where theory reaches it.

2D compares the degenerate (1,0)/(0,1) pair as a subspace (procrustes-aligned principal
angles). At K=9 that pair is exactly degenerate only when the retained rows sit
symmetrically about the lattice centre, which p=3 ({0,3,6}) does not -- see the config.

Reads results/sweep_summaries/stepping_stone_stride/{curves,crossings,points}.csv
Writes results/figures/SteppingStoneStrideOverlapVsM.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from crossing_panel import crossing_panel_from_summary, crossing_table, encoding_legend
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "stepping_stone_stride")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP = cm.plasma
TARGET = 0.9
SWEEPS = [("ss1d_vary_deme_stride", "1D"), ("ss2d_vary_deme_stride", "2D")]
LEGEND_LABEL = r"sampling dilution ($p$)"


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    setup_plotting_style()
    curves = pd.read_csv(os.path.join(summary_dir, "curves.csv"))
    crossings = pd.read_csv(os.path.join(summary_dir, "crossings.csv"))
    points = pd.read_csv(os.path.join(summary_dir, "points.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    results = []
    for ax, (sweep, label) in zip(axes, SWEEPS):
        res = crossing_panel_from_summary(ax, curves, crossings, points, sweep=sweep,
                                          cmap=CMAP, cbar_label=LEGEND_LABEL,
                                          target=target)
        ax.set_xlabel(r"$1/M$")
        ax.set_title(label)
        results.append(res)

    encoding_legend(axes[0], target=target, loc="lower right")
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    # tight_layout does not reserve room for a legend anchored outside the axes, so the
    # inter-panel gap is set explicitly to hold panel A's legend clear of panel B's
    # y-axis label.
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.62)
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneStrideOverlapVsM.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    for (_, label), res in zip(SWEEPS, results):
        print(f"\n{label}")
        print(crossing_table(res, "p", "1/M"))
    return fig


if __name__ == "__main__":
    make_figure()

# %%
