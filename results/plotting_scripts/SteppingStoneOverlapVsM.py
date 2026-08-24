#%%
"""
Spatial-axis recovery vs migration, sampling every deme.

A  1D stepping stone, every deme sampled, curves are n-per-deme.
B  2D stepping stone, every deme sampled, curves are n-per-deme (degenerate leading pair).

x is 1/M rather than M: overlap grows as migration weakens, so the reciprocal makes the
transition curve increasing. That reciprocal is applied in the analysis, not here.

The fit, the band and the theoretical crossing are computed upstream by
src/sweeps_analysis/summarise.py and read here as columns -- this script draws, it does not
fit, and it holds no theory of its own. A thin solid vertical line marks the 1/M where the
fit reaches squared overlap 0.9, a dashed line where theory does.

2D compares the degenerate (1,0)/(0,1) pair as a subspace (procrustes-aligned principal
angles), since no canonical basis exists inside a degenerate eigenspace.

Reads results/sweep_summaries/stepping_stone_even/{curves,crossings,points}.csv
Writes results/figures/SteppingStoneOverlapVsM.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from crossing_panel import crossing_panel_from_summary, crossing_table, encoding_legend
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "stepping_stone_even")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP = cm.viridis
TARGET = 0.9
SWEEPS = [("ss1d_vary_n", "1D"), ("ss2d_vary_n", "2D")]


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    setup_plotting_style()
    curves = pd.read_csv(os.path.join(summary_dir, "curves.csv"))
    crossings = pd.read_csv(os.path.join(summary_dir, "crossings.csv"))
    points = pd.read_csv(os.path.join(summary_dir, "points.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    results = []
    for ax, (sweep, label) in zip(axes, SWEEPS):
        res = crossing_panel_from_summary(ax, curves, crossings, points, sweep=sweep,
                                          cmap=CMAP, cbar_label=r"$n_{\mathrm{per\,deme}}$", target=target)
        ax.set_xlabel(r"$1/M$")
        ax.set_title(label)
        results.append(res)

    encoding_legend(axes[0], target=target, loc="lower right")
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SteppingStoneOverlapVsM.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    for (_, label), res in zip(SWEEPS, results):
        print(f"\n{label}")
        print(crossing_table(res, "n", "1/M"))
    return fig


if __name__ == "__main__":
    make_figure()

# %%
