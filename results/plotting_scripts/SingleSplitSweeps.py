#%%
"""
PC1 recovery across the two-population split sweep, as a detection threshold in Fst.

A  PC1 squared overlap vs Fst at ten sample sizes N (phi = 0.5).
B  PC1 squared overlap vs Fst at ten sampling fractions phi (N = 200).

Each level is a monotone shape-constrained (scam) fit through every replicate, with its
95% confidence band. The comparison against theory is made at a single point rather than
along the whole curve: a filled marker sits on the curve where it reaches squared overlap
0.9, and a dashed vertical line marks the Fst where the theory reaches it.

The fit, the band and the theoretical crossing are all computed upstream by
src/sweeps_analysis/summarise.py and read here as columns -- this script draws, it does not
fit, and it holds no theory of its own. The theoretical crossing inverts the exact Ding &
Yang overlap (src/threshold_and_overlap) for nu = 2 N phi (1-phi) Fst using each level's own
measured L_eff and locus moments, so nothing in it is fitted: the horizontal gap between
the marker and the dashed line at each level is the whole result.

Reads results/sweep_summaries/single_split/{curves,crossings,points}.csv
    (sweeps split_vary_N and split_vary_phi)
Writes results/figures/SingleSplitSweeps.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from crossing_panel import crossing_panel_from_summary, crossing_table, encoding_legend
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "single_split")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis
CMAP_PHI = cm.plasma
TARGET = 0.9


def load_summary(summary_dir):
    """The three summary frames a panel needs."""
    return (pd.read_csv(os.path.join(summary_dir, "curves.csv")),
            pd.read_csv(os.path.join(summary_dir, "crossings.csv")),
            pd.read_csv(os.path.join(summary_dir, "points.csv")))


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    setup_plotting_style()
    curves, crossings, points = load_summary(summary_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    res_N = crossing_panel_from_summary(
        axes[0], curves, crossings, points, sweep="split_vary_N",
        cmap=CMAP_N, cbar_label=r"$N$", target=target)
    res_phi = crossing_panel_from_summary(
        axes[1], curves, crossings, points, sweep="split_vary_phi",
        cmap=CMAP_PHI, cbar_label=r"$\phi$", target=target)

    for ax in axes:
        ax.set_xlabel(r"$F_{ST}$")
    axes[0].set_title(r"varying $N$  ($\phi=0.5$)")
    axes[1].set_title(r"varying $\phi$  ($N=200$)")
    encoding_legend(axes[0], target=target)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SingleSplitSweeps.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    print(crossing_table(res_N, "N", "Fst"))
    print(crossing_table(res_phi, "phi", "Fst"))
    return fig


if __name__ == "__main__":
    make_figure()

# %%
