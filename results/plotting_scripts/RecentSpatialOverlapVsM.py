#%%
"""
Spatial-axis recovery vs migration for the recent-split demography: a panmictic population
A plus a K x K grid B that split off t_split generations ago.

A  n-per-deme-B varied at a fixed 50/50 split between the structured and panmictic halves.
B  phi (B's share of the total sample) varied at n-per-deme-B = 4.

phi here is the *total sample fraction* drawn from the structured demes -- not the
fraction-of-demes-sampled phi in SteppingStoneRandomOverlapVsM, which is a different sweep
entirely. Every deme is always sampled here; phi only trades n_A against B's total.

The target axis is the (1,0)/(0,1) spatial pair, located by overlap rather than by rank:
it is not reliably the leading eigenvalue of the full A+B system once A carries a large
share of the sample.

x is 1/M, applied in the analysis rather than here. The fit and the band come from
src/sweeps_analysis/summarise.py -- this script draws only.

Unlike the other overlap figures, this one reports a PEAK rather than a detection crossing.
cos^2 is not monotone in 1/M here: it rises to ~0.93 near 1/M ~ 0.3 and then decays, because
at very weak migration the demes drift into independent clusters and a single (1,0)/(0,1)
pair stops describing the structure. The summariser therefore fits these sweeps with a
*concave* shape constraint (scam's "cv") rather than the monotone one used elsewhere, which
captures the turnover without prescribing where it falls, and marks each curve at its peak.
The 0.9 line is kept as a reference level, not as something claimed to be crossed.

Reads results/sweep_summaries/split_stepping_stone/{curves,crossings,points}.csv
    (sweeps split_ss_vary_N and split_ss_vary_phi)
Writes results/figures/RecentSpatialOverlapVsM.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from crossing_panel import peak_legend, peak_panel_from_summary, peak_table
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries",
                           "split_stepping_stone")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP = cm.viridis
CMAP_PHI = cm.plasma
TARGET = 0.9


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    setup_plotting_style()
    curves = pd.read_csv(os.path.join(summary_dir, "curves.csv"))
    crossings = pd.read_csv(os.path.join(summary_dir, "crossings.csv"))
    points = pd.read_csv(os.path.join(summary_dir, "points.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    # Both panels' curves fill the upper band, so the per-curve legends go low; the
    # encoding key is then placed in the opposite corner automatically.
    res_n = peak_panel_from_summary(
        axes[0], curves, crossings, points, sweep="split_ss_vary_N",
        cmap=CMAP, cbar_label=r"$n_{B,\mathrm{per\,deme}}$", target=target,
        legend_loc="lower left")
    res_phi = peak_panel_from_summary(
        axes[1], curves, crossings, points, sweep="split_ss_vary_phi",
        cmap=CMAP_PHI, cbar_label=r"$\phi$", target=target,
        legend_loc="lower left")

    for ax in axes:
        ax.set_xlabel(r"$1/M$")
    axes[0].set_title(r"fixed $\phi=0.5$")
    axes[1].set_title(r"fixed $n_{B,\mathrm{per\,deme}}=4$")
    peak_legend(axes[0], target=target)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "RecentSpatialOverlapVsM.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    print(peak_table(res_n, "n_B", "1/M"))
    print(peak_table(res_phi, "phi", "1/M"))
    return fig


if __name__ == "__main__":
    make_figure()

# %%
