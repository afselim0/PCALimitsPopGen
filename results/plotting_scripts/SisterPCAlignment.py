#%%
"""
PC alignment with the recently-split sister axis (YRI1/YRI2), as a detection threshold in Fst.

A  squared overlap vs Fst at nine sample sizes N, each sister held at 20% of N (H = 0.20).
B  squared overlap vs Fst at ten sister balances, N = 1000 with the pair supplying 200
   samples between them -- so every level spends the same sequencing effort on the sisters
   and only their *balance* differs. H is the harmonic mean of the two sampling fractions,
   which is dominated by the smaller one: (5,195) gives H = 0.0097 and (100,100) gives
   H = 0.100, a factor of ten apart on identical effort. That is the point of the panel.

Panel A holds H at 0.20 rather than 0.05: at 0.05 the exact nu topped out at 1.67 across
the whole simulated Fst range, so every curve plateaued near squared overlap 0.78 and never
reached the 0.9 target -- the sweep had no crossing to report.

The fit, the band and the theoretical crossing are computed upstream by
src/sweeps_analysis/summarise.py and read here as columns; this script draws only. The
theory uses the exact heteroskedastic route (eps is genuinely nonzero on this demography,
tau/tau_bar spans 0.81-1.17), anchored on the exact nu recorded per cell.

Reads results/sweep_summaries/african_split/{curves,crossings,points}.csv
    (sweeps african_split_vary_N and african_split_vary_harmonic_phi)
Writes results/figures/SisterPCAlignment.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from crossing_panel import crossing_panel_from_summary, crossing_table, encoding_legend
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "african_split")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis
CMAP_H = cm.plasma
TARGET = 0.9


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    setup_plotting_style()
    curves = pd.read_csv(os.path.join(summary_dir, "curves.csv"))
    crossings = pd.read_csv(os.path.join(summary_dir, "crossings.csv"))
    points = pd.read_csv(os.path.join(summary_dir, "points.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    res_N = crossing_panel_from_summary(
        axes[0], curves, crossings, points, sweep="african_split_vary_N",
        cmap=CMAP_N, cbar_label=r"$N$", target=target)
    res_H = crossing_panel_from_summary(
        axes[1], curves, crossings, points, sweep="african_split_vary_harmonic_phi",
        cmap=CMAP_H, cbar_label=r"$H$", target=target)

    for ax in axes:
        ax.set_xlabel(r"$F_{ST}$")
    axes[0].set_title(r"varying $N$  ($H=0.20$)")
    axes[1].set_title(r"varying $\phi_1{:}\phi_2$  ($N=1000$, $n_1{+}n_2=200$)")
    encoding_legend(axes[0], target=target, loc="lower right")

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SisterPCAlignment.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    print(crossing_table(res_N, "N", "Fst"))
    print(crossing_table(res_H, "H", "Fst"))
    return fig


if __name__ == "__main__":
    make_figure()

# %%
