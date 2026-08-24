#%%
"""
Spatial-axis PC recovery vs M for the recent-split demography, two panels.

A  n_per_deme_B varied at fixed phi=0.5 (B's share of N held at half).
B  phi varied at fixed n_per_deme_B=4.

Each level is a curve fitted through every replicate (not through their medians) with a
bootstrap band. The comparison against theory is a pair of vertical lines rather than a
prediction drawn at every x: a thin solid line at the 1/M where the fit reaches squared
overlap 0.9, and a dashed line at the 1/M where the theory does.

The x axis is 1/M rather than M: overlap grows as migration weakens, and the transition
fit expects an increasing curve. nu_1 comes from the analysis (split_stepping_stone_theory
via the phase-type coalescent calculation, not the equilibrium formula -- the split is too
recent for equilibrium) and is rescaled as 1/M at fixed design.

Reads results/analysis/RecentSpatialOverlapVsM/overlap_replicates.csv, writes
results/figures/RecentSpatialOverlapVsM.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from crossing_panel import crossing_panel, crossing_table, encoding_legend
from plotting_style import setup_plotting_style
from SteppingStoneOverlapVsM import flat_overlap

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RecentSpatialOverlapVsM")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP = cm.viridis
CMAP_PHI = cm.plasma
TARGET = 0.9


def theory_for(df, level_key):
    """
    level -> (1/M -> predicted squared overlap), for the dashed vertical lines.

    nu_1 is read from the analysis rather than recomputed: for this demography it comes
    from a sparse phase-type coalescent calculation over the recent split, which the
    plotting script has no business redoing. It scales as 1/M at fixed sampling design,
    so it rescales from the cell it was recorded at.
    """
    def make(level):
        d = df[df[level_key] == level]
        if d.empty or "nu_1" not in d:
            return None
        gamma = float(d.gamma.median())
        ref = d.iloc[0]
        nu_ref, inv_M_ref = float(ref.nu_1), 1.0 / float(ref.M)
        if not np.isfinite(nu_ref) or inv_M_ref <= 0:
            return None
        return lambda inv_M: flat_overlap(nu_ref * inv_M / inv_M_ref, gamma)
    return make


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    setup_plotting_style()
    df = pd.read_csv(os.path.join(analysis_dir, "overlap_replicates.csv"))
    df["inv_M"] = 1.0 / df["M"]
    df_n = df[df.sweep == "vary_n"]
    df_phi = df[df.sweep == "vary_phi"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    res_n = crossing_panel(axes[0], df_n, "n_per_deme_B", "inv_M", CMAP,
                           r"$n_{B,\mathrm{per\,deme}}$", target=target,
                           theory_overlap=theory_for(df_n, "n_per_deme_B"))
    res_phi = crossing_panel(axes[1], df_phi, "phi", "inv_M", CMAP_PHI, r"$\phi$",
                             target=target, theory_overlap=theory_for(df_phi, "phi"))

    for ax in axes:
        ax.set_xlabel(r"$1/M$")
    axes[0].set_title(r"fixed $\phi=0.5$")
    axes[1].set_title(r"fixed $n_{B,\mathrm{per\,deme}}=4$")
    encoding_legend(axes[0], target=target, loc="lower right")

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "RecentSpatialOverlapVsM.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    print(crossing_table(res_n, "n_B", "1/M"))
    print(crossing_table(res_phi, "phi", "1/M"))
    return fig


if __name__ == "__main__":
    make_figure()

# %%
