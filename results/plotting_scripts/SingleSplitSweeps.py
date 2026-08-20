#%%
"""
PC1 recovery across the two-population split sweep, against the zero-parameter prediction.

A  PC1 overlap vs Fst at ten sample sizes N (phi = 0.5).
B  PC1 overlap vs Fst at ten sampling fractions phi (N = 200).

In both panels the line is the BBP prediction

    |cos| = sqrt((1 - gamma/nu^2) / (1 + gamma/nu)),   gamma = N/L_eff,
    nu = 2*N*phi*(1-phi)*Fst

and the scatter is the measured median over replicates. The prediction is built from the
measured L_eff and the theoretical nu, so nothing in it is fitted -- the points either land
on the curve or they do not.

Colour encodes the swept parameter through a colourbar rather than a legend: ten levels
would otherwise need ten legend entries per panel, and the sweep is an ordered quantity,
which a sequential colour map conveys directly.

Reads results/analysis/SingleSplitSweeps, writes results/figures/SingleSplitSweeps.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SingleSplitSweeps")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis
CMAP_PHI = cm.plasma


def panel(ax, df, key, cmap, label, title):
    """
    One sweep: measured median as a line, zero-parameter prediction as scatter.

    Levels are coloured by the swept parameter on a shared normalization, so the colourbar
    reads directly onto the curves.
    """
    levels = np.sort(df[key].unique())
    norm = colors.Normalize(vmin=levels.min(), vmax=levels.max())

    for level in levels:
        d = df[df[key] == level].sort_values("Fst")
        color = cmap(norm(level))
        ax.plot(d.Fst, d.analytical_cos_axis0, "-", color=color, linewidth=1.4, zorder=2)
        # Measurements are subsampled: at every one of the 100 Fst values they would crowd
        # into a solid band and hide the curve they are meant to be compared against.
        s = d.iloc[::4]
        ax.scatter(s.Fst, s.cos_axis0, s=11, facecolors="none",
                   edgecolors=color, linewidths=0.9, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{ST}$")
    ax.set_ylabel(r"Cosine similarity: $|\langle \hat u_1, u_1\rangle|$")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title(title)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(label)
    cbar.outline.set_visible(False)
    return norm


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    vary_N = pd.read_csv(os.path.join(analysis_dir, "vary_N.csv"))
    vary_phi = pd.read_csv(os.path.join(analysis_dir, "vary_phi.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    n_reps = int(vary_N.num_replicates.iloc[0])
    phi_N = float(vary_N.phi_min.iloc[0])
    N_phi = int(vary_phi.N.iloc[0])

    # Report L_eff, not the requested L. Loci that are monomorphic in a subsample are
    # dropped, and how many that is depends on N: the realized raw count runs 12.6k at
    # N = 100 to 20k at N = 1000, so quoting the requested 20k would imply the sweep held
    # something fixed that it did not. L_eff is what the theory uses and is genuinely flat
    # across both sweeps (2.0% and 0.5% spread), because the dropped loci are the ones with
    # negligible p(1-p) that barely enter the participation ratio in the first place.
    L_eff_N = vary_N.L_eff.median()
    L_eff_phi = vary_phi.L_eff.median()

    panel(axes[0], vary_N, "N", CMAP_N, r"$N$",
          rf"varying $N$  ($\phi={phi_N:g}$, $L_{{\mathrm{{eff}}}}\approx{L_eff_N:.0f}$)")
    panel(axes[1], vary_phi, "phi_min", CMAP_PHI, r"$\phi$",
          rf"varying $\phi$  ($N={N_phi}$, $L_{{\mathrm{{eff}}}}\approx{L_eff_phi:.0f}$)")

    # One shared legend for the encoding, since colour is already spent on the sweep.
    axes[0].plot([], [], "-", color="0.35", linewidth=1.4, label="BBP prediction")
    axes[0].scatter([], [], s=11, facecolors="none", edgecolors="0.35", linewidths=0.9,
                    label=f"simulation (median of {n_reps})")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SingleSplitSweeps.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
