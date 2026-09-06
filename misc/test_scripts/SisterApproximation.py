#%%
"""
What the harmonic-mean approximation costs, measured against the exact eigenvalue.

Both quantities are theoretical -- the exact structure eigenvalue of G = -H T H / T_bar
from the full OOA3 coalescent-time matrix, and the approximation nu_1 = N*H*Fst derived
under eps_a = 0 -- so this is a noise-free comparison of a formula against its own exact
value, not a simulation check.

The error is **not** a single multiplicative factor. Across 900 sweep cells,

    nu_exact = 1.379 * (N H Fst) + 0.344        R^2 = 0.99988

two separable corrections, both traceable to eps_a != 0:

  * multiplicative 1.379 -- the sisters sit at tau/tau_bar = 1.166, 1.168 against CEU/CHB at
    0.855, 0.811, so the eps_a weighting rescales the Fst-driven term.
  * additive 0.344 -- a pedestal surviving as Fst -> 0. YRI1 and YRI2 differ from the
    *sample* mean even when identical to each other, so the sister contrast keeps a
    nonzero eigenvalue at zero split depth.

Plotting the ratio alone hides this: an additive offset renders as a divergence wherever
N*H*Fst is small, which is why the ratio looked like a design-dependent 1.10-1.99 factor.
Panel A shows the decomposition directly; the ratio panels are kept but read as its
consequence rather than as the primary finding.

A  the fit -- nu_exact against N*H*Fst over every cell, with the regression and the
   pedestal marked. This is the result; the rest is where it comes from.
B  vs H   at fixed N, the sisters sharing a fixed sample budget. The ratio diverges as
   H -> 0 because the pedestal dominates there, not because the slope changes.
C  vs N   at fixed H, same story along the other axis.
D  absolute values vs H, showing both curves parallel on log-log -- the approximation has
   the right shape and the wrong offset.

Reads results/analysis/SisterApproximation, writes misc/test_figures/SisterApproximation.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# This script lives outside results/plotting_scripts, so that directory has to be put on
# the path explicitly for the shared plotting helpers.
sys.path.append(os.path.join(_PROJECT_ROOT, "results", "plotting_scripts"))
from crossing_panel import save_figure                              # noqa: E402
from plotting_style import okabe_ito, setup_plotting_style          # noqa: E402

ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SisterApproximation")
# A side comparison rather than a paper figure, so it writes to the git-ignored misc tree.
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "misc", "test_figures")

C_RATIO = okabe_ito[1]      # vermillion
C_EXACT = okabe_ito[0]      # blue
C_PRED = okabe_ito[2]       # green


def panel_ratio(ax, d, x, xlabel, title, logx=False):
    """The ratio nu_exact / (N H Fst), with 1 marked as exact agreement."""
    d = d.sort_values(x)
    ax.axhline(1.0, ls="--", color="0.45", linewidth=1.1, zorder=1,
               label="exact agreement")
    ax.plot(d[x], d.ratio, "-o", color=C_RATIO, linewidth=1.5, markersize=4.5,
            markerfacecolor="none", markeredgewidth=1.1, zorder=3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\nu_{\mathrm{exact}} \,/\, N H F_{ST}$")
    ax.set_title(title)


def panel_values(ax, d, x, xlabel, title, logx=False, logy=False):
    """Both eigenvalues on the same axes, so the gap is visible in absolute terms."""
    d = d.sort_values(x)
    ax.plot(d[x], d.nu_exact, "-o", color=C_EXACT, linewidth=1.5, markersize=4.5,
            markerfacecolor="none", markeredgewidth=1.1, label=r"exact $\nu_1$", zorder=3)
    ax.plot(d[x], d.nu_pred, "-s", color=C_PRED, linewidth=1.5, markersize=4.5,
            markerfacecolor="none", markeredgewidth=1.1, label=r"$N H F_{ST}$", zorder=3)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\nu_1$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)


def panel_decomposition(ax, sweep_csv):
    """
    nu_exact against N*H*Fst over every sweep cell, with the linear fit.

    Uses the 900-cell vary_N sweep rather than the small theory grid: it spans two orders of
    magnitude in Fst, which is what makes the intercept identifiable.
    """
    d = pd.read_csv(sweep_csv)
    slope, intercept = np.polyfit(d.nu_approx, d.nu, 1)
    r2 = np.corrcoef(d.nu_approx, d.nu)[0, 1] ** 2

    ax.scatter(d.nu_approx, d.nu, s=7, facecolors="none", edgecolors=C_RATIO,
               linewidths=0.5, alpha=0.45, zorder=2)
    grid = np.linspace(0, d.nu_approx.max() * 1.05, 100)
    ax.plot(grid, slope * grid + intercept, "-", color=C_EXACT, linewidth=1.8, zorder=4,
            label=rf"${slope:.3f}\,\nu_{{\rm approx}} + {intercept:.3f}$")
    ax.plot(grid, grid, "--", color="0.45", linewidth=1.2, zorder=3,
            label=r"$\nu_{\rm exact} = N H F_{ST}$")
    ax.axhline(intercept, color=C_PRED, linewidth=1.0, ls=":", zorder=3,
               label=rf"pedestal $= {intercept:.3f}$")

    ax.set_xlabel(r"$N H F_{ST}$")
    ax.set_ylabel(r"exact $\nu_1$")
    ax.set_title(rf"the decomposition  ($R^2={r2:.5f}$)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    df = pd.read_csv(os.path.join(analysis_dir, "approximation.csv"))
    sweep_csv = os.path.join(os.path.dirname(analysis_dir), "SisterPCAlignment",
                             "vary_N.csv")

    d_N = df[df.sweep == "vary_N"]
    d_H = df[df.sweep == "vary_H"]

    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.8))

    panel_decomposition(axes[0], sweep_csv)

    panel_ratio(axes[1], d_H, "H", r"$H = 2\phi_1\phi_2/(\phi_1+\phi_2)$",
                rf"varying $H$  ($N={int(d_H.N.iloc[0])}$, "
                rf"$n_1{{+}}n_2={int(d_H.n_sister1.iloc[0] + d_H.n_sister2.iloc[0])}$)",
                logx=True)
    axes[1].legend(loc="upper right", fontsize=8.5, frameon=False)

    panel_ratio(axes[2], d_N, "N", r"$N$",
                rf"varying $N$  ($H\approx{d_N.H.median():.3f}$)", logx=True)

    panel_values(axes[3], d_H, "H", r"$H$", "absolute values, varying $H$",
                 logx=True, logy=True)

    for ax, letter in zip(axes, "ABCD"):
        ax.text(-0.18, 1.08, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    out = save_figure(fig, figure_dir, "SisterApproximation")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
