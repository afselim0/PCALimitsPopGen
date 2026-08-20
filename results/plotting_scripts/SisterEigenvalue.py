#%%
"""
Testing nu_1 = N*H*Fst for a recently split sister pair, H the harmonic mean of phi_1, phi_2.

The prediction is a product of three factors, so it factorizes into claims that can be
tested one at a time rather than over a 4-D grid. H = 2*phi_1*phi_2/(phi_1+phi_2) is the
harmonic mean of the two sister sampling fractions, and reduces to the K = 2 factor
2*phi*(1-phi) when the sisters are the whole sample:

A  collapse -- nu_meas against nu_pred across every cell, however generated. All points on
   the identity line, whatever combination of N, H and Fst produced them. This is the
   whole test in one panel; the heterogeneity of the cells is what makes it strong.
B  Fst -- nu against Fst at fixed N and H.
C  N   -- nu against N at fixed Fst and H.
D  H   -- nu against H at fixed N and Fst, the discriminating panel. Two sampling families
   are shown: one holds s = phi_1+phi_2 fixed while varying phi_1*phi_2, the other holds
   phi_1 = phi_2 while varying s. An alternative sampling factor that agrees along one
   family will depart along the other, which is what separates them.

Each panel draws the zero-parameter prediction as a line -- nothing is fitted anywhere.

Reads results/analysis/SisterEigenvalue, writes results/figures/SisterEigenvalue.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_style import okabe_ito, setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SisterEigenvalue")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

C_MEAS = okabe_ito[1]        # vermillion: measurement
C_PRED = okabe_ito[0]        # blue: the prediction
C_FAM_B = okabe_ito[2]       # green: the second sampling family in panel D


def panel_collapse(ax, s):
    """nu_meas vs nu_pred over every cell."""
    lim = [min(s.nu_pred.min(), s.nu_meas.min()) * 0.8,
           max(s.nu_pred.max(), s.nu_meas.max()) * 1.2]
    ax.plot(lim, lim, '-', color=C_PRED, linewidth=1.6, zorder=1,
            label=r'$\nu_{\mathrm{meas}} = \nu_{\mathrm{pred}}$')
    for fam, marker in [("Fst", "o"), ("N", "s"), ("H", "^"), ("scatter", "D")]:
        d = s[s.family == fam]
        if d.empty:
            continue
        ax.scatter(d.nu_pred, d.nu_meas, s=22, marker=marker, facecolors='none',
                   edgecolors=C_MEAS, linewidths=1.0, zorder=3, label=fam)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r'predicted $\nu_1 = N H F_{ST}$')
    ax.set_ylabel(r'measured $\nu_1$')
    ax.set_title('all cells')
    ax.legend(loc='upper left', fontsize=7.5, frameon=False)


def _one_factor(ax, d, x, xlabel, title):
    """A one-factor panel: measurement as points, prediction as a line."""
    d = d.sort_values(x)
    ax.plot(d[x], d.nu_pred, '-', color=C_PRED, linewidth=1.6, zorder=2,
            label='prediction')
    ax.scatter(d[x], d.nu_meas, s=26, facecolors='none', edgecolors=C_MEAS,
               linewidths=1.1, zorder=3, label='simulation')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\nu_1$')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, frameon=False)


def panel_H(ax, s, cells):
    """
    nu vs H, with the two sampling families.

    Family A fixes s and varies phi_1*phi_2; family B fixes phi_1 = phi_2 and varies s. An
    alternative that happens to agree along one family will disagree along the other, so
    plotting both is what makes this panel discriminating rather than merely consistent.
    """
    d = s[s.family == "H"].copy()
    if d.empty:
        return
    fam = d.design.str.startswith("H_A")

    for mask, color, label, marker in [
            (fam, C_MEAS, r'family A (fixed $s$)', 'o'),
            (~fam, C_FAM_B, r'family B ($\phi_1=\phi_2$)', '^')]:
        g = d[mask].sort_values("H")
        if g.empty:
            continue
        ax.scatter(g.H, g.nu_meas, s=28, marker=marker, facecolors='none',
                   edgecolors=color, linewidths=1.1, zorder=3, label=label)

    grid = np.linspace(d.H.min() * 0.9, d.H.max() * 1.05, 100)
    # The prediction is linear through the origin in H, at the N and Fst of family A
    # (family B varies N, so its points are not expected to lie on this same line).
    a = d[fam]
    if not a.empty:
        N0, Fst0 = a.N.iloc[0], a.Fst.iloc[0]
        ax.plot(grid, N0 * grid * Fst0, '-', color=C_PRED, linewidth=1.6, zorder=2,
                label=r'$N H F_{ST}$ (family A)')

    ax.set_xlabel(r'$H = 2\phi_1\phi_2/(\phi_1+\phi_2)$')
    ax.set_ylabel(r'$\nu_1$')
    ax.set_title(r'varying $H$')
    ax.legend(loc='upper left', fontsize=7.5, frameon=False)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    cells = pd.read_csv(os.path.join(analysis_dir, "cells.csv"))
    s = pd.read_csv(os.path.join(analysis_dir, "summary.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8))

    panel_collapse(axes[0], s)

    d_fst = s[s.family == "Fst"]
    if not d_fst.empty:
        _one_factor(axes[1], d_fst, "Fst", r'$F_{ST}$',
                    rf'varying $F_{{ST}}$  ($N={int(d_fst.N.iloc[0])}$, '
                    rf'$H={d_fst.H.iloc[0]:.3f}$)')

    d_N = s[s.family == "N"]
    if not d_N.empty:
        _one_factor(axes[2], d_N, "N", r'$N$',
                    rf'varying $N$  ($H={d_N.H.iloc[0]:.3f}$)')

    panel_H(axes[3], s, cells)

    for ax, letter in zip(axes, "ABCD"):
        ax.text(-0.18, 1.08, letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SisterEigenvalue.pdf")
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
