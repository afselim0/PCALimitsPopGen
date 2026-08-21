#%%
"""
The measured sister eigenvalue against theory: exact formula (A) and approximation (B).

Both panels plot nu_meas (median over replicates, from src/analysis/SisterEigenvalue.py)
against a theoretical value, one point per cell across every sweep family:

A  against nu_exact, the structure eigenvalue of the full 4x4 OOA3 coalescent-time matrix
   (exact_sister_nu.exact_sister_nu) -- no approximation, so this tests whether measurement
   recovers theory.
B  against nu_pred = N*H*Fst, the harmonic-mean approximation (eps_a=0, tau/tau_bar=1) --
   this tests what that approximation costs on top of the exact-theory agreement in A.

Both share the same y=x dotted reference line and log-log axes; the gap between A and B is
the approximation error alone, since A already controls for measurement noise.

Reads results/analysis/SisterEigenvalue, writes results/figures/SisterEigenvalueVerification.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from plotting_style import okabe_ito, setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SisterEigenvalue")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

C_MEAS = okabe_ito[1]        # vermillion
FAMILY_MARKERS = [("Fst", "o"), ("N", "s"), ("H", "^"), ("scatter", "D")]


def panel(ax, s, x_col, x_label, title):
    """nu_meas (y) against a theoretical nu (x), one point per cell, y=x dotted."""
    lim = [min(s[x_col].min(), s.nu_meas.min()) * 0.8,
           max(s[x_col].max(), s.nu_meas.max()) * 1.2]
    ax.plot(lim, lim, ':', color='0.35', linewidth=1.4, zorder=1, label=r'$y=x$')
    for fam, marker in FAMILY_MARKERS:
        d = s[s.family == fam]
        if d.empty:
            continue
        ax.scatter(d[x_col], d.nu_meas, s=24, marker=marker, facecolors='none',
                  edgecolors=C_MEAS, linewidths=1.1, zorder=3, label=fam)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'measured $\nu_1$ (median)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, frameon=False)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    s = pd.read_csv(os.path.join(analysis_dir, "summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))

    panel(axes[0], s, "nu_exact", r'exact $\nu_1$ (from $\hat{F}$)', 'exact formula')
    panel(axes[1], s, "nu_pred", r'approximate $\nu_1 = N H F_{ST}$', 'approximate formula')

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SisterEigenvalueVerification.pdf")
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
