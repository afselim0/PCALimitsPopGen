#%%
"""
The measured sister eigenvalue against theory: analytical formula (A) and approximation (B).

Both panels plot nu_meas (median over replicates, from src/analysis/SisterEigenvalue.py)
against a theoretical value, one point per cell across every sweep family:

A  against nu_analytical = eps_0 + (2*N*phi_1*phi_2/s)*(tau/tau_bar)*Fst
   (exact_sister_nu.analytical_nu), the closed-form sister eigenvalue that uses the sister
   pair's real within-population coalescent time and eps_0 rather than assuming them away --
   this tests whether measurement recovers the analytical prediction.
B  against nu_pred = N*H*Fst, the cruder harmonic-mean approximation that further assumes
   eps_0 = 0 and tau/tau_bar = 1 -- this isolates what that extra assumption costs on top of
   the agreement already shown in A.

Both share the same y=x dotted reference line, log-log axes, and y-axis (measured nu_1); the
gap between A and B is the eps_0 = 0, tau/tau_bar = 1 assumption's cost alone, since A
already controls for measurement noise. B's legend is omitted since its series are the same
ones already keyed in A.

Each series states exactly which parameters produced it, as the fixed values plus the set
varied, e.g. "Fst = {0.0016, 0.0034, ...}, N=1000, phi_tilde=0.250" for the family that
varies Fst -- rather than a bare family name. build_designs() (src/analysis/
SisterEigenvalue.py) fixes two of (Fst, N, phi_tilde) and varies the third per family,
except "H", which is two sub-designs at once (H_A fixes N, H_B fixes phi_1=phi_2 and so
varies N along with phi_tilde) and is therefore split into two series here.

Reads results/analysis/SisterEigenvalue, writes results/figures/SisterEigenvalueVerification.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from crossing_panel import save_figure
from plotting_style import okabe_ito, setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SisterEigenvalue")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

C_MEAS = okabe_ito[1]        # vermillion
NAMES = {"Fst": r"F_{ST}", "N": "N", "H": r"\tilde\phi"}

# (series filter, varied column, fixed columns, marker). "H" is two series because H_B
# varies N along with phi_tilde (s = phi_1+phi_2 varies at fixed phi_1=phi_2), so it is not
# a clean one-factor sweep like the others -- see build_designs().
SERIES = [
    (lambda s: s.family == "Fst", "Fst", ["N", "H"], "o"),
    (lambda s: s.family == "N", "N", ["Fst", "H"], "s"),
    (lambda s: s.design.str.startswith("H_A"), "H", ["Fst", "N"], "^"),
    (lambda s: s.design.str.startswith("H_B"), "H", ["Fst"], "v"),
    (lambda s: s.family == "scatter", None, [], "D"),
]


def _fmt(v):
    """3 significant figures, with whole numbers written out rather than in exponent form."""
    v = float(f"{float(v):.3g}")
    return f"{v:.0f}" if v == int(v) else f"{v:.3g}"


def _values_str(x, max_shown=10):
    """Unique values as '{v1, v2, ...}', truncated with an ellipsis count if long."""
    vals = sorted(set(float(f"{float(v):.3g}") for v in x))
    shown = ", ".join(_fmt(v) for v in vals[:max_shown])
    return "{" + shown + (", ..." if len(vals) > max_shown else "") + "}"


def _series_label(d, varied, fixed_cols):
    """
    What this series fixed and varied, e.g. 'Fst = {0.0016, ...}, N=1000, phi_tilde=0.250'.

    Fixed columns are reported as a single value; the varied column (and any column that
    is not constant within the series, such as N for H_B) is reported as the realized set,
    so the label always states exactly which cells produced this marker rather than
    asserting a fixed value that the data does not actually hold to.
    """
    parts = []
    for c in ["Fst", "N", "H"]:
        if c == varied:
            parts.append(rf"${NAMES[c]}$ = {_values_str(d[c])}")
        elif c in fixed_cols:
            parts.append(rf"${NAMES[c]}$={_fmt(d[c].iloc[0])}")
        else:
            parts.append(rf"${NAMES[c]}$ = {_values_str(d[c])}")
    return ", ".join(parts)


def panel(ax, s, x_col, x_label, title, lim, show_legend=True):
    """nu_meas (y) against a theoretical nu (x), one point per cell, y=x dotted."""
    ax.plot(lim, lim, ':', color='0.35', linewidth=1.4, zorder=1, label=r'$y=x$')
    for select, varied, fixed_cols, marker in SERIES:
        d = s[select(s)]
        if d.empty:
            continue
        label = "scatter (N, $\\tilde\\phi$, $F_{ST}$ all varied)" if varied is None \
            else _series_label(d, varied, fixed_cols)
        ax.scatter(d[x_col], d.nu_meas, s=24, marker=marker, facecolors='none',
                  edgecolors=C_MEAS, linewidths=1.1, zorder=3, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'measured $\nu_1$ (median)')
    ax.set_title(title)
    if show_legend:
        ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0), fontsize=6, frameon=False)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    s = pd.read_csv(os.path.join(analysis_dir, "summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)

    # One shared limit for both panels (and both axes within each, since each panel is
    # already square in log-log) so A and B are directly comparable at a glance.
    lo = min(s.nu_analytical.min(), s.nu_pred.min(), s.nu_meas.min()) * 0.8
    hi = max(s.nu_analytical.max(), s.nu_pred.max(), s.nu_meas.max()) * 1.2
    lim = [lo, hi]

    panel(axes[0], s, "nu_analytical",
         r'$\nu_1 = \epsilon_0 + N\tilde{\phi}'
         r'\dfrac{\tau}{\bar\tau} F_{ST}$', 'analytical formula', lim)
    panel(axes[1], s, "nu_pred", r'approximate $\nu_1 = N \tilde{\phi} F_{ST}$', 'approximate formula',
          lim, show_legend=False)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

    fig.tight_layout()
    out = save_figure(fig, figure_dir, "SisterEigenvalueVerification")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
