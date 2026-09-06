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

Both share the same y=x dotted reference line, log-log axes, and y-axis (measured nu_3); the
gap between A and B is the eps_0 = 0, tau/tau_bar = 1 assumption's cost alone, since A
already controls for measurement noise. B's legend is omitted since its series are the same
ones already keyed in A.

Each series states exactly which parameters produced it, as the fixed values plus the set
varied, e.g. "Fst = {0.0016, 0.0034, ...}, N=1000, phi_tilde=0.250" for the family that
varies Fst -- rather than a bare family name. build_designs() (src/analysis/
SisterEigenvalue.py) fixes two of (Fst, N, phi_tilde) and varies the third per family,
except "H", which is two sub-designs at once (H_A fixes N, H_B fixes phi_1=phi_2 and so
varies N along with phi_tilde) and is therefore split into two series here.

Reads results/analysis/SisterEigenvalue, writes results/figures/EigenvalueAfricanSplit.pdf
"""
import os

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import pandas as pd

from crossing_panel import ENCODING_KEY_SIZE, PANEL_LETTER_SIZE, figsize, save_figure
from plotting_style import okabe_ito, setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SisterEigenvalue")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

C_MEAS = okabe_ito[1]        # vermillion
NAMES = {"Fst": r"F_{ST}", "N": "N", "H": r"\tilde\phi"}

# (series filter, varied column, fixed columns, marker). "H" is two series because H_B
# varies N along with phi_tilde (s = phi_1+phi_2 varies at fixed phi_1=phi_2), so it is not
# a clean one-factor sweep like the others -- see build_designs().
# (row selector, varied parameter, columns to report as fixed, marker, extra note).
#
# The two H families both vary phi-tilde at the same Fst, and without the note their labels
# differ only in whether N is a value or a set -- which understates what separates them.
# A holds n1 + n2 = 500 and makes the sisters UNEQUAL (n1 runs 50..250 against n2 = 500-n1),
# so N stays 1000; B keeps the sisters EQUAL and grows their share, so N runs 650..1400.
# Since phi-tilde = 2 phi_1 phi_2 / s, A varies the numerator at fixed denominator and B
# varies both at phi_1 = phi_2; run together they pin the dependence on the harmonic mean
# itself rather than on phi_1 phi_2 or on the sisters' share alone.
SERIES = [
    (lambda s: s.family == "Fst", "Fst", ["N", "H"], "o", None),
    (lambda s: s.family == "N", "N", ["Fst", "H"], "s", None),
    (lambda s: s.design.str.startswith("H_A"), "H", ["Fst", "N"], "^",
     r"$\phi_1 \neq \phi_2$"),
    (lambda s: s.design.str.startswith("H_B"), ("H", "N"), ["Fst"], "v",
     r"$\phi_1 = \phi_2$"),
    (lambda s: s.family == "scatter", None, [], "D", None),
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


def _series_label(d, varied, fixed_cols, note=None):
    """
    What this series varies, with the others held at their fixed value: 'varying Fst
    (N=1000, phi_tilde=0.25)'.

    Names the varied parameter rather than enumerating its values. The full sets were
    spelled out previously ("Fst = {0.0149, 0.0154, ...}, N=1000, phi=0.25"), which made a
    single label ~5 in wide -- five of them could not be laid out under a 6.5 in figure at
    any column count, and bbox_inches="tight" grew the figure to 24 in to contain them. The
    values themselves are readable off the axes; which parameter moves is what the legend
    has to say. A column that is not constant within the series is still reported as a set,
    since claiming a fixed value the data does not hold would be wrong.

    `note` is appended inside the parentheses for a series whose reported fixed values do
    not by themselves say what makes it distinct -- the two phi-tilde families, which are
    told apart by whether the two sisters are equally sampled, not by N.
    """
    # `varied` may name more than one column: family B moves phi-tilde by adding samples to
    # the sisters, so N rises with it. Naming both ("varying phi-tilde & N") says that
    # directly, where reporting N as the set it takes only listed the consequence.
    varied_cols = [varied] if isinstance(varied, str) else list(varied)
    varied_part = "varying " + " & ".join(rf"${NAMES[c]}$" for c in varied_cols)
    fixed_parts = []
    for c in ["Fst", "N", "H"]:
        if c in varied_cols:
            continue
        elif c in fixed_cols:
            fixed_parts.append(rf"${NAMES[c]}$={_fmt(d[c].iloc[0])}")
        else:
            fixed_parts.append(rf"${NAMES[c]}$ = {_values_str(d[c], max_shown=3)}")
    if note:
        fixed_parts.append(note)
    return f"{varied_part} ({', '.join(fixed_parts)})" if fixed_parts else varied_part


def panel(ax, s, x_col, x_label, title, lim, show_legend=True):
    """nu_meas (y) against a theoretical nu (x), one point per cell, y=x dotted."""
    ax.plot(lim, lim, ':', color='0.35', linewidth=1.4, zorder=1, label=r'$y=x$')
    for select, varied, fixed_cols, marker, note in SERIES:
        d = s[select(s)]
        if d.empty:
            continue
        label = "scatter (N, $\\tilde\\phi$, $F_{ST}$ all varied)" if varied is None \
            else _series_label(d, varied, fixed_cols, note)
        ax.scatter(d[x_col], d.nu_meas, s=24, marker=marker, facecolors='none',
                  edgecolors=C_MEAS, linewidths=1.1, zorder=3, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'measured $\nu_3$ (median)')
    ax.set_title(title)
    if show_legend:
        # Returned, not drawn here: the figure places one key centred under both panels.
        pass


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    s = pd.read_csv(os.path.join(analysis_dir, "summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=figsize(5.2, 3.5), sharey=True)

    # One shared limit for both panels (and both axes within each, since each panel is
    # already square in log-log) so A and B are directly comparable at a glance.
    lo = min(s.nu_analytical.min(), s.nu_pred.min(), s.nu_meas.min()) * 0.8
    hi = max(s.nu_analytical.max(), s.nu_pred.max(), s.nu_meas.max()) * 1.2
    lim = [lo, hi]

    panel(axes[0], s, "nu_analytical",
         r'exact $\nu_3 = \epsilon_0 + N\tilde{\phi}'
         r'\dfrac{\tau}{\bar\tau} F_{ST}$', 'exact formula', lim)
    panel(axes[1], s, "nu_pred", r'approximate $\nu_3 = N \tilde{\phi} F_{ST}$', 'approximate formula',
          lim, show_legend=False)

    # Square axes boxes. The shared log-log limits make the two panels comparable, but the
    # BOX only comes out square if it is set: at 6.5 in wide the two panels were wide
    # rectangles.
    for ax in axes:
        ax.set_box_aspect(1.0)

    for ax, letter in zip(axes, "AB"):
        # Offset in POINTS, not axes fraction: a fraction of a 2.6 in panel is much
        # less page than a fraction of the old wide one, so the fractional offset put
        # panel B's letter inside panel A.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -22 / 72, 12 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight='bold', va='top')

    # Both panels show the same series, so one key centred under the row serves both.
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    # Two columns, not one row: these labels spell out the whole swept set
    # ("F_ST={0.0149, ...}, N=1000, phi=0.25"), so a single row of five is ~24 in wide and
    # bbox_inches="tight" blows the figure up to match. Stacked, and at the key size rather
    # than the 11 pt of a data legend, since this is a definition of the marks.
    # The x labels are the two formulas themselves, so they are TALL (a built-up fraction
    # in panel A) and hang well below the axes. The key has to clear them: it is dropped to
    # the figure foot and the bottom margin opened to match, rather than sitting at 0.02
    # where its top row ran straight through both labels.
    fig.subplots_adjust(bottom=0.30)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncol=2, frameon=False, fontsize=ENCODING_KEY_SIZE - 2,
               handlelength=1.2, columnspacing=0.8, labelspacing=0.35,
               handletextpad=0.4)
    out = save_figure(fig, figure_dir, "EigenvalueAfricanSplit")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
