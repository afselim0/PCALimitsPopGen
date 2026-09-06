#%%
"""
Verify the closed-form stepping-stone eigenvalue against the measured one, 1D and 2D.

A single K per dimension (no K sweep), so the swept quantities are n_per_deme (color) and
M (marker shape): the median measured eigenvalue (from
src/analysis/SteppingStoneEvenSampling.py) against the closed-form nu_theory, at a handful
of representative M per n_per_deme in the even-sampling sweep.

A  1D
B  2D (nu_theory is the degenerate pair's shared value; nu_meas is the mean eigenvalue
   of the two sample eigenvectors matched to it by overlap)

Only cells whose measured cosine similarity exceeds MIN_COS are plotted: where the axis is
not recovered the matched sample eigenvalue is not an estimate of nu_1 (see MIN_COS).

Reads results/analysis/SteppingStoneEvenSampling, writes
results/figures/EigenvalueSS.pdf
"""
import os

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

from crossing_panel import PANEL_LETTER_SIZE, figsize, save_figure
from plotting_style import setup_plotting_style, okabe_ito

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SteppingStoneEvenSampling")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

NU1_LABEL = r'$\nu_1 = N / (M K \sin^2(\pi / 2K))$'
NUM_M_LEVELS = 5
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
N_PER_DEME = [2, 4, 8, 16, 32]
# Only cells whose eigenvector is actually recovered. Below this the sample eigenvector is
# a poor match for the theoretical axis, and the "measured" eigenvalue is then not an
# estimate of nu_1 at all -- it is the eigenvalue of whichever sample axis happened to be
# matched, biased low by the bulk it has not yet separated from. Including those points
# would show a spurious downward curvature that is an artefact of the matching, not a
# failure of the closed form.
MIN_COS = 0.75


def representative_M(df, num_levels=NUM_M_LEVELS):
    """
    num_levels simulated M values, log-spaced targets snapped to the nearest available M.

    Chosen from whatever `df` holds, so when it is the cosine-filtered frame the levels are
    drawn from the M range that actually survives the filter. Picking them from the full
    grid instead would put most levels at high M, where nothing is recovered, and leave the
    panel nearly empty.
    """
    m_available = np.sort(df.M.unique())
    targets = np.geomspace(m_available.min(), m_available.max(), num_levels)
    # Snap each target to the nearest M not already taken, so num_levels distinct levels
    # come back: two adjacent targets can otherwise snap to the same simulated M and the
    # legend silently loses a marker.
    chosen = []
    for t in targets:
        order = np.argsort(np.abs(m_available - t))
        for j in order:
            if j not in chosen:
                chosen.append(int(j))
                break
    return m_available[np.sort(chosen)]


def _fmt_M(v):
    """M as a plain number when it is one, else LaTeX 'm x 10^e' -- never '1.03e+03'."""
    v = float(v)
    if v >= 1000:
        e = int(np.floor(np.log10(v)))
        return rf"{v / 10.0 ** e:.1f} \times 10^{{{e}}}"
    return f"{v:.0f}" if v == int(v) else f"{v:.3g}"


def panel(ax, df, title):
    """nu_meas (median over replicates) vs nu_theory: color = n_per_deme, marker = M, y=x dotted."""
    df = df[df.n_per_deme.isin(N_PER_DEME) & (df.median_cos > MIN_COS)]
    n_values = np.sort(df.n_per_deme.unique())
    m_values = representative_M(df)
    color_of = {n: okabe_ito[i % len(okabe_ito)] for i, n in enumerate(n_values)}
    marker_of = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(m_values)}

    d = df[df.M.isin(m_values)]
    lim = [min(d.nu_theory.min(), d.nu_meas.min()) * 0.8,
           max(d.nu_theory.max(), d.nu_meas.max()) * 1.2]
    ax.plot(lim, lim, ':', color='0.35', linewidth=1.4, zorder=1, label=r'$y=x$')

    for n_per in n_values:
        for m in m_values:
            cell = d[(d.n_per_deme == n_per) & (d.M == m)]
            ax.scatter(cell.nu_theory, cell.nu_meas, s=28, color=color_of[n_per],
                      marker=marker_of[m], edgecolors='none', alpha=0.85, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(NU1_LABEL)
    ax.set_ylabel(r'measured $\nu_1$ (median)')
    ax.set_title(title)

    n_handles = [plt.Line2D([], [], marker='o', linestyle='none', color=color_of[n],
                            markersize=6, label=f'{n:g}') for n in n_values]
    m_handles = [plt.Line2D([], [], marker=marker_of[m], linestyle='none', color='0.3',
                            markersize=6, label=f'${_fmt_M(m)}$') for m in m_values]
    # y=x is NOT one of the n-per-deme colours, so it does not belong under that title --
    # listed there it read as another curve level. It goes in the M key's frame instead,
    # which has no such heading conflict. Sizes come from the shared table, not a local
    # number, so these keys match every other legend in the manuscript.
    leg1 = ax.legend(handles=n_handles, title=r'$n_{\mathrm{per\,deme}}$',
                     loc='upper left', frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=[*m_handles,
                       plt.Line2D([], [], linestyle=':', color='0.35', label=r'$y=x$')],
              title=r'$M$', loc='lower right', frameon=False)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    df_1d = pd.read_csv(os.path.join(analysis_dir, "even_1D.csv"))
    df_2d = pd.read_csv(os.path.join(analysis_dir, "even_2D.csv"))

    fig, axes = plt.subplots(1, 2, figsize=figsize(9.6, 4.4))
    panel(axes[0], df_1d, rf'1D  ($K={int(df_1d.K.iloc[0])}$)')
    panel(axes[1], df_2d, rf'2D  ($K={int(df_2d.K.iloc[0])}$)')

    for ax, letter in zip(axes, "AB"):
        # Offset in POINTS, not axes fraction: a fraction of a 2.6 in panel is much
        # less page than a fraction of the old wide one, so the fractional offset put
        # panel B's letter inside panel A.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -22 / 72, 12 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight='bold', va='top')

    fig.tight_layout()
    out = save_figure(fig, figure_dir, "EigenvalueSS")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
