#%%
"""
Closed-form stepping-stone eigenvalue against the measured one (supplementary figure).

a  1D (K = 25 demes)
b  2D (K x K = 9 x 9 demes; nu_theory is the degenerate pair's shared value, nu_meas the mean
   eigenvalue of the two sample eigenvectors matched to it)

Median measured eigenvalue (src/analysis/SteppingStoneEvenSampling.py) against
nu_1 = n / (4 M sin^2(pi/2K)); colour = n per deme, marker = M (a few representative values
per panel). Only cells with median cos^2 > MIN_COS are drawn: below that the matched sample
eigenvalue is not an estimate of nu_1.

Reads results/analysis/SteppingStoneEvenSampling/even_{1D,2D}.csv
Writes results/figures/supplement/EigenvalueSS.pdf
"""
import os

import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

import genetics_style as gs

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SteppingStoneEvenSampling")

W, H = 130, 75
AX, LEFT, GAP, BOTTOM = 47, 13, 19, 24
NU1_LABEL = r"$\nu_1 = n\,/\,(2M\sin^2(\pi/2K))$"   # n in individuals
NUM_M_LEVELS = 5
MARKERS = ["o", "s", "^", "D", "v"]
N_PER_DEME = [4, 8, 16, 32, 64]     # as stored (haplotypes); one individual has no floor
MIN_COS = 0.75
COLORS = gs.seq_colors(len(N_PER_DEME))


def representative_M(df, num_levels=NUM_M_LEVELS):
    """num_levels distinct simulated M values nearest to log-spaced targets."""
    m_available = np.sort(df.M.unique())
    chosen = []
    for t in np.geomspace(m_available.min(), m_available.max(), num_levels):
        j = next(int(j) for j in np.argsort(np.abs(m_available - t)) if j not in chosen)
        chosen.append(j)
    return m_available[np.sort(chosen)]


def _fmt_M(v):
    v = float(v)
    if v >= 1000:
        e = int(np.floor(np.log10(v)))
        return rf"${v / 10.0 ** e:.1f} \times 10^{{{e}}}$"
    return f"{v:.0f}" if v == int(v) else f"{v:.3g}"


def panel(ax, df):
    """nu_meas vs nu_theory; returns the M-key handles."""
    df = df[df.n_per_deme.isin(N_PER_DEME) & (df.median_cos > MIN_COS)]
    m_values = representative_M(df)
    d = df[df.M.isin(m_values)]
    lim = [min(d.nu_theory.min(), d.nu_meas.min()) * 0.8,
           max(d.nu_theory.max(), d.nu_meas.max()) * 1.2]
    ax.plot(lim, lim, ":", color="0.35", linewidth=0.8, zorder=1)
    for n_per, c in zip(N_PER_DEME, COLORS):
        for m, mk in zip(m_values, MARKERS):
            cell = d[(d.n_per_deme == n_per) & (d.M == m)]
            ax.scatter(cell.nu_theory, cell.nu_meas, s=14, color=c, marker=mk,
                       edgecolors="none", alpha=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(NU1_LABEL)
    ax.set_ylabel(r"measured $\nu_1$ (median)")
    return [Line2D([], [], marker=mk, linestyle="none", color="0.3", markersize=3.5,
                   label=_fmt_M(m)) for m, mk in zip(m_values, MARKERS)]


def make_figure(analysis_dir=ANALYSIS_DIR):
    gs.setup()
    fig = gs.blank_figure(W, H)
    lefts = [LEFT, LEFT + AX + GAP]
    axes, m_keys = [], []
    for name, left, letter in zip(["even_1D.csv", "even_2D.csv"], lefts, "ab"):
        ax = gs.axes_mm(fig, left, BOTTOM, AX, AX)
        m_handles = panel(ax, pd.read_csv(os.path.join(analysis_dir, name)))
        axes.append(ax)
        m_keys.append((m_handles, [h.get_label() for h in m_handles]))
        gs.label_panel_mm(fig, letter, left - 12, H - 1)
    # M keys: one row centred under each panel, matched spacing; y = x inside each panel
    gs.matched_row_keys(axes, m_keys, 10.5, [r"$M$", r"$M$"])
    y_eq_x = Line2D([], [], linestyle=":", color="0.35", linewidth=0.8, label=r"$y = x$")
    for ax in axes:
        ax.add_artist(Legend(ax, [y_eq_x], [y_eq_x.get_label()], loc="upper left",
                             handlelength=1.6))

    # n key: colours shared by both panels, centred under the pair of panels
    n_handles = [Line2D([], [], marker="o", linestyle="none", color=c, markersize=3.5,
                        label=f"{gs.individuals(n):g}") for n, c in zip(N_PER_DEME, COLORS)]
    fig.legend(handles=n_handles, title="individuals per deme $n$", loc="lower center",
               bbox_to_anchor=((LEFT + AX + GAP / 2) / W, 0.8 / H), ncol=len(n_handles),
               borderaxespad=0, borderpad=0, handlelength=1.2, handletextpad=0.3,
               columnspacing=1.0)
    out = gs.save(fig, "EigenvalueSS", "supplement")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
