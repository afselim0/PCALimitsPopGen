#%%
"""
The variance-effective SNP count L_eff on panmictic genotypes (supplementary figure).

a  L_eff against raw L, one line per sample size N, with the L_eff = L identity.
b  L_eff against N at the largest L: the slope of a, read directly.

Reads results/analysis/EffectiveL/summary.csv
Writes results/figures/supplement/EffectiveL.pdf
"""
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import genetics_style as gs

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "EffectiveL")

W, H = 130, 68            # figure size (mm)
AX_W, AX_H = 47, 47       # axes size (mm)
LEFT, GAP, BOTTOM = 13, 19, 18


def _thousands(ax, axis):
    # tick labels in thousands, unit stated in the axis label
    fmt = lambda v, _: f"{v / 1e3:g}"
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)


def panel_a(ax, summary, colors):
    N_values = np.sort(summary.N.unique())
    L_grid = np.array([0, summary.L_requested.max()])
    ax.plot(L_grid, L_grid, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.30 * L_grid[1], 0.36 * L_grid[1], r"$L_{\mathrm{eff}} = L$", color="0.35",
            ha="right", va="bottom")
    for N, c in zip(N_values, colors):
        d = summary[summary.N == N].sort_values("L_requested")
        ax.plot(d.L_requested, d.L_eff, "-", color=c, linewidth=1.0, zorder=2)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    _thousands(ax, "x")
    _thousands(ax, "y")
    ax.set_xlabel(r"$L$ (independent SNPs, $\times 10^3$)")
    ax.set_ylabel(r"$L_{\mathrm{eff}}$ ($\times 10^3$)")


def panel_b(ax, summary, colors):
    N_values = np.sort(summary.N.unique())
    L_max = summary.L_requested.max()
    d = summary[summary.L_requested == L_max].sort_values("N")
    ax.plot(gs.individuals(d.N), d.L_eff, "-", color="0.5", linewidth=0.8, zorder=1)
    cmap = dict(zip(N_values, colors))
    ax.scatter(gs.individuals(d.N), d.L_eff, s=12, c=[cmap[n] for n in d.N], zorder=2, linewidths=0)
    ax.set_ylim(0, d.L_eff.max() * 1.08)
    ax.set_xlim(0, None)
    _thousands(ax, "x")
    _thousands(ax, "y")
    ax.set_xlabel(r"$N$ (individuals, $\times 10^3$)")
    ax.set_ylabel(r"$L_{\mathrm{eff}}$ ($\times 10^3$)")


def make_figure(analysis_dir=ANALYSIS_DIR):
    gs.setup()
    summary = pd.read_csv(os.path.join(analysis_dir, "summary.csv"))
    N_values = np.sort(summary.N.unique())
    colors = gs.seq_colors(len(N_values))

    fig = gs.blank_figure(W, H)
    ax_a = gs.axes_mm(fig, LEFT, BOTTOM, AX_W, AX_H)
    ax_b = gs.axes_mm(fig, LEFT + AX_W + GAP, BOTTOM, AX_W, AX_H)
    panel_a(ax_a, summary, colors)
    panel_b(ax_b, summary, colors)

    handles = [Line2D([], [], color=c, linewidth=1.0, marker="o", markersize=3,
                      markeredgewidth=0) for c in colors]
    fig.legend(handles, [f"{gs.individuals(N):g}" for N in N_values], title=r"$N$ (individuals)",
               loc="lower center", bbox_to_anchor=((LEFT + AX_W + GAP / 2) / W, 0.5 / H),
               ncol=len(N_values), columnspacing=0.9, handlelength=1.2)
    gs.label_panel_mm(fig, "a", 1, H - 1)
    gs.label_panel_mm(fig, "b", LEFT + AX_W + GAP - 12, H - 1)
    out = gs.save(fig, "EffectiveL", "supplement")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
