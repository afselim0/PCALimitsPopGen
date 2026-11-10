#%%
"""
Figure 4 (double column): sample PCs in the stepping-stone models at three regimes.

a  1D (K = 25 demes): the chain coloured by deme, then PC1 (with vertical jitter that
   carries no information) at no, weak and strong structure.
b  2D (9 x 9 demes): the grid coloured by position, then PC1 vs PC2 (Procrustes-aligned
   to the degenerate closed-form pair in the analysis script) at the same three regimes.

Regimes are chosen in the analysis script by the measured <u_hat, u>^2; each PC panel is
annotated with its M and that value.

Reads results/analysis/SteppingStoneEigenvectorShapes
Writes results/figures/main/DemoSS.pdf
"""
import os

import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

import genetics_style as gs

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_ROOT, "results", "analysis", "SteppingStoneEigenvectorShapes")

REGIMES = ["no_structure", "weak", "strong"]
COLOR_EDGE = "0.2"
JITTER_1D = 0.16
MARKER_S = 2.0

# Layout (mm)
W, H = gs.WIDTH["double"] / gs.MM, 72.0
PC_X = [56.0, 101.5, 147.0]
PC_W = 35.0
ROW_B = (8.0, 35.0)       # bottom, height of the 2D row (square panels)
ROW_A = (56.5, 8.5)       # bottom, height of the 1D strips
SCHEMATIC_X, SCHEMATIC_W = 4.0, 38.0


def fmt_M(v):
    v = float(v)
    if v >= 1000:
        e = int(np.floor(np.log10(v)))
        return rf"{v / 10.0 ** e:.1f} \times 10^{{{e}}}"
    return f"{v:.0f}" if v == int(v) else f"{v:.3g}"


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def annotate(ax, row):
    ax.text(0.5, 1.0, rf"$M = {fmt_M(row.M)}$,  $\langle\hat u,u\rangle^2 = {row.median_cos:.2f}$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=gs.LABEL_PT)


def schematic_1d(ax, K, n):
    ax.plot([0, K - 1], [0, 0], "-", color=COLOR_EDGE, linewidth=0.6, zorder=1)
    for j in range(K):
        ax.add_patch(Circle((j, 0), 0.42, facecolor=gs.chain_color(j, K),
                            edgecolor=COLOR_EDGE, linewidth=0.4, zorder=2))
    ax.set_xlim(-0.8, K - 0.2)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    blank(ax)


def schematic_2d(ax, K):
    for r in range(K):
        ax.plot([0, K - 1], [r, r], "-", color=COLOR_EDGE, linewidth=0.6, zorder=1)
        ax.plot([r, r], [0, K - 1], "-", color=COLOR_EDGE, linewidth=0.6, zorder=1)
    for r in range(K):
        for c in range(K):
            ax.add_patch(Circle((c, r), 0.30, facecolor=gs.grid_color(r, c, K),
                                edgecolor=COLOR_EDGE, linewidth=0.4, zorder=2))
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(-0.5, K - 0.5)
    ax.set_aspect("equal")
    blank(ax)


def panel_1d(ax, d, K, rng):
    y = rng.uniform(-JITTER_1D, JITTER_1D, len(d))
    ax.scatter(d.pc1, y, c=[gs.chain_color(j, K) for j in d.deme], s=MARKER_S,
               edgecolors="none", alpha=0.85, rasterized=True)
    ax.set_xlabel("PC1")
    ax.set_ylim(-1.6 * JITTER_1D, 1.6 * JITTER_1D)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(3, symmetric=True))
    ax.spines["left"].set_visible(False)


def panel_2d(ax, d, K):
    c = [gs.grid_color(int(r), int(cc), K) for r, cc in zip(d.row, d.col)]
    ax.scatter(d.pc1, d.pc2, c=c, s=MARKER_S, edgecolors="none", alpha=0.85,
               rasterized=True)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2", labelpad=1)
    lim = 1.08 * np.abs(np.r_[d.pc1, d.pc2]).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ticks = MaxNLocator(3, symmetric=True)
    ax.xaxis.set_major_locator(ticks)
    ax.yaxis.set_major_locator(ticks)


def make_figure(analysis_dir=ANALYSIS_DIR):
    gs.setup()
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index(["dimension", "regime"])
    rng = np.random.default_rng(0)
    K1, K2 = (int(meta.loc[(k, REGIMES[0])].K) for k in (1, 2))
    n1, n2 = (int(meta.loc[(k, REGIMES[0])].n_per_deme) for k in (1, 2))

    fig = gs.blank_figure(W, H)

    # a: 1D row
    ax = gs.axes_mm(fig, SCHEMATIC_X, ROW_A[0] + 1.0, SCHEMATIC_W, ROW_A[1])
    schematic_1d(ax, K1, n1)
    ax.text(0.5, -0.15, f"1D stepping stones\n$K = {K1}$ demes, $n = {n1}$ per deme",
            transform=ax.transAxes, ha="center", va="top", fontsize=gs.LABEL_PT)
    for x, regime in zip(PC_X, REGIMES):
        ax = gs.axes_mm(fig, x, ROW_A[0], PC_W, ROW_A[1])
        panel_1d(ax, pcs[(pcs.dimension == 1) & (pcs.regime == regime)], K1, rng)
        annotate(ax, meta.loc[(1, regime)])

    # b: 2D row
    side = 27.0
    ax = gs.axes_mm(fig, SCHEMATIC_X + (SCHEMATIC_W - side) / 2,
                    ROW_B[0] + ROW_B[1] - side, side, side)
    schematic_2d(ax, K2)
    ax.text(0.5, -0.04, f"2D stepping stones\n${K2}\\times{K2}$ demes, $n = {n2}$ per deme",
            transform=ax.transAxes, ha="center", va="top", fontsize=gs.LABEL_PT)
    for x, regime in zip(PC_X, REGIMES):
        ax = gs.axes_mm(fig, x, ROW_B[0], PC_W, ROW_B[1])
        panel_2d(ax, pcs[(pcs.dimension == 2) & (pcs.regime == regime)], K2)
        annotate(ax, meta.loc[(2, regime)])

    gs.label_panel_mm(fig, "a", 1.0, H - 0.5)
    gs.label_panel_mm(fig, "b", 1.0, ROW_B[0] + ROW_B[1] + 5.0)
    out = gs.save(fig, "DemoSS", "main")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
