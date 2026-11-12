#%%
"""
Figure 4 (single column): sample PCs in the stepping-stone models at three regimes.

a  1D (K = 25 demes): title and the chain coloured by deme, centred above PC1 (with vertical jitter that
   carries no information) at no, weak and strong structure (left to right).
b  2D (9 x 9 demes): title and the grid coloured by position, centred above PC1 vs PC2 (Procrustes-aligned
   to the degenerate closed-form pair in the analysis script) at the same three regimes.

Regimes are chosen in the analysis script by the measured <u_hat, u>^2; each PC panel is
annotated with its M and that value.

Reads results/analysis/SteppingStoneEigenvectorShapes
Writes results/figures/main/DemoSS.pdf (single column)
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

# Layout (mm), per block top to bottom: title, schematic, three PC panels
W = gs.WIDTH["single"] / gs.MM
PC_X0, PC_W, PC_GAP = 11.3, 20.4, 7.3     # three PC columns
STRIP_H, SQUARE = 7.0, 22.0
ANNOT_H, XAXIS_H = 6.0, 6.5               # two-line annotation above, ticks + label below
CHAIN_H, CHAIN_W, GRID = 5.0, 48.0, 15.0  # schematic sizes
TITLE_H = 3.6                             # title row, its baseline level with the letter
H = (TITLE_H + CHAIN_H + ANNOT_H + STRIP_H + XAXIS_H + 2.5
     + TITLE_H + GRID + ANNOT_H + SQUARE + XAXIS_H)


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


def annotate(ax, row, height_mm):
    ax.text(0.5, 1 + 0.6 / height_mm,
            rf"$M = {fmt_M(row.M)}$" "\n" rf"$\langle\hat u,u\rangle^2 = {row.median_cos ** 2:.2f}$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=gs.TICK_PT,
            linespacing=1.1)


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
    # points per deme as drawn (one per individual)
    n1, n2 = (int(((pcs.dimension == k) & (pcs.regime == REGIMES[0]) & (pcs.deme == 0)).sum())
              for k in (1, 2))

    fig = gs.blank_figure(W, H)
    pc_x = [PC_X0 + i * (PC_W + PC_GAP) for i in range(3)]
    cx = (pc_x[0] + pc_x[-1] + PC_W) / 2      # centre of the three-panel row

    def title(letter, top, text):
        # letter at the block's top-left; title centred, on the letter's baseline
        gs.label_panel_mm(fig, letter, 0.3, top)
        fig.text(cx / W, (top - 2.4) / H, text, ha="center", va="baseline",
                 fontsize=gs.LABEL_PT)

    # a: title, 1D chain, then the three PC1 strips
    top = H
    title("a", top, rf"1D stepping stones, $K = {K1}$ demes, $n = {n1}$ individuals per deme")
    y = top - TITLE_H - CHAIN_H
    schematic_1d(gs.axes_mm(fig, cx - CHAIN_W / 2, y, CHAIN_W, CHAIN_H), K1, n1)
    y -= ANNOT_H + STRIP_H
    for x, regime in zip(pc_x, REGIMES):
        ax = gs.axes_mm(fig, x, y, PC_W, STRIP_H)
        panel_1d(ax, pcs[(pcs.dimension == 1) & (pcs.regime == regime)], K1, rng)
        annotate(ax, meta.loc[(1, regime)], STRIP_H)

    # b: title, 2D grid, then the three PC1-vs-PC2 squares
    top = y - XAXIS_H - 2.5
    title("b", top, rf"2D stepping stones, ${K2}\times{K2}$ demes, $n = {n2}$ individuals per deme")
    y = top - TITLE_H - GRID
    schematic_2d(gs.axes_mm(fig, cx - GRID / 2, y, GRID, GRID), K2)
    y -= ANNOT_H + SQUARE
    for i, (x, regime) in enumerate(zip(pc_x, REGIMES)):
        ax = gs.axes_mm(fig, x, y, PC_W, SQUARE)
        panel_2d(ax, pcs[(pcs.dimension == 2) & (pcs.regime == regime)], K2)
        annotate(ax, meta.loc[(2, regime)], SQUARE)
        if i:
            ax.set_ylabel("")

    out = gs.save(fig, "DemoSS", "main")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
