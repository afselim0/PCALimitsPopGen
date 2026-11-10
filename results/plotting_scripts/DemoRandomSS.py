#%%
"""
Sampling density decides whether PCA sees spatial structure, 1D and 2D (supplementary).

M is fixed per dimension and total N is fixed, so only the fraction of demes sampled (phi)
changes across the three columns (phi = 0.2, 0.4, 0.6).

a  1D: the sampled demes on the chain (unsampled faded), and PC1 of that sample, coloured
   by chain position and annotated with <u_hat_phi, u_1>^2 against the full-lattice axis.
b  2D: the sampled demes on the K x K grid, and PC1 vs PC2, coloured by grid position and
   annotated with the mean principal cos^2 against the full-lattice degenerate pair.

Reads results/analysis/SteppingStoneRandomDemo
Writes results/figures/supplement/DemoRandomSS.pdf
"""
import os

import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

import genetics_style as gs

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SteppingStoneRandomDemo")

COLOR_EDGE = "0.25"
ALPHA_SAMPLED, ALPHA_UNSAMPLED = 1.0, 0.15
JITTER_1D = 0.16

# layout (mm)
LEFT, CW, CGAP = 12, 38, 9
W = LEFT + 3 * CW + 2 * CGAP + 2
PC2D_B, SQ = 9, 38              # 2D PC panel bottom and size; 2D scheme is also SQ
SCH2D_B = PC2D_B + SQ + 5
PC1D_B, PC1D_H = SCH2D_B + SQ + 17, 10
CH1D_B, CH1D_H = PC1D_B + PC1D_H + 5, 4
H = CH1D_B + CH1D_H + 9


def _text_mm(fig, x, y, s, **kw):
    fig.text(x / W, y / H, s, **kw)


def _blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scheme_1d(ax, demes, K):
    ax.plot([0, K - 1], [0, 0], "-", color=COLOR_EDGE, linewidth=0.5, zorder=1)
    sampled = demes.set_index("deme").sampled
    for x in range(K):
        ax.add_patch(Circle((x, 0), 0.42, facecolor=gs.chain_color(x, K),
                            edgecolor=COLOR_EDGE, linewidth=0.4, zorder=2,
                            alpha=ALPHA_SAMPLED if sampled[x] else ALPHA_UNSAMPLED))
    ax.set_xlim(-0.6, K - 0.4)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    _blank(ax)


def scheme_2d(ax, demes, K):
    for i in range(K):
        ax.plot([0, K - 1], [i, i], "-", color=COLOR_EDGE, linewidth=0.5, zorder=1)
        ax.plot([i, i], [0, K - 1], "-", color=COLOR_EDGE, linewidth=0.5, zorder=1)
    sampled = demes.set_index(["row", "col"]).sampled
    for r in range(K):
        for c in range(K):
            ax.add_patch(Circle((c, r), 0.38, facecolor=gs.grid_color(r, c, K),
                                edgecolor=COLOR_EDGE, linewidth=0.4, zorder=2,
                                alpha=ALPHA_SAMPLED if sampled[(r, c)] else ALPHA_UNSAMPLED))
    ax.set_xlim(-0.45, K - 0.55)
    ax.set_ylim(-0.45, K - 0.55)
    ax.set_aspect("equal")
    _blank(ax)


def pc_panel_1d(ax, d, K, rng):
    y = rng.uniform(-JITTER_1D, JITTER_1D, len(d))
    ax.scatter(d.pc1, y, c=[gs.chain_color(int(j), K) for j in d.deme], s=2,
               edgecolors="none", alpha=0.85)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("PC1")


def pc_panel_2d(ax, d, K):
    c = [gs.grid_color(int(r), int(cc), K) for r, cc in zip(d.row, d.col)]
    ax.scatter(d.pc1, d.pc2, c=c, s=2, edgecolors="none", alpha=0.85)
    lim = 1.08 * max(np.abs(d.pc1).max(), np.abs(d.pc2).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    ax.set_xlabel("PC1")


def make_figure(analysis_dir=ANALYSIS_DIR):
    gs.setup()
    demes = pd.read_csv(os.path.join(analysis_dir, "sampled_demes.csv"))
    pcs = pd.read_csv(os.path.join(analysis_dir, "pcs.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv"))
    phis = sorted(pcs.phi.unique())
    K1 = int(demes[demes.dimension == 1].deme.max()) + 1
    K2 = int(np.sqrt(int(demes[demes.dimension == 2].deme.max()) + 1))

    fig = gs.blank_figure(W, H)
    for j, phi in enumerate(phis):
        x0 = LEFT + j * (CW + CGAP)
        xc = x0 + CW / 2
        d1 = demes[(demes.dimension == 1) & (demes.phi == phi)]
        d2 = demes[(demes.dimension == 2) & (demes.phi == phi)]
        p1 = pcs[(pcs.dimension == 1) & (pcs.phi == phi)]
        p2 = pcs[(pcs.dimension == 2) & (pcs.phi == phi)]
        n1, n2 = (int(meta.loc[(meta.dimension == k) & (meta.phi == phi), "n_per_deme"].iloc[0])
                  for k in (1, 2))

        scheme_1d(gs.axes_mm(fig, x0, CH1D_B, CW, CH1D_H), d1, K1)
        pc_panel_1d(gs.axes_mm(fig, x0, PC1D_B, CW, PC1D_H), p1, K1, np.random.default_rng(j))
        scheme_2d(gs.axes_mm(fig, x0 + (CW - SQ) / 2, SCH2D_B, SQ, SQ), d2, K2)
        ax2 = gs.axes_mm(fig, x0, PC2D_B, CW, SQ)
        pc_panel_2d(ax2, p2, K2)
        if j == 0:
            ax2.set_ylabel("PC2")

        for y, d, K, n in ((CH1D_B + CH1D_H + 0.8, d1, K1, n1),
                           (SCH2D_B + SQ + 1.0, d2, K2, n2)):
            _text_mm(fig, xc, y, rf"$\phi = {phi:g}$ ({int(d.sampled.sum())}/{len(d)} demes)"
                     "\n" rf"$n = {n}$ per deme", ha="center", va="bottom",
                     linespacing=1.15)
        cos_lab = r"$\langle\hat u_{%g}, u_1\rangle^2 = %.2f$"
        _text_mm(fig, xc, PC1D_B + PC1D_H + 0.8, cos_lab % (phi, p1["cos"].iloc[0]),
                 ha="center", va="bottom")
        _text_mm(fig, xc, PC2D_B + SQ + 0.8, cos_lab % (phi, p2["cos"].iloc[0]),
                 ha="center", va="bottom")

    gs.label_panel_mm(fig, "a", 1, H - 1)
    gs.label_panel_mm(fig, "b", 1, SCH2D_B + SQ + 8)
    out = gs.save(fig, "DemoRandomSS", "supplement")
    print(f"wrote {out}")
    for k in (1, 2):
        m = meta[meta.dimension == k].iloc[0]
        print(f"{k}D: M = {m.M:.0f}, N_total = {int(m.N_total)}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
