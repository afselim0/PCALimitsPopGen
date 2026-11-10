#%%
"""
Supplementary figure: the spatial PC pair after a recent split (schematic: main Fig. 6a).

a-c  PC2 vs PC3 (the grid's spatial pair; A in grey) at weak, intermediate and strong
   migration, annotated with M and <u_hat_t, u_inf>^2 -- the observed axis at t_split
   against the equilibrium stepping-stone axis.

Reads results/analysis/RecentSpatialDemography
Writes results/figures/supplement/DemoRecentSS.pdf
"""
import os

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

import genetics_style as gs
from genetics_style import grid_color  # re-exported: DemoStrideSS imports it from here

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_ROOT, "results", "analysis", "RecentSpatialDemography")

LABELS = ("weak", "intermediate", "strong")
MARKER_S = 1.2

# Layout (mm)
PC_W, PC_Y = 42.0, 8.0
PC_X = [11.0, 68.0, 125.0]
W, H = PC_X[-1] + PC_W + 2.0, PC_Y + PC_W + 9.0



def panel_pc(ax, d, K):
    is_A = d.deme_row.isna().values
    ax.scatter(d.pc2[is_A], d.pc3[is_A], s=MARKER_S * 0.7, color=gs.BACKGROUND, alpha=0.6,
               edgecolors="none", zorder=1, rasterized=True)
    dB = d[~is_A]
    c = [grid_color(int(r), int(cc), K) for r, cc in zip(dB.deme_row, dB.deme_col)]
    ax.scatter(dB.pc2, dB.pc3, s=MARKER_S, c=c, edgecolors="none", zorder=2,
               rasterized=True)
    lim = 1.05 * np.abs(np.r_[d.pc2, d.pc3]).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(3, symmetric=True, steps=[1, 2, 5, 10]))
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3", labelpad=1)


def make_figure(analysis_dir=ANALYSIS_DIR):
    gs.setup()
    pc = pd.read_csv(os.path.join(analysis_dir, "pc.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("label")
    K = int(pc.deme_row.max() + 1)
    t_split = float(meta["t_split"].iloc[0]) if "t_split" in meta else 500.0

    fig = gs.blank_figure(W, H)
    for x, label, letter in zip(PC_X, LABELS, "abc"):
        ax = gs.axes_mm(fig, x, PC_Y, PC_W, PC_W)
        panel_pc(ax, pc[pc.label == label], K)
        M, cos = float(meta.loc[label, "M"]), float(meta.loc[label, "cos"])
        ax.text(0.5, 1.02, rf"$M = {M:.2g}$" "\n"
                rf"$\langle\hat u_{{t={t_split:g}}}, u_\infty\rangle^2 = {cos:.2f}$",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=gs.LABEL_PT)
        gs.label_panel_mm(fig, letter, x - 10.0, H - 0.5)
    out = gs.save(fig, "DemoRecentSS", "supplement")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
