#%%
"""
The spatial detection threshold: PCA recovers the leading spatial axis iff M < M_c.

A  1D chain, recovery vs M / M_c, coloured by K
B  2D grid, recovery vs M / M_c, coloured by K
C  1D chain at fixed K, coloured by L        (M_c ~ sqrt(L))
D  2D grid at fixed K, coloured by n         (M_c ~ sqrt(N))

The claim under test is not that recovery falls off -- it obviously does as migration
homogenizes the demes -- but that it falls off at M_c and nowhere else. Rescaling the x
axis by each cell's own M_c is what makes that testable: curves computed at different K,
L and N have thresholds spanning more than a decade in raw M, and if the theory is right
they collapse onto a single transition at M / M_c = 1, marked by the dashed line. Curves
that collapse but cross at 1.5, or that fail to collapse at all, are both visible
failures.

The y axis is the mean principal cosine between the sample PCs and the theoretical
structure axes. In 2D that is a *subspace* comparison over the degenerate pair
mu_10 = mu_01, not a per-PC correlation: no canonical basis exists inside a degenerate
eigenspace, so an individual axis is not a well-posed target while the plane they span
is. Panels A and B are therefore on the same scale despite comparing 1 axis and 2.

Reads results/analysis/stepping_stone_detection, writes
results/figures/SteppingStoneDetection.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from crossing_panel import save_figure
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "stepping_stone_detection")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

DEFAULT_N_PER_DEME = 10


def _colour_levels(values, cmap):
    """A colour per distinct level, on a log normalization over the swept parameter."""
    levels = np.array(sorted(set(values)))
    norm = colors.LogNorm(vmin=levels.min(), vmax=levels.max()) if levels.min() > 0 \
        else colors.Normalize(vmin=levels.min(), vmax=levels.max())
    return levels, norm, {v: cmap(norm(v)) for v in levels}


def panel(ax, df, key, cmap, label, title, n_filter=None, L_filter=None):
    """
    Recovery against M / M_c, one curve per level of `key`.

    Cells other than the swept one are pinned so a curve varies in `key` alone; without
    that the same (K, m) simulation would appear once per subsampled n and the curves
    would be a scatter over several thresholds at once.
    """
    if n_filter is not None:
        df = df[df.n_per_deme == n_filter]
    if L_filter is not None:
        df = df[df.L == L_filter]
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return None, None

    levels, norm, colour = _colour_levels(df[key], cmap)
    for level in levels:
        cell = df[df[key] == level].sort_values("M_over_Mc")
        ax.plot(cell.M_over_Mc, cell.median_cos, "-", lw=1.3,
                color=colour[level], zorder=2)
        ax.fill_between(cell.M_over_Mc,
                        cell.median_cos - cell.se_cos,
                        cell.median_cos + cell.se_cos,
                        color=colour[level], alpha=0.18, lw=0, zorder=1)

    ax.axvline(1.0, ls="--", lw=1.0, color="0.35", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$M / M_c$")
    ax.set_ylabel(r"mean principal cosine")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    return levels, norm


def make_figure(df1, df2, outpath, n_pin=DEFAULT_N_PER_DEME):
    """The four panels, with a colourbar per panel since each encodes a different sweep."""
    setup_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))

    specs = []

    L_pin_1 = int(df1.L.mode().iloc[0]) if not df1.empty else None
    lv, nm = panel(axes[0, 0], df1, "K", cm.viridis, "K",
                   "A   1D chain, varying K",
                   n_filter=n_pin, L_filter=L_pin_1)
    specs.append((axes[0, 0], lv, nm, cm.viridis, "K"))

    L_pin_2 = int(df2.L.mode().iloc[0]) if not df2.empty else None
    lv, nm = panel(axes[0, 1], df2, "K", cm.viridis, "K",
                   "B   2D grid, varying K (leading pair)",
                   n_filter=n_pin, L_filter=L_pin_2)
    specs.append((axes[0, 1], lv, nm, cm.viridis, "K"))

    # Fixed K for the L and N panels: the cell with the most levels available.
    if not df1.empty:
        K1 = df1.groupby("K").L.nunique().idxmax()
        d = df1[(df1.K == K1) & (df1.n_per_deme == n_pin)]
        lv, nm = panel(axes[1, 0], d, "L", cm.plasma, "L",
                       f"C   1D chain, K = {K1}, varying L")
        specs.append((axes[1, 0], lv, nm, cm.plasma, "L"))
    else:
        specs.append((axes[1, 0], None, None, cm.plasma, "L"))

    if not df2.empty:
        K2 = df2.groupby("K").n_per_deme.nunique().idxmax()
        d = df2[(df2.K == K2) & (df2.L == L_pin_2)]
        lv, nm = panel(axes[1, 1], d, "n_per_deme", cm.cividis, "n",
                       f"D   2D grid, K = {K2}, varying n")
        specs.append((axes[1, 1], lv, nm, cm.cividis, "n per deme"))
    else:
        specs.append((axes[1, 1], None, None, cm.cividis, "n per deme"))

    for ax, levels, norm, cmap, label in specs:
        if levels is None:
            continue
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label(label)
        cb.set_ticks(levels)
        cb.set_ticklabels([f"{int(v)}" for v in levels])

    fig.tight_layout()
    # outpath is a directory + stem rather than a directory, so it is split back apart
    # to reach the shared writer (PNG to results/figures, PDF to results/pdf_figs).
    out = save_figure(fig, os.path.dirname(outpath), os.path.basename(outpath))
    print(f"wrote {out}")
    return fig


def _load(name):
    path = os.path.join(ANALYSIS_DIR, name)
    if not os.path.exists(path):
        print(f"missing {path}; run src/analysis/stepping_stone_detection.py first")
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == "__main__":
    df1 = _load("detection_1D.csv")
    df2 = _load("detection_2D.csv")
    if df1.empty and df2.empty:
        raise SystemExit("no analysis output to plot")
    make_figure(df1, df2, os.path.join(FIGURE_DIR, "SteppingStoneDetection"))

# %%
