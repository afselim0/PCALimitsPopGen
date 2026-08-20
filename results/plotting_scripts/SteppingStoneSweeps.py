#%%
"""
Cosine similarity against migration rate, separating the K and N dependences.

One figure per dimension, two panels each:

    A  varying K at (near-)constant N
    B  varying N at constant K

Both panels put raw M on the x axis, so the absolute migration scale is visible and each
curve is read against its own threshold rather than a shared rescaled one. Each curve
carries its own M_c as a short tick along the top: the prediction is that a curve's
fall-off tracks its own marker, and a curve that drops well away from its tick is the
visible failure mode.

What panel A shows differs sharply between the two dimensions, and that contrast is the
point of holding N fixed:

    1D:  M_c = K sqrt(N L) / pi^2   -- still grows with K at fixed N, so the curves fan
         out, here over a factor of ~3.3 from K = 15 to K = 50.
    2D:  M_c = sqrt(n K^2 L) / pi^2 = sqrt(N L) / pi^2  -- the K dependence cancels
         entirely against n = N / K^2, so at constant N every curve is predicted to lie
         on top of every other.

The 2D collapse is a strong prediction but an awkward thing to draw: six coincident
curves look the same whether the theory holds or fails. Panel A therefore gains a
residual inset in 2D, showing each curve's half-recovery point against the shared
prediction on a scale fine enough to resolve departures from it.

The y axis is the mean principal cosine between the sample PCs and the theoretical
structure axes. In 2D that is a subspace comparison over the degenerate leading pair
mu_10 = mu_01 rather than a per-PC correlation -- no canonical basis exists inside a
degenerate eigenspace -- so the 1D and 2D figures are on the same scale despite comparing
one axis and two.

Panel A's N is only as constant as the geometry allows: N = n * K_demes must be a whole
multiple, which in 2D no N satisfies for several K^2 at once. The analysis records the
realized N per curve and whether it is exact, and the legend reports it, so the panel
never implies a constancy it does not have.

Reads results/analysis/stepping_stone_sweeps, writes
results/figures/SteppingStoneSweeps1D.pdf and SteppingStoneSweeps2D.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors, lines

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "stepping_stone_sweeps")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_K = cm.viridis
CMAP_N = cm.plasma


def _colours(levels, cmap, lo=0.12, hi=0.88):
    """A colour per level, spread over the interior of the map so the ends stay legible."""
    levels = np.array(sorted(set(levels)))
    if len(levels) == 1:
        return levels, {levels[0]: cmap(0.5)}
    t = np.linspace(lo, hi, len(levels))
    return levels, {v: cmap(t[i]) for i, v in enumerate(levels)}


def half_recovery_M(cell, level=0.5):
    """
    The M at which a curve crosses `level`, by linear interpolation in log M.

    A single location per curve, so curves that overlap on the raw axis can still be
    compared against each other and against M_c. Returns nan when the curve does not
    cross -- a sweep that never falls that far has no crossing to report, and inventing
    one by extrapolating past the simulated range would be worse than an absent point.
    """
    c = cell.sort_values("M")
    y, x = c.median_cos.to_numpy(), np.log(c.M.to_numpy())
    above = y >= level
    if above.all() or not above.any():
        return np.nan
    i = int(np.argmax(~above))          # first index below the level
    if i == 0:
        return np.nan
    y0, y1, x0, x1 = y[i - 1], y[i], x[i - 1], x[i]
    if y0 == y1:
        return float(np.exp(x0))
    return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))


def collapse_inset(ax, df, key, colour, levels):
    """
    Measured half-recovery point against predicted M_c, for curves that overlap.

    Drawn only where the raw panel cannot separate the curves -- the 2D constant-N case,
    where M_c = sqrt(N L) / pi^2 has no K dependence left. The identity line is the
    prediction; points on it mean the collapse is real, a trend across K means it is not.
    """
    xs, ys, cs = [], [], []
    for level in levels:
        cell = df[df[key] == level]
        if cell.empty:
            continue
        M_half = half_recovery_M(cell)
        M_c = float(cell.M_c.median())
        if np.isfinite(M_half) and np.isfinite(M_c):
            xs.append(M_c)
            ys.append(M_half)
            cs.append(colour[level])
    if len(xs) < 2:
        ax.set_visible(False)
        return

    lim = (min(min(xs), min(ys)) * 0.75, max(max(xs), max(ys)) * 1.33)
    ax.plot(lim, lim, "-", lw=0.9, color="0.5", zorder=1)
    ax.scatter(xs, ys, s=22, c=cs, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"predicted $M_c$", fontsize=7, labelpad=2)
    ax.set_ylabel(r"measured $M_{1/2}$", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, length=2.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def panel(ax, df, key, cmap, legend_title, label_fn, title):
    """
    One sweep: median cosine vs raw M, one curve per level of `key`.

    Each curve's own M_c is drawn as a tick at the top of the axes in the matching
    colour, since with raw M on the x axis the threshold is different for every curve.
    """
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return None, None

    levels, colour = _colours(df[key], cmap)
    for level in levels:
        cell = df[df[key] == level].sort_values("M")
        if cell.empty:
            continue
        c = colour[level]
        ax.plot(cell.M, cell.median_cos, "-", lw=1.4, color=c,
                label=label_fn(cell), zorder=3)
        ax.fill_between(cell.M, cell.median_cos - cell.se_cos,
                        cell.median_cos + cell.se_cos,
                        color=c, alpha=0.18, lw=0, zorder=2)

        M_c = float(cell.M_c.median())
        if np.isfinite(M_c):
            ax.plot([M_c, M_c], [1.02, 1.07], "-", lw=1.6, color=c,
                    clip_on=False, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel(r"$M = 4N_e\bar{m}$")
    ax.set_ylabel("mean principal cosine")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, pad=14)
    leg = ax.legend(title=legend_title, frameon=False, fontsize=8, title_fontsize=8,
                    loc="lower left")
    leg._legend_box.align = "left"
    return levels, colour


def make_figure(df, dimension, outpath):
    """The two panels for one dimension."""
    setup_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    geometry = "chain" if dimension == 1 else "grid"

    a = df[df.sweep == "vary_K_fixed_N"]
    if not a.empty:
        exact = (a.constancy == "exact").all()
        Ns = sorted(a.N.unique())
        note = f"N = {Ns[0]}" if len(Ns) == 1 else f"N ≈ {int(np.median(Ns))}"
        levels, colour = panel(
            axes[0], a, "K", CMAP_K, "K",
            lambda c: f"{int(c.K.iloc[0])}  (n={int(c.n_per_deme.iloc[0])},"
                      f" N={int(c.N.iloc[0])})",
            f"A   {dimension}D {geometry}: varying K at {'constant' if exact else 'near-constant'} {note}")

        # In 2D the K dependence of M_c cancels at fixed N, so the curves are predicted
        # to coincide and the raw axis cannot show whether they do. Resolve it.
        if dimension == 2 and levels is not None:
            inset = axes[0].inset_axes([0.62, 0.52, 0.35, 0.42])
            collapse_inset(inset, a, "K", colour, levels)
    else:
        panel(axes[0], a, "K", CMAP_K, "K", lambda c: "", "A")

    b = df[df.sweep == "vary_N_fixed_K"]
    if not b.empty:
        K_fixed = int(b.K.iloc[0])
        panel(axes[1], b, "N", CMAP_N, "N",
              lambda c: f"{int(c.N.iloc[0])}  (n={int(c.n_per_deme.iloc[0])})",
              f"B   {dimension}D {geometry}: varying N at K = {K_fixed}")
    else:
        panel(axes[1], b, "N", CMAP_N, "N", lambda c: "", "B")

    # One shared note for the M_c ticks rather than an entry per curve in both legends.
    handle = lines.Line2D([], [], color="0.35", lw=1.6)
    fig.legend([handle], [r"ticks above each panel: that curve's own $M_c$"],
               loc="lower center", frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{outpath}.{ext}")
    print(f"wrote {outpath}.pdf / .png")
    return fig


def _load(dimension):
    path = os.path.join(ANALYSIS_DIR, f"sweeps_{dimension}D.csv")
    if not os.path.exists(path):
        print(f"missing {path}; run src/analysis/stepping_stone_sweeps.py "
              f"--dimension {dimension} first")
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == "__main__":
    made = False
    for dimension in (1, 2):
        df = _load(dimension)
        if df.empty:
            continue
        make_figure(df, dimension,
                    os.path.join(FIGURE_DIR, f"SteppingStoneSweeps{dimension}D"))
        made = True
    if not made:
        raise SystemExit("no analysis output to plot")

# %%
