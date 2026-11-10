#%%
"""
Figure 1 (single column): three RMT phenomena on a two-population split, N = 400.

a  Eigenvalue spreading: bulk histogram against the Marchenko-Pastur density with raw L
   and with L_eff; the black line is the (repeated) theoretical eigenvalue.
b  BBP phase transition: lambda_1 / lambda_2 against Fst, with the critical Fst from raw
   L and from L_eff (dotted).
c  Eigenvector inconsistency: <v1_hat, v1>^2 against Fst -- the single_split sweep's fit,
   band and exact theory at N = 400 -- with PC1 at three Fst values underneath.

Reads results/analysis/RMTPhenomena and results/sweep_summaries/single_split.
Writes results/figures/main/RMTPhenomena.pdf
"""
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import genetics_style as gs

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_ROOT, "results", "analysis", "RMTPhenomena")
SUMMARY_DIR = os.path.join(_ROOT, "results", "sweep_summaries", "single_split")
SWEEP_LEVEL = 400

COLOR_RAW, COLOR_EFF = "#E69F00", "#0072B2"   # theory with raw L / with L_eff
COLOR_GAP = "#009E73"                         # measured eigenvalue ratio
COLOR_FIT = "#D55E00"                         # measured cosine similarity
COLOR_THEORY = "0.25"
INSET_JITTER = 0.18

# Layout (mm): three panels across 87.9 mm
W, H = gs.WIDTH["single"] / gs.MM, 46.0
PANEL_Y, PANEL_H = 20.5, 21.5
PANEL_X = [(7.6, 19.5), (35.0, 19.5), (63.0, 23.5)]
KEY_TOP = 12.6


def sci_latex(x, digits=1):
    e = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    return rf"{x / 10.0 ** e:.{digits}f} \times 10^{{{e}}}"


def mp_density(x, gamma, scale=1.0):
    """MP density for a floor at `scale`."""
    lo, hi = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2
    u = np.asarray(x, dtype=float) / scale
    out = np.zeros_like(u)
    m = (u > lo) & (u < hi)
    out[m] = np.sqrt((hi - u[m]) * (u[m] - lo)) / (2 * np.pi * gamma * u[m])
    return out / scale


def panel_a(ax, spectrum, meta):
    bulk = spectrum.eigenvalue.values
    N = int(meta.N)
    floor = float(bulk.mean())   # MP has mean 1 in floor units
    ax.hist(bulk, bins=28, density=True, color="0.78", edgecolor="none")
    for L, color in [(float(spectrum.L_raw.iloc[0]), COLOR_RAW),
                     (float(spectrum.L_eff.iloc[0]), COLOR_EFF)]:
        g = N / L
        x = np.linspace(floor * (1 - np.sqrt(g)) ** 2, floor * (1 + np.sqrt(g)) ** 2, 600)
        ax.plot(x, mp_density(x, g, scale=floor), "--", color=color, linewidth=1.0)
    ax.axvline(floor, color="black", linewidth=1.0)
    ax.set_xlabel(r"eigenvalue $\hat\lambda$")
    ax.set_ylabel("density", labelpad=1)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0.75, 1.0, 1.25])
    L_raw, L_eff = float(spectrum.L_raw.iloc[0]), float(spectrum.L_eff.iloc[0])
    return [Patch(color="0.78", label=f"bulk ({bulk.size})"),
            Line2D([], [], color=COLOR_RAW, ls="--", label=rf"MP, $L$ = {L_raw:,.0f}"),
            Line2D([], [], color=COLOR_EFF, ls="--",
                   label=rf"MP, $L_{{\mathrm{{eff}}}}$ = {L_eff:,.0f}"),
            Line2D([], [], color="black", label=r"theory $\lambda$")]


def threshold_lines(ax, meta):
    for x, c in [(meta.Fst_c_raw, COLOR_RAW), (meta.Fst_c_eff, COLOR_EFF)]:
        ax.axvline(float(x), color=c, linestyle=":", linewidth=1.0)


def panel_b(ax, gap, meta):
    ax.plot(gap.Fst, gap.gap, "-", color=COLOR_GAP, linewidth=1.0)
    threshold_lines(ax, meta)
    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{ST}$")
    ax.set_ylabel(r"$\hat\lambda_1/\hat\lambda_2$", labelpad=1)
    return [Line2D([], [], color=COLOR_GAP, label=r"median $\hat\lambda_1/\hat\lambda_2$"),
            Line2D([], [], color=COLOR_RAW, ls=":", label=r"$F_{ST}^{\,c}$ with $L$"),
            Line2D([], [], color=COLOR_EFF, ls=":",
                   label=r"$F_{ST}^{\,c}$ with $L_{\mathrm{eff}}$")]


def load_sweep_summary(level=SWEEP_LEVEL, sweep="split_vary_N", summary_dir=SUMMARY_DIR):
    """(curve, theory, points) for one level of the single_split sweep, or None."""
    try:
        curve = pd.read_csv(os.path.join(summary_dir, "curves.csv"))
        theory = pd.read_csv(os.path.join(summary_dir, "theory.csv"))
        points = pd.read_csv(os.path.join(summary_dir, "points.csv"))
    except FileNotFoundError:
        return None
    sel = lambda d: d[(d.sweep == sweep) & (d.level == level)].sort_values("x")
    curve, theory, points = sel(curve), sel(theory), sel(points)
    return None if curve.empty or theory.empty else (curve, theory, points)


def panel_c(ax, cos_sweep, meta, summary):
    if summary is not None:
        curve, theory, pts = summary
        ax.scatter(pts.x, pts.cos2, s=1.5, alpha=0.3, linewidths=0, color=COLOR_FIT,
                   zorder=1, rasterized=True)
        ax.fill_between(curve.x, curve.y_lo, curve.y_hi, color=COLOR_FIT, alpha=0.3,
                        linewidth=0, zorder=2)
        ax.plot(curve.x, curve.y_fit, "-", color=COLOR_FIT, linewidth=1.0, zorder=3)
        ax.plot(theory.x, theory.cos2_exact, "--", color=COLOR_THEORY, linewidth=1.0, zorder=4)
    else:
        d = cos_sweep.sort_values("Fst")
        ax.plot(d.Fst, d.median_cos ** 2, "-", color=COLOR_FIT)
        ax.plot(d.Fst, d.analytical_cos ** 2, "--", color=COLOR_THEORY)
    threshold_lines(ax, meta)
    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{ST}$")
    ax.set_ylabel(r"$\langle\hat v_1, v_1\rangle^2$", labelpad=1)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.legend([Line2D([], [], color=COLOR_FIT), Line2D([], [], color=COLOR_THEORY, ls="--")],
              ["fit", "theory"], loc="lower right", handlelength=1.6,
              borderaxespad=0.1)


def add_insets(fig, pc1, x0, width, top):
    """PC1 scatters at three Fst values, in a row under panel c."""
    fig.text(x0 / W + width / W / 2, top / H, r"PC1 at three values of $F_{ST}$",
             ha="center", va="top", fontsize=gs.LEGEND_PT)
    gap, n = 1.6, 3
    w = (width - gap * (n - 1)) / n
    h = 4.6
    for j, key in enumerate(["small", "threshold", "large"]):
        g = pc1[pc1.inset == key]
        ax = gs.axes_mm(fig, x0 + j * (w + gap), top - 6.0 - h, w, h)
        jit = np.random.default_rng(j).uniform(-INSET_JITTER, INSET_JITTER, len(g))
        for p, color in [(0, gs.POP1), (1, gs.POP2)]:
            m = g.population.values == p
            ax.scatter(g.pc1.values[m], jit[m], s=0.6, color=color, alpha=0.75,
                       edgecolors="none", rasterized=True)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
        ax.set_title(rf"${sci_latex(float(g.Fst.iloc[0]), digits=0)}$",
                     fontsize=gs.TICK_PT, pad=1.5)


def make_figure(analysis_dir=ANALYSIS_DIR):
    gs.setup()
    read = lambda f: pd.read_csv(os.path.join(analysis_dir, f))
    gap, cos_sweep, spectrum, pc1 = (read(f) for f in
                                     ("gap.csv", "cos_sweep.csv", "spectrum.csv", "pc1.csv"))
    meta = read("meta.csv").iloc[0]

    fig = gs.blank_figure(W, H)
    ax_a, ax_b, ax_c = [gs.axes_mm(fig, x, PANEL_Y, w, PANEL_H) for x, w in PANEL_X]
    h_a = panel_a(ax_a, spectrum, meta)
    h_b = panel_b(ax_b, gap, meta)
    panel_c(ax_c, cos_sweep, meta, load_sweep_summary())

    # keys centred under their panels on one shared row grid; insets centred under c
    centers = [x + w / 2 for x, w in PANEL_X]
    gs.grid_key(fig, [h_a, h_b], centers[:2], KEY_TOP + 0.4, pitch_mm=3.0, handlelength=1.6)
    inset_w = 2 * min(centers[2] - PANEL_X[1][0] - PANEL_X[1][1] - 2.0, W - centers[2] - 0.4)
    add_insets(fig, pc1, centers[2] - inset_w / 2, inset_w, KEY_TOP)

    for (x, _), letter in zip(PANEL_X, "abc"):
        gs.label_panel_mm(fig, letter, x - 7.4, H - 0.3)
    out = gs.save(fig, "RMTPhenomena", "main")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
