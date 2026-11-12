#%%
"""
Figure 1 (single column): RMT phenomena on a two-population split.

a     The spectrum at three Fst values: below, just above and far above the critical
      detectability point lambda_c. Bulk eigenvalues (histogram) against the Marchenko-Pastur
      density with L_eff; the true spike lambda_1 = 1 + nu and lambda_c as lines; the largest sample
      eigenvalue over replicate subsamples as a histogram (on a broken axis at the largest Fst). The
      sketch shows the true axis u_1 and the sample PC u_hat_1 at the angle of the measured
      mean cosine similarity (drawn as a right angle below threshold).
b     BBP phase transition: lambda_1 / lambda_2 against Fst, with the critical Fst from raw
      L and from L_eff (dotted).
c     Eigenvector inconsistency: <u1_hat, u1>^2 against Fst -- the fit, band and exact theory
      of the figure's own sweep -- with PC1 at the same three Fst values underneath.

Reads results/analysis/RMTPhenomena and results/sweep_summaries/rmt_phenomena.
Writes results/figures/main/RMTPhenomena.pdf
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Patch

import genetics_style as gs

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_ROOT, "results", "analysis", "RMTPhenomena")
SUMMARY_DIR = os.path.join(_ROOT, "results", "sweep_summaries", "rmt_phenomena")

COLOR_RAW, COLOR_EFF = "#E69F00", "#0072B2"   # theory with raw L / with L_eff
COLOR_GAP = "#AA4499"                         # measured eigenvalue ratio
COLOR_FIT = "#D55E00"                         # measured cosine similarity
COLOR_SAMPLE = "#009E73"                      # largest sample eigenvalue / eigenvector (a)
COLOR_CRIT = "#D62728"                        # critical detectability point lambda_c
COLOR_THEORY = "0.25"
INSET_JITTER = 0.18
INSETS = ("small", "threshold", "large")      # the three Fst values of panel a and the PC1 strips

# Layout (mm): three spectra over two panels, across 87.9 mm
W, H = gs.WIDTH["single"] / gs.MM, 83.5
PANEL_H = 21.5
PANEL_X = [(9.5, 30.0), (53.0, 32.0)]
TOP_Y, BOT_Y = 57.5, 20.5
KEY_TOP = 12.6
TOP_X, TOP_W = [3.8, 32.2, 60.6], 25.5     # the three spectra: left edges and width
SKETCH_W = 10.5                            # right part of each spectrum panel holds the sketch
BREAK_W, SEG2_W, SEG2_PAD = 1.3, 5.6, 0.22  # broken axis for a spike far beyond the bulk
SPECTRUM_PAD = 0.07                        # eigenvalue margin either side of what is shown
LABEL_SEP_MM = 2.6                         # smallest spacing at which two mark labels clear


def sci_latex(x, digits=1):
    e = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    return rf"{x / 10.0 ** e:.{digits}f} \times 10^{{{e}}}"


def mp_density(x, gamma):
    """Marchenko-Pastur density for a floor at 1."""
    lo, hi = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = (x > lo) & (x < hi)
    out[m] = np.sqrt((hi - x[m]) * (x[m] - lo)) / (2 * np.pi * gamma * x[m])
    return out


def eigenvector_sketch(ax, cos2, width, height):
    """u_1 and u_hat_1 as two arrows from one origin, at the angle of the measured cos^2."""
    ortho = cos2 < 0.05                      # labelled as ~0, so drawn as a right angle
    th = np.pi / 2 if ortho else np.arccos(np.sqrt(cos2))
    near = th < np.radians(15)               # arrows almost coincide: labels on either side
    span = 1.4 * width / height              # equal aspect: x span that fills the box
    x_lo = -span / 2 + 0.05 if near else -0.2
    ax.set_xlim(x_lo, x_lo + span)
    ax.set_ylim(-0.1, 1.3)
    ax.axis("off")
    for (dx, dy), color in (((0, 1), "black"), ((np.sin(th), np.cos(th)), COLOR_SAMPLE)):
        ax.add_patch(FancyArrowPatch((0, 0), (dx, dy), arrowstyle="-|>", mutation_scale=5,
                                     lw=1.0, color=color, shrinkA=0, shrinkB=0))
    x_u, y_u, ha_u = (-0.12, 0.9, "right") if near else (-0.02, 1.12, "center")
    ax.text(x_u, y_u, r"$u_1$", fontsize=gs.TICK_PT, ha=ha_u, va="center")
    x_hat, y_hat, ha = (0.62, 0.2, "center") if ortho else (np.sin(th) + 0.1, np.cos(th), "left")
    ax.text(x_hat + (0.04 if near else 0), y_hat - (0.1 if near else 0), r"$\hat u_1$", color=COLOR_SAMPLE,
            fontsize=gs.TICK_PT, ha=ha, va="center")
    if ortho:
        ax.plot([0, 0.16, 0.16], [0.16, 0.16, 0], color="0.3", lw=0.5)
    elif not near:
        ax.add_patch(Arc((0, 0), 0.6, 0.6, theta1=90 - np.degrees(th), theta2=90, color="0.3",
                         lw=0.5))


def draw_marks(ax, marks, peak, min_sep):
    """Equal-height lines with their labels on one row, pushed apart where they would collide."""
    h = 1.16 * peak
    marks = sorted(marks)
    xs = [m[0] for m in marks]
    for k in range(len(xs) - 1):
        short = min_sep - (xs[k + 1] - xs[k])
        if short > 0:
            xs[k] -= short / 2
            xs[k + 1] += short / 2
    for (xv, color, label), xl in zip(marks, xs):
        if color != COLOR_SAMPLE:
            ax.plot([xv] * 2, [0, h], color=color, ls=":" if color == COLOR_CRIT else "-",
                    linewidth=1.0)
        ax.text(xl, h + 0.03 * peak, label, color=color, ha="center", va="bottom",
                fontsize=gs.TICK_PT)


def spectrum_axes(fig, x0, width, xlim, peak, ticks):
    ax = gs.axes_mm(fig, x0, TOP_Y, width, PANEL_H)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.45 * peak)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.spines["left"].set_visible(False)
    return ax


def spectrum_frame(top, fst, N):
    """Shared eigenvalue axis of the top row: (left edge, mm per unit, bulk right edge).

    Sized so the widest bulk and every spike near it fit beside the sketch; a spike more
    than a bulk width beyond the edge is left to the broken axis.
    """
    g = N / top.groupby("label").L_eff.median()
    lo, edge = (1 - np.sqrt(g.max())) ** 2 - SPECTRUM_PAD, (1 + np.sqrt(g.max())) ** 2
    reach = [max(N / 2 * fst[k] / (1 - fst[k]) + 1, d.lam1.max()) for k, d in top.groupby("label")]
    near = max([r for r in reach if r < edge + (edge - lo)] + [edge]) + SPECTRUM_PAD
    return lo, (TOP_W - SKETCH_W) / (near - lo), edge + SPECTRUM_PAD


def spectrum_ticks(lo, hi, scale):
    """Round tick values at least 5 mm apart."""
    step = next(t for t in (0.2, 0.4, 0.5, 1.0, 2.0) if t * scale >= 5.0)
    return np.arange(np.ceil(lo / step), np.floor(hi / step) + 1) * step


def panel_spike(fig, x0, top, bulk, fst, N, frame):
    """Spectrum with the true spike, lambda_c and the sample-eigenvalue histogram; + sketch.

    A spike beyond the panel is drawn on a second axis segment after a break.
    """
    x_lo, scale, bulk_hi = frame
    g = N / float(top.L_eff.median())
    nu = N / 2 * fst / (1 - fst)                 # 2 N phi (1 - phi) Fst / (1 - Fst) at phi = 1/2, as the sweep
    lam_hat = float(top.lam1.median())
    x_hi = x_lo + TOP_W / scale
    broken = max(1 + nu, top.lam1.max()) > x_hi - SKETCH_W / scale
    mp_x = np.linspace((1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2, 1500)   # edge to edge, so it meets the axis
    mp_y = mp_density(mp_x, g)
    peak = mp_y.max()

    if broken:
        seg1_w = (bulk_hi - x_lo) * scale
        ax = spectrum_axes(fig, x0, seg1_w, (x_lo, bulk_hi), peak, spectrum_ticks(x_lo, bulk_hi, scale))
        lo, hi = min(1 + nu, top.lam1.min()) - SEG2_PAD, max(1 + nu, top.lam1.max()) + SEG2_PAD
        ax_s = spectrum_axes(fig, x0 + seg1_w + BREAK_W, SEG2_W, (lo, hi), peak,
                             [round((lo + hi) / 2, 1)])
        for a, at in ((ax, 1), (ax_s, 0)):       # break marks on the facing spine ends
            a.plot([at], [0], transform=a.transAxes, marker=[(-1, -2), (1, 2)], markersize=4.5,
                   linestyle="none", color="k", markeredgewidth=0.6, clip_on=False)
        sketch_w = TOP_W - seg1_w - BREAK_W - SEG2_W
        label_x = (seg1_w + BREAK_W + SEG2_W) / 2 / seg1_w
    else:
        ax = ax_s = spectrum_axes(fig, x0, TOP_W, (x_lo, x_hi), peak,
                                  spectrum_ticks(x_lo, x_hi - SKETCH_W / scale + 0.2, scale))
        sketch_w, label_x = SKETCH_W, 0.5
    ax.set_xlabel(r"eigenvalue $\lambda$", labelpad=1, x=label_x)
    ax.hist(bulk, bins=np.linspace(x_lo, bulk_hi, 70), density=True, color="0.78",
            edgecolor="none")
    ax.plot(mp_x, mp_y, "-", color=COLOR_EFF, linewidth=1.0, solid_capstyle="butt")
    # lambda_1_hat over replicates, drawn at the height of the bulk peak
    edges = np.linspace(top.lam1.min() - 0.01, top.lam1.max() + 0.01, 22)
    dens, _ = np.histogram(top.lam1, bins=edges, density=True)
    ax_s.bar(edges[:-1], dens * peak / dens.max(), width=np.diff(edges), align="edge",
             color=COLOR_SAMPLE, alpha=0.8, linewidth=0)
    crit = [1 + np.sqrt(g), COLOR_CRIT, r"$\lambda_c$"]
    spike = [[1 + nu, "black", r"$\lambda_1$"], [lam_hat, COLOR_SAMPLE, r"$\hat\lambda_1$"]]
    if broken:
        draw_marks(ax, [crit], peak, 0)
        draw_marks(ax_s, spike, peak, LABEL_SEP_MM * (hi - lo) / SEG2_W)
    else:
        draw_marks(ax, [crit] + spike, peak, LABEL_SEP_MM / scale)

    cos2 = float(top.cos2.mean())
    lift = 2.6 if broken else 0              # the narrow sketch carries its label on two lines
    eigenvector_sketch(gs.axes_mm(fig, x0 + TOP_W - sketch_w, TOP_Y + 5.5 + lift, sketch_w, 9.5),
                       cos2, sketch_w, 9.5)
    label = rf"$\langle\hat u_1,u_1\rangle^2${'|' if broken else ''}$\approx{cos2:.1f}$"
    fig.text((x0 + TOP_W - sketch_w / 2) / W, (TOP_Y + 1.6) / H,
             label.replace("0.0", "0").replace("|", "\n"), ha="center", va="bottom",
             fontsize=gs.TICK_PT, linespacing=1.1)
    fig.text((x0 + TOP_W / 2) / W, (TOP_Y + PANEL_H + 0.8) / H,
             rf"$F_{{ST}} = {sci_latex(fst, digits=0)}$", ha="center", va="baseline",
             fontsize=gs.TICK_PT)


def spike_key(fig, L_eff):
    """One-line key for panel a, centred on the page unless that leaves it."""
    handles = [Patch(color="0.78", label="bulk eigenvalues"),
               Line2D([], [], color=COLOR_EFF,
                      label=rf"Marchenko–Pastur ($L_{{\mathrm{{eff}}}}$ = {round(L_eff, -2):,.0f})"),
               Patch(color=COLOR_SAMPLE, alpha=0.8, label=r"$\hat\lambda_1$ (400 replicates)")]
    opts = dict(handles=handles, ncol=3, loc="center", frameon=False, fontsize=gs.LEGEND_PT,
                handlelength=1.3, columnspacing=0.9, handletextpad=0.4, borderpad=0)
    mid = W / 2
    y = (TOP_Y - 8.0) / H
    fig.canvas.draw()
    probe = fig.legend(bbox_to_anchor=(mid / W, y), **opts)
    width = probe.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / gs.MM
    probe.remove()
    fig.legend(bbox_to_anchor=(min(mid, W - width / 2 - 0.6) / W, y), **opts)


def threshold_lines(ax, meta):
    for x, c in [(meta.Fst_c_raw, COLOR_RAW), (meta.Fst_c_eff, COLOR_EFF)]:
        ax.axvline(float(x), color=c, linestyle=":", linewidth=1.0)


def panel_gap(ax, gap, meta):
    ax.plot(gap.Fst, gap.gap, "-", color=COLOR_GAP, linewidth=1.0)
    threshold_lines(ax, meta)
    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{ST}$")
    ax.set_ylabel("median ratio\n" r"$\hat\lambda_1/\hat\lambda_2$", labelpad=1, linespacing=1.3)
    return [Line2D([], [], color=COLOR_GAP, label=r"median $\hat\lambda_1/\hat\lambda_2$"),
            Line2D([], [], color=COLOR_RAW, ls=":", label=r"$F_{ST}^{\,c}$ with $L$"),
            Line2D([], [], color=COLOR_EFF, ls=":",
                   label=r"$F_{ST}^{\,c}$ with $L_{\mathrm{eff}}$")]


def load_sweep_summary(summary_dir):
    """(curve, theory, points) of the figure's own sweep, which has a single curve."""
    read = lambda f: pd.read_csv(os.path.join(summary_dir, f)).sort_values("x")
    return read("curves.csv"), read("theory.csv"), read("points.csv")


def panel_cosine(ax, meta, summary):
    curve, theory, pts = summary
    ax.scatter(pts.x, pts.cos2, s=1.5, alpha=0.3, linewidths=0, color=COLOR_FIT,
               zorder=1, rasterized=True)
    ax.fill_between(curve.x, curve.y_lo, curve.y_hi, color=COLOR_FIT, alpha=0.3,
                    linewidth=0, zorder=2)
    ax.plot(curve.x, curve.y_fit, "-", color=COLOR_FIT, linewidth=1.0, zorder=3)
    ax.plot(theory.x, theory.cos2_exact, "--", color=COLOR_THEORY, linewidth=1.0, zorder=4)
    threshold_lines(ax, meta)
    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{ST}$")
    ax.set_ylabel("squared cosine\n" r"similarity $\langle\hat u_1, u_1\rangle^2$", labelpad=1, linespacing=1.3)
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
    for j, key in enumerate(INSETS):
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


def make_figure(analysis_dir=ANALYSIS_DIR, summary_dir=SUMMARY_DIR, name="RMTPhenomena"):
    gs.setup()
    read = lambda f: pd.read_csv(os.path.join(analysis_dir, f))
    gap, pc1 = read("gap.csv"), read("pc1.csv")
    top, bulk = read("spike_top.csv"), read("spike_bulk.csv")
    spike_meta = read("spike_meta.csv").set_index("label")
    meta = read("meta.csv").iloc[0]

    fig = gs.blank_figure(W, H)
    frame = spectrum_frame(top, spike_meta.Fst, int(meta.N))
    for x, label in zip(TOP_X, INSETS):
        panel_spike(fig, x, top[top.label == label], bulk[bulk.label == label].eigenvalue,
                    float(spike_meta.loc[label, "Fst"]), int(meta.N), frame)
    gs.label_panel_mm(fig, "a", TOP_X[0] - 3.5, H - 0.3)
    # L_eff is stored on the haploid-equivalent scale; the key shows the SNP count itself
    ploidy = float(spike_meta.ploidy.iloc[0]) if "ploidy" in spike_meta else 1.0
    spike_key(fig, float(top.L_eff.median()) / ploidy)

    ax_gap, ax_cos = [gs.axes_mm(fig, x, BOT_Y, w, PANEL_H) for x, w in PANEL_X]
    h_gap = panel_gap(ax_gap, gap, meta)
    panel_cosine(ax_cos, meta, load_sweep_summary(summary_dir))
    centers = [x + w / 2 for x, w in PANEL_X]
    gs.grid_key(fig, [h_gap], centers[:1], KEY_TOP + 0.4, pitch_mm=3.0, handlelength=1.6)
    add_insets(fig, pc1, centers[1] - 17.0, 34.0, KEY_TOP)
    for (x, _), letter in zip(PANEL_X, "bc"):
        gs.label_panel_mm(fig, letter, x - 9.2, BOT_Y + PANEL_H + 3.2)
    out = gs.save(fig, name, "main")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure(*sys.argv[1:])       # optional: analysis dir, summary dir, output name

# %%
