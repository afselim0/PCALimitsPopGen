"""
Main Fig. 6: spatial-axis recovery vs migration after a recent split (panmictic A + 9 x 9 grid B).

a  schematic: an ancestral population splits t_split generations ago into a panmictic A and
   a K x K stepping-stone grid B (demes coloured by grid position, as in DemoRecentSS).
b  cosine similarity of the spatial pair vs 1/M, n per deme of B varied (phi = 0.5).
   Curves are unconstrained smooths (cos^2 peaks, then decays); dots mark each maximum.
c  cumulative spatial signal p_r over PCs 2..r (r = 3, 5, 10, 21), per-replicate median and
   IQR, with the theoretical non-equilibrium pair p_2 as the dashed ceiling.
d  theoretical spectrum: relative gap of the spatial pair to the next mode (solid, left) and
   the spatial eigenvalue (dashed, right, log), non-equilibrium vs equilibrium.
c and d use the cell phi = 0.5, n_B = 4; the dotted vertical line is b's maximum for n_B = 4.

Reads results/sweep_summaries/split_stepping_stone/{curves,crossings,points}.csv,
      results/analysis/RecentSpatialSpectrum/{spectrum,gaps}.csv,
      results/analysis/RecentSpatialParticipation/{participation,summary,per_rank_replicates}.csv,
      results/analysis/RecentSpatialDemography/pc.csv (grid size K only)
Writes results/figures/main/PhaseTransitionRecentSS.pdf (single column)
"""
import os

import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

import genetics_style as gs
from crossing_panel import load_summary, peak_panel_from_summary, peak_table
from sweep_figure import LINE_H, XAXIS_H as KEY_GAP

_PROJECT_ROOT = gs._ROOT
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "split_stepping_stone")
SPECTRUM_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RecentSpatialSpectrum")
DEMO_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RecentSpatialDemography")
PARTICIPATION_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis",
                                 "RecentSpatialParticipation")

WIDTH = gs.WIDTH["single"] / gs.MM
# mm: two aligned columns; the right margin holds d's twin (log) axis
COL_X, AX_W, AX_H = (9.5, 49.0), 30.5, 31.0
TOP, XAXIS_H = 5.0, 6.5
COLOR_EDGE = "0.2"

TRANSIENT_COLOR, EQUILIBRIUM_COLOR = "#D55E00", "black"
GAP_KW = dict(linewidth=1.0, linestyle="-", zorder=4)               # left axis
NU_KW = dict(linewidth=1.0, linestyle=(0, (3.5, 2)), zorder=4)      # right axis
PEAK_VLINE_KW = dict(color="0.35", linewidth=0.7, linestyle=(0, (1.2, 1.8)), zorder=2)
# Nested PC windows: a light-to-dark single-hue ramp, so the nesting reads as an order
CUM_WINDOWS = (3, 5, 10, 21)
CUM_COLORS = ("#80cbc4", "#26a69a", "#00695c", "#00332e")
CUM_RAMP, CUM_WIDE = CUM_WINDOWS[:-1], CUM_WINDOWS[-1]   # used by the printed checks
THEORY_P2_KW = dict(color="black", linewidth=1.0, linestyle=(0, (4, 2)), zorder=6)
BAND_KW = dict(alpha=0.22, linewidth=0, zorder=2)


def split_time():
    """The simulations' t_split, from the simulation config."""
    path = os.path.join(_PROJECT_ROOT, "src", "simulations", "configs",
                        "split_stepping_stone.yaml")
    with open(path) as f:
        return float(yaml.safe_load(f)["fixed"]["t_split"])


def load_spectrum(spectrum_dir=None):
    """(spectrum, gaps) from RecentSpatialSpectrum, or (None, None) if absent."""
    if spectrum_dir is None:
        spectrum_dir = SPECTRUM_DIR
    spec = os.path.join(spectrum_dir, "spectrum.csv")
    gaps = os.path.join(spectrum_dir, "gaps.csv")
    if not (os.path.exists(spec) and os.path.exists(gaps)):
        return None, None
    return pd.read_csv(spec), pd.read_csv(gaps)


def load_participation(participation_dir=None):
    """(participation, summary, per_rank_replicates), or Nones if absent."""
    if participation_dir is None:
        participation_dir = PARTICIPATION_DIR
    part = os.path.join(participation_dir, "participation.csv")
    summ = os.path.join(participation_dir, "summary.csv")
    ranks = os.path.join(participation_dir, "per_rank_replicates.csv")
    if not (os.path.exists(part) and os.path.exists(summ) and os.path.exists(ranks)):
        return None, None, None
    return pd.read_csv(part), pd.read_csv(summ), pd.read_csv(ranks)


def _cumulative(per_rank_rep, windows=None):
    """Median and IQR of p_r = sum_{k=2..r} p_k, formed per replicate then reduced."""
    if windows is None:
        windows = list(CUM_WINDOWS)
    d = per_rank_rep[per_rank_rep.k >= 2]
    out = []
    for r in windows:
        sub = d[d.k <= r].groupby(["inv_M", "replicate"]).p_k.sum().reset_index()
        g = sub.groupby("inv_M").p_k
        out.append(pd.DataFrame({"inv_M": g.median().index, "r": r,
                                 "med": g.median().values,
                                 "lo": g.quantile(0.25).values,
                                 "hi": g.quantile(0.75).values}))
    return pd.concat(out, ignore_index=True)


def _cumulative_table(per_rank_rep, part, curves, peak_x):
    """Panel c's numbers and its consistency checks, as printable text."""
    cum = _cumulative(per_rank_rep)
    xs = np.sort(cum.inv_M.unique())
    lo_x, hi_x = xs[0], xs[-1]
    mid_x = xs[int(np.argmin(np.abs(xs - peak_x)))]
    windows = list(CUM_WINDOWS)

    def val(r, x):
        g = cum[(cum.r == r) & (cum.inv_M == x)]
        return float(g.med.iloc[0]) if len(g) else np.nan

    lines = ["\npanel C: cumulative spatial signal p_r (median over replicates)",
             f"{'1/M':>9}  " + "  ".join(f"{'p_'+str(r):>7}" for r in windows)]
    for label, x in (("left edge", lo_x), ("A maximum", mid_x), ("right edge", hi_x)):
        lines.append(f"{x:>9.4g}  "
                     + "  ".join(f"{val(r, x):>7.4f}" for r in windows)
                     + f"   ({label})")

    # --- check 1: p_3 against panel A's n_B = 4 curve, doubled ---------------
    ca = curves[(curves.sweep == "split_ss_vary_N") & (curves.level == 4.0)].sort_values("x")
    p3 = cum[cum.r == 3].sort_values("inv_M")
    a_interp = np.interp(np.log(p3.inv_M.values), np.log(ca.x.values), ca.y_fit.values) * 2.0
    d1 = np.abs(p3.med.values - a_interp)
    lines.append(f"\n  1. p_3 vs 2 x panel A (n_B=4): max |diff| = {d1.max():.4f} "
                 f"(at 1/M = {p3.inv_M.values[int(np.argmax(d1))]:.4g}); "
                 f"at the A maximum {np.abs(val(3, mid_x) - 2*np.interp(np.log(mid_x), np.log(ca.x.values), ca.y_fit.values)):.4f}")

    # --- check 2: the widest window against the stored S ---------------------
    # Compared at the SAME window. p_21 sums k = 2..21 while S_k20 sums k = 2..20, so they
    # differ by exactly the k=21 term (~5e-3 at its largest) and are not the same quantity;
    # matching them at k <= 20 is what actually tests that the two routes agree.
    S = part.groupby("inv_M").S_k20.median()
    cum20 = _cumulative(per_rank_rep, windows=[20])
    p20 = cum20.set_index("inv_M").med
    common = S.index.intersection(p20.index)
    d2 = np.abs(S.loc[common].values - p20.loc[common].values)
    p21 = cum[cum.r == CUM_WIDE].set_index("inv_M").med
    lines.append(f"  2. cumulative k<=20 vs stored S (k_max=20): max |diff| = {d2.max():.3e} "
                 f"over {len(common)} M values")
    lines.append(f"     (p_21 itself exceeds S by the k=21 term: max "
                 f"{np.abs(p21.loc[common].values - S.loc[common].values).max():.3e})")

    # --- check 3: free growth from widening ---------------------------------
    N = int(part.N.iloc[0])
    free = 2.0 / (N - 1)
    n_extra = CUM_RAMP[-1] - CUM_RAMP[0]
    lines.append(f"  3. free growth per extra PC ~ 2/(N-1) = {free:.5f} (N={N}); "
                 f"the {n_extra} extra PCs between the narrowest and widest solid window "
                 f"buy {n_extra*free:.5f}")
    narrow, wide = CUM_RAMP[0], CUM_RAMP[-1]
    lines.append(f"     measured p_{wide} - p_{narrow}: "
                 f"{val(wide, mid_x) - val(narrow, mid_x):+.4f} at the A maximum, "
                 f"{val(wide, hi_x) - val(narrow, hi_x):+.4f} at the tail")

    # --- check 4: replicate spread ------------------------------------------
    d = per_rank_rep[per_rank_rep.k >= 2]
    lines.append("  4. replicate std:")
    for label, x in (("A maximum", mid_x), ("tail", hi_x)):
        parts = []
        for r in (3, CUM_WIDE):
            sub = d[(d.inv_M == x) & (d.k <= r)].groupby("replicate").p_k.sum()
            parts.append(f"p_{r} {sub.std():.4f}")
        lines.append(f"     {label:>10}: " + ", ".join(parts))

    # --- the fanning claim, asserted ----------------------------------------
    # Nesting is checked over whatever windows are actually drawn, at EVERY x rather than
    # only at the tail: they are nested by construction, so a violation anywhere would mean
    # the cumulative sums had been built wrong.
    fan = val(wide, hi_x) - val(narrow, hi_x)
    seq = list(CUM_WINDOWS)
    ordered = all(val(a, x) <= val(b, x) + 1e-12
                  for x in xs for a, b in zip(seq, seq[1:]))
    lines.append(f"  curves fan apart at the tail: p_{wide} - p_{narrow} = {fan:+.4f}, "
                 f"nesting holds at all {len(xs)} x: {ordered}")
    return "\n".join(lines)


def _spectrum_table(spec, gaps, peak_x, phi=0.5, n_B=4):
    """Panel d's numbers and the spatial-pair checks, as printable text."""
    g = gaps[(gaps.n_B == n_B) & (gaps.phi == phi)].sort_values("inv_M")
    sp = spec[(spec.n_B == n_B) & (spec.phi == phi)]
    eq_nu = sp[sp.is_spatial].groupby("inv_M").nu_eq.mean().sort_index()

    def at(x):
        i = int(np.argmin(np.abs(g.inv_M.values - x)))
        row = g.iloc[i]
        return row, float(eq_nu.iloc[int(np.argmin(np.abs(eq_nu.index.values - x)))])

    lines = [f"\npanel D (phi={phi:g}, n_B={n_B}): gap and nu_spatial",
             f"{'1/M':>9}  {'gap tr':>8} {'gap eq':>8}  "
             f"{'nu_sp tr':>9} {'nu_sp eq':>9}"]
    for label, x in (("left edge", g.inv_M.min()),
                     ("A maximum", peak_x),
                     ("right edge", g.inv_M.max())):
        row, nu_eq = at(x)
        lines.append(f"{row.inv_M:>9.4g}  {row.gap:>8.4f} {row.gap_eq:>8.4f}  "
                     f"{row.nu_spatial:>9.4f} {nu_eq:>9.4f}   ({label})")

    # Degeneracy of the transient pair: nu_2 vs nu_3 on the is_spatial rows.
    pair = sp[sp.is_spatial].pivot(index="inv_M", columns="mode", values="nu")
    cols = list(pair.columns)
    rel = ((pair[cols[0]] - pair[cols[1]]).abs() / pair[cols[0]]).max()
    lines.append(f"  spatial pair degeneracy: max rel |nu_a - nu_b| = {rel:.3e} "
                 f"(drawn as one curve)")

    at_12 = ((g.pair_i == 1) & (g.pair_j == 2)).mean()
    lines.append(f"  spatial pair at ranks (1,2): {100*at_12:.1f}% of M values"
                 + ("" if at_12 == 1.0
                    else f"  [OTHER RANKS: {sorted(set(zip(g.pair_i, g.pair_j)))}]"))
    lines.append(f"  nu_next is the largest structural mode below the pair, "
                 f"A-vs-B mode excluded")
    return "\n".join(lines)


def schematic(ax, K, t_split):
    """Ancestor (top) splitting t_split generations ago into panmictic A and a K x K grid B."""
    span, x0 = K - 1, 3.4
    for r in range(K):
        ax.plot([x0, x0 + span], [r, r], "-", color=COLOR_EDGE, linewidth=0.5, zorder=1)
        ax.plot([x0 + r, x0 + r], [0, span], "-", color=COLOR_EDGE, linewidth=0.5, zorder=1)
        for c in range(K):
            ax.add_patch(Circle((x0 + c, r), 0.30, facecolor=gs.grid_color(r, c, K),
                                edgecolor=COLOR_EDGE, linewidth=0.4, zorder=2))
    ax.text(x0 + span / 2, -0.7, rf"B (${K}\times{K}$ demes)", ha="center", va="top",
            fontsize=gs.TICK_PT)

    a_r = 1.15
    a_x, a_y = -0.6, span - a_r
    bar_y, split_x = span + 2.4, x0 + span * 0.62
    anc_x, anc_y, anc_r = (a_x + split_x) / 2, bar_y + 2.6, 1.35
    for x, y, r, lab in ((anc_x, anc_y, anc_r, ""), (a_x, a_y, a_r, "A")):
        ax.add_patch(Circle((x, y), r, facecolor=gs.BACKGROUND, edgecolor=COLOR_EDGE,
                            linewidth=0.6, zorder=2))
        ax.text(x, y, lab, ha="center", va="center", fontsize=gs.LABEL_PT)
    line = dict(color=COLOR_EDGE, linewidth=0.8)
    ax.plot([a_x, split_x], [bar_y, bar_y], "-", **line)
    ax.plot([anc_x, anc_x], [anc_y - anc_r, bar_y], "-", **line)
    for x, y in ((a_x, a_y + a_r), (split_x, span + 0.6)):
        ax.add_patch(FancyArrowPatch((x, bar_y), (x, y), arrowstyle="-|>", mutation_scale=6,
                                     shrinkA=0, shrinkB=0, **line))
    ax.text(anc_x + 0.3, bar_y + 0.35, f"{t_split:g} generations ago", ha="left",
            va="bottom", fontsize=gs.TICK_PT)
    ax.set_xlim(a_x - a_r - 0.2, x0 + span + 0.5)
    ax.set_ylim(-2.0, anc_y + anc_r + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")


def cumulative_panel(ax, per_rank_rep, gaps, peak_x, phi=0.5, n_B=4):
    """Panel c: share of the equilibrium spatial plane held by successively wider PC windows."""
    cum = _cumulative(per_rank_rep)
    ax.axhline(2.0, color="0.6", linewidth=0.6, linestyle=":", zorder=1)
    for r, color in zip(CUM_WINDOWS, CUM_COLORS):
        g = cum[cum.r == r].sort_values("inv_M")
        ax.fill_between(g.inv_M, g.lo, g.hi, color=color, **BAND_KW)
        ax.plot(g.inv_M, g.med, color=color, linewidth=1.0, zorder=5)
    gp = gaps[(gaps.n_B == n_B) & (gaps.phi == phi)].sort_values("inv_M")
    if "pair_p2" in gp.columns and gp.pair_p2.notna().any():
        ax.plot(gp.inv_M, gp.pair_p2, **THEORY_P2_KW)
    if np.isfinite(peak_x):
        ax.axvline(peak_x, **PEAK_VLINE_KW)
    ax.set_xscale("log")
    ax.set_xlim(cum.inv_M.min(), cum.inv_M.max())
    ax.set_ylim(0.0, 2.08)
    ax.set_ylabel(r"cumulative spatial signal $S_r$")


def mechanism_panel(ax, spec, gaps, peak_x, phi=0.5, n_B=4):
    """Panel d: the spatial pair's relative gap (left) and eigenvalue (right, log)."""
    g = gaps[(gaps.n_B == n_B) & (gaps.phi == phi)].sort_values("inv_M")
    sp = spec[(spec.n_B == n_B) & (spec.phi == phi)]
    # equilibrium pair is aligned to the transient one upstream (is_spatial rows)
    eq_nu = sp[sp.is_spatial].groupby("inv_M").nu_eq.mean().sort_index()
    ax.plot(g.inv_M, g.gap, color=TRANSIENT_COLOR, **GAP_KW)
    ax.plot(g.inv_M, g.gap_eq, color=EQUILIBRIUM_COLOR, **GAP_KW)
    ax2 = ax.twinx()
    ax2.plot(g.inv_M, g.nu_spatial, color=TRANSIENT_COLOR, **NU_KW)
    ax2.plot(eq_nu.index, eq_nu.values, color=EQUILIBRIUM_COLOR, **NU_KW)
    if np.isfinite(peak_x):
        ax.axvline(peak_x, **PEAK_VLINE_KW)
    ax.set_xscale("log")
    ax.set_xlim(g.inv_M.min(), g.inv_M.max())
    ax.set_ylim(0.0, 0.6)
    ax.set_ylabel("relative spectral gap")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"spatial eigenvalue $\nu_{\mathrm{spatial}}$")
    for a in (ax, ax2):
        a.spines["right"].set_visible(True)
    return ax2


def key_handles(t_split):
    """Bottom key: panel c's windows (left column), panel d's encoding (right column)."""
    c = [Line2D([], [], **THEORY_P2_KW, label="theoretical eigenvectors")]
    c += [Line2D([], [], color=col, linewidth=1.0, label=f"PCs 2–{r}")
          for r, col in zip(CUM_WINDOWS, CUM_COLORS)]
    d = [Line2D([], [], color="0.35", **GAP_KW, label="relative gap (left axis)"),
         Line2D([], [], color="0.35", **NU_KW, label=r"$\nu_{\mathrm{spatial}}$ (right axis)"),
         Line2D([], [], color=TRANSIENT_COLOR, linewidth=2.0,
                label=rf"non-equilibrium ($t_{{\mathrm{{split}}}}={t_split:g}$)"),
         Line2D([], [], color=EQUILIBRIUM_COLOR, linewidth=2.0, label="equilibrium")]
    return c, d


def make_figure(summary_dir=SUMMARY_DIR):
    gs.setup()
    curves, crossings, points, _ = load_summary(summary_dir, with_theory=False)
    t_split = split_time()
    spec, gaps = load_spectrum()
    part, _, per_rank_rep = load_participation()
    K = int(pd.read_csv(os.path.join(DEMO_DIR, "pc.csv"), usecols=["deme_row"]).deme_row.max() + 1)

    curve_key_h = KEY_GAP - XAXIS_H + LINE_H * 2 + 1.0
    pitch = LINE_H + 0.3
    bottom_key_h = pitch * 5 + 1.0
    H = TOP + AX_H + XAXIS_H + curve_key_h + TOP + AX_H + XAXIS_H + bottom_key_h
    fig = gs.blank_figure(WIDTH, H)
    y1 = H - TOP - AX_H
    y2 = y1 - XAXIS_H - curve_key_h - TOP - AX_H

    # a: schematic in the left cell of row 1, using the cell's axis-label space too
    ax_s = gs.axes_mm(fig, 1.0, y1 - XAXIS_H, COL_X[0] + AX_W, AX_H + XAXIS_H + TOP - 1.0)
    schematic(ax_s, K, t_split)

    # b: cosine similarity vs 1/M across n per deme of B (phi = 0.5)
    ax_b = gs.axes_mm(fig, COL_X[1], y1, AX_W, AX_H)
    res_n = peak_panel_from_summary(ax_b, curves, crossings, points, sweep="split_ss_vary_N",
                                    cmap=gs.cm.viridis, show_legend=False)
    h, l = ax_b.get_legend_handles_labels()
    gs.row_key(ax_b, h, [f"{gs.individuals(float(x)):g}" for x in l], KEY_GAP,
               title=r"$n$ per deme of B")
    ax_b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_ylabel(rf"$\langle \hat u_{{t={t_split:g}}}, u_\infty\rangle^2$")

    row2 = [gs.axes_mm(fig, x, y2, AX_W, AX_H) for x in COL_X]
    peak_x = next((r["peak_x"] for r in res_n if np.isclose(r["level"], 4.0)), np.nan)
    cumulative_panel(row2[0], per_rank_rep, gaps, peak_x)
    mechanism_panel(row2[1], spec, gaps, peak_x)
    for ax in [ax_b] + row2:
        ax.set_xlabel(r"$1/M$")

    gs.label_panel_mm(fig, "a", 0.3, H)
    for ax, letter in zip([ax_b] + row2, "bcd"):
        x0, y0 = ax.get_position().x0 * WIDTH, ax.get_position().y1 * H
        gs.label_panel_mm(fig, letter, x0 - (9.2 if letter == "c" else 9.0), y0 + TOP - 0.3)

    centers = [(ax.get_position().x0 + ax.get_position().width / 2) * WIDTH for ax in row2]
    gs.grid_key(fig, key_handles(t_split), centers, y2 - XAXIS_H - 0.5, pitch_mm=pitch)

    print(f"wrote {gs.save(fig, 'PhaseTransitionRecentSS', 'main')}")
    print(peak_table(res_n, "n_B", "1/M"))
    print(_spectrum_table(spec, gaps, peak_x))
    print(_cumulative_table(per_rank_rep, part, curves, peak_x))
    return fig


if __name__ == "__main__":
    make_figure()
