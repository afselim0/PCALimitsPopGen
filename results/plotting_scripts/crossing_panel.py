#%%
"""
Draw one overlap-vs-swept-parameter panel from the pre-fitted summary files.

The shared panel behind the overlap figures. Each curve of the sweep (a value of N, phi or
n-per-deme -- the *curve parameter*) is drawn across the *swept parameter* on the x axis
(Fst or 1/M) as:

    1. the threshold fit through every replicate (flat below the fitted transition,
       monotone increasing above), with its 95% bootstrap band
    2. a filled marker sitting on the curve where it reaches the target cos^2 -- the
       measured detection point, on the line rather than beside it
    3. a hollow marker at the measured detection threshold, taken from the SAMPLE
       EIGENVALUES rather than from the cosine (see below)
    4. a dash-dot vertical line at the theoretical detection threshold (nu(x) = nu_c) and a
       dashed one at the theoretical target crossing, each with a tick per theory route

Each measured mark pairs with the theory line beside it: the filled marker with the dashed
0.9 line, the hollow marker with the dash-dot threshold. The horizontal gap in each pair is
what the detection claim actually is.

Why the threshold marker comes from the eigenvalues. The cosine cannot locate it at finite
N: cos^2 leaves the 1/N random-vector noise floor far below the threshold and already reads
0.1-0.2 when it arrives, so every rule tried for picking a transition out of that smeared
foot was either badly biased (slope onset, 0.63x) or rested on a level with no principled
value. The eigenvalue has a real feature there instead -- below threshold the structure
eigenvalue is PINNED to the bulk edge, so lambda/lambda_bulk = 1 exactly -- and fitting the
spiked-model curve to it recovers the threshold at 0.998 +/- 0.022 with no fitted constant.
See src/analysis/eigenvalue_threshold.py. tau, the fitted transition from threshold_fit,
is the cosine-based counterpart to it; the superseded pickup_x stays in crossings.csv as a
diagnostic and is no longer drawn.

Two theoretical quantities x three routes is six positions per curve, but they cannot be six
styled lines: exact and approx land a fraction of a line width apart (see ROUTE_TICK_KW), so
quantity is carried by line style and route by ticks.

Curves are keyed by a discrete legend, not a continuous colourbar: the curve parameter takes
a handful of chosen values, not a continuum, and a legend states them exactly instead of
asking the reader to interpolate a colour ramp.

This module renders columns and holds no fitting or theory of its own -- stage 3
(src/sweeps_analysis/summarise.py) produces curves.csv, crossings.csv and points.csv.
"""
import os
import sys

import numpy as np
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Appended, not inserted: eleven analysis scripts share a filename with their plotting
# counterpart (SingleSplitSweeps.py, SisterPCAlignment.py, ...), so putting src/analysis
# ahead of this directory would make a plotting script importing a sibling by name get the
# analysis module instead.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src", "analysis"))
from crossing_fit import DEFAULT_TARGET, fit_with_band, theory_crossing

# The scatter is deliberately faint: it is context for the fit, and at 8 replicates x ~100
# x-values it will otherwise swamp the band drawn on top of it.
POINT_KW = dict(s=3, alpha=0.10, linewidths=0, zorder=1)
# Bands here are genuinely narrow (a typical one is ~0.015 in cos^2, a few points on the
# page), so they need enough alpha to register against the scatter rather than blending
# into it.
BAND_KW = dict(alpha=0.45, linewidth=0, zorder=2)
FIT_KW = dict(linewidth=1.7, zorder=3)
# The crossing is marked on the curve itself. A white edge keeps it legible where several
# curves converge at the right of the panel.
CROSS_MARKER_KW = dict(marker="o", markersize=7, markeredgecolor="white",
                       markeredgewidth=1.0, zorder=6, linestyle="none")
# Kept for callers that want to mark a transition themselves; the panels here do not draw
# one (see the module docstring).
# The fitted transition, drawn ON the curve, where it leaves its plateau. A square keeps it
# distinct from the round marker at the target crossing.
# The target crossing's CI, styled to match the transition's so the two read as the same
# kind of statement about the x axis.
CROSS_CI_KW = dict(linewidth=1.6, solid_capstyle="butt", alpha=0.9, zorder=5)

# The transition's bootstrap CI, drawn as a horizontal bar THROUGH the marker at the
# curve's height. Thin and under the marker so a tight CI reads as a marker with whiskers
# rather than as a second mark.
TAU_CI_KW = dict(linewidth=1.6, solid_capstyle="butt", alpha=0.9, zorder=5)

TAU_MARKER_KW = dict(marker="s", markersize=5.5, linestyle="none",
                     markeredgecolor="white", markeredgewidth=0.8, zorder=6)

THEORY_VLINE_KW = dict(linewidth=1.3, linestyle=(0, (4, 2)), zorder=4)
# The theoretical detection threshold, at nu(x) = nu_c: where the eigenvalue separates from
# the bulk and cos^2 leaves 0 in the N -> infinity limit. Dash-dot separates it from the
# target crossing's dashes without reading as the fitted curve's solid line.
THEORY_PICKUP_VLINE_KW = dict(linewidth=1.1, linestyle=(0, (3, 1.5, 1, 1.5)), zorder=4)

# --- the six theoretical positions -----------------------------------------
#
# Two quantities (the detectability transition nu(x)=nu_c, and the target crossing) x three
# routes (exact, approx, crude) = six x-positions per curve. They cannot all be six styled
# lines: measured on african_split_vary_N, exact and approx sit 0.27 points apart at the 0.9
# crossing on a 4-inch log axis, against a ~1.3 pt line width, so two of every three lines
# would overplot into one thicker line while the legend claimed two. The crude route is the
# one that genuinely separates (4-11 pt).
#
# So QUANTITY is carried by line style, drawn once at the exact route -- the reference --
# and ROUTE by small ticks along the top spine, which mark all three positions without
# pretending three lines are resolvable where they are not. Where the routes agree the ticks
# coincide and say so honestly; where they disagree the spread is visible.
ROUTES = ("exact", "approx", "crude")
# Route is carried by its OWN SUB-ROW, not by the shape or weight of a shared mark.
#
# Two earlier attempts failed for the same underlying reason. Three sizes/opacities of one
# bar encode a categorical distinction as a magnitude, and read as "the same mark, fainter".
# Three different shapes at one height collide instead: exact and approx are typically 0.06%
# apart, so a circle drawn for approx simply rings the exact bar, and with ten curves per
# panel the marks smear together.
#
# Separating the routes vertically removes the collision by construction -- two marks can
# now only overlap if they are the same route on two curves -- and keeps one simple shape,
# so what varies between rows is position alone. ROUTE_ROW_DY is in cos^2 units, matching
# the data coordinates the ticks are drawn in.
ROUTE_TICK_KW = {
    "exact":  dict(marker="|", markersize=3.2, markeredgewidth=1.7, linestyle="none"),
    "approx": dict(marker="|", markersize=3.2, markeredgewidth=1.5, linestyle="none",
                   alpha=0.85),
    "crude":  dict(marker="|", markersize=3.2, markeredgewidth=1.5, linestyle="none",
                   alpha=0.85),
}
ROUTE_ROW_DY = {"exact": 0.000, "approx": -0.030, "crude": -0.060}

# Okabe-Ito, matching plotting_style's cycle: discrete curves get discrete colours.
DISCRETE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
                   "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def level_colors(levels, cmap=None):
    """
    One colour per curve, and a formatter for its legend label.

    A sequential colormap is sampled across the levels when one is given, so panels keep
    the ordered light-to-dark reading that suits an ordered parameter; otherwise the
    categorical Okabe-Ito cycle is used.
    """
    levels = list(levels)
    if cmap is not None and len(levels) > 1:
        # Avoid the extreme ends of a sequential map: the palest is invisible on white and
        # the darkest is hard to tell from the next one down.
        return [cmap(0.12 + 0.76 * i / (len(levels) - 1)) for i in range(len(levels))]
    if cmap is not None:
        return [cmap(0.5)]
    return [DISCRETE_COLORS[i % len(DISCRETE_COLORS)] for i in range(len(levels))]


def _curve_legend(ax, title, loc=None, ncol=1):
    """
    The per-curve key, placed outside the axes to the right by default.

    Outside rather than inside because these panels have no reliably empty corner: the
    curves sweep from bottom-left to top-right and the crossings sit along the top, so an
    inset legend covers either the transition or the markers. loc may still be given to
    force it inside.
    """
    if loc is None:
        return ax.legend(title=title, loc="center left", bbox_to_anchor=(1.02, 0.5),
                         fontsize=8, title_fontsize=9, frameon=False,
                         handlelength=1.4, labelspacing=0.3, ncol=ncol,
                         columnspacing=1.2, borderaxespad=0.0)
    return ax.legend(title=title, loc=loc, fontsize=8, title_fontsize=9, frameon=False,
                     handlelength=1.4, labelspacing=0.25, ncol=ncol, columnspacing=1.2)


def format_level(v):
    """Legend label for a curve parameter value: integers stay integers."""
    f = float(v)
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


def crossing_panel(ax, df, level_key, x_key, cmap, cbar_label, target=DEFAULT_TARGET,
                   theory_overlap=None, show_points=True, n_boot=300, seed=0,
                   value_key="cos2", levels=None, norm=None, log_levels=False):
    """
    Draw every curve of one sweep into `ax`.

    Args:
        df: per-replicate rows; needs level_key, x_key and value_key (cos^2) columns.
        level_key: the column holding the curve parameter (e.g. "N", "phi") -- one fitted
            curve per distinct value of it.
        x_key: the column holding the swept parameter on the x axis (e.g. "Fst", "inv_M").
        theory_overlap: curve value -> (x -> predicted cos^2), or None to skip the dashed
            lines. Returning None for a curve skips just that one.
        show_points: scatter the individual replicates behind the fit.
        norm: a shared Normalize, so several panels can key onto one colourbar.

    Returns a list of per-curve result dicts (level, crossing, CI, theory crossing), which
    the caller can print or write out.
    """
    if levels is None:
        levels = np.sort(df[level_key].unique())
    if norm is None:
        norm = (colors.LogNorm(vmin=levels.min(), vmax=levels.max()) if log_levels
                else colors.Normalize(vmin=levels.min(), vmax=levels.max()))

    out = []
    for level in levels:
        d = df[df[level_key] == level]
        x = d[x_key].to_numpy(dtype=float)
        y = d[value_key].to_numpy(dtype=float)
        color = cmap(norm(level))

        if show_points:
            ax.scatter(x, y, color=color, **POINT_KW)

        fit = fit_with_band(x, y, target=target, n_boot=n_boot, seed=seed)
        if fit is None:
            out.append(dict(level=level, crossing=np.nan, crossing_lo=np.nan,
                            crossing_hi=np.nan, theory=np.nan))
            continue

        ax.fill_between(fit["x_grid"], fit["y_lo"], fit["y_hi"], color=color, **BAND_KW)
        ax.plot(fit["x_grid"], fit["y_fit"], "-", color=color, **FIT_KW)

        if np.isfinite(fit["crossing"]):
            ax.axvline(fit["crossing"], color=color, **EMP_VLINE_KW)

        x_star = np.nan
        if theory_overlap is not None:
            fn = theory_overlap(level)
            if fn is not None:
                x_star = theory_crossing(fn, float(np.min(x)), float(np.max(x)),
                                         target=target)
                # Drawn only where the fitted curve also reaches the target. A curve that
                # plateaus below it has no measured crossing, and a lone dashed line there
                # reads as a detection point the data never actually supports -- the
                # number still goes to the crossings table either way.
                if np.isfinite(x_star) and np.isfinite(fit["crossing"]):
                    ax.axvline(x_star, color=color, **THEORY_VLINE_KW)

        out.append(dict(level=level, crossing=fit["crossing"],
                        crossing_lo=fit["crossing_lo"], crossing_hi=fit["crossing_hi"],
                        theory=x_star))

    ax.axhline(target, color="0.6", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(r"squared overlap  $\langle \hat u, u\rangle^2$")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    return out


def load_summary(summary_dir, with_theory=True):
    """
    (curves, crossings, points, theory) for one summary directory.

    theory is None when theory.csv is absent or empty, which is the normal state for a
    theory-free sweep (the recent-split ones) and for summaries written before stage 3
    produced it. The random-deme sweeps write joint.csv instead, whose per-replicate theory
    columns this reads the same way, so a caller needs no special case.
    """
    import pandas as pd

    read = lambda name: pd.read_csv(os.path.join(summary_dir, name))
    curves, crossings, points = read("curves.csv"), read("crossings.csv"), read("points.csv")

    theory = None
    if with_theory:
        for name in ("theory.csv", "joint.csv"):
            path = os.path.join(summary_dir, name)
            if os.path.exists(path) and os.path.getsize(path) > 2:
                try:
                    t = pd.read_csv(path)
                except Exception:
                    continue
                if len(t) and any(c.startswith(("x_c_", "x_cross_", "nu_c_"))
                                  for c in t.columns):
                    theory = t
                    break
    return curves, crossings, points, theory


def variant_suffix(standardized):
    """"_std" for the standardized version of a figure, "" for the plain one."""
    return "_std" if standardized else ""


def pdf_dir_for(figure_dir):
    """
    Where the PDF of a figure goes, given where its PNG goes.

    The two formats are kept in sibling directories -- results/figures holds the PNGs and
    results/pdf_figs the PDFs -- so a caller that passes results/figures gets that pairing
    for free. Any other directory keeps both formats together, which is what the test
    renders under misc/ want.
    """
    parent, name = os.path.split(os.path.normpath(figure_dir))
    return os.path.join(parent, "pdf_figs") if name == "figures" else figure_dir


def save_figure(fig, figure_dir, stem, standardized=False, dpi=300):
    """
    Write <stem><suffix>.png to figure_dir and the matching .pdf to pdf_dir_for(figure_dir).

    Every sweep figure exists twice: once fitted to the overlap measured on the ordinary
    GRM and once on the standardized one, the latter suffixed _std. Both carry the same
    theoretical lines and ticks, so the pair is read by flipping between them.

    Returns the PNG path, which is the one worth printing: it is what gets looked at.
    """
    name = f"{stem}{variant_suffix(standardized)}"
    os.makedirs(figure_dir, exist_ok=True)
    png = os.path.join(figure_dir, f"{name}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")

    pdf_dir = pdf_dir_for(figure_dir)
    os.makedirs(pdf_dir, exist_ok=True)
    fig.savefig(os.path.join(pdf_dir, f"{name}.pdf"), bbox_inches="tight")
    return png


def variant_note(ax, standardized, loc=None):
    """
    Mark a standardized panel as such, so the two versions are never confused.

    Only the standardized one is annotated: the ordinary GRM is the default the rest of the
    project assumes, and labelling both would put a redundant line on every figure.

    Placed inside the axes rather than above them -- the top is already carrying the title
    and two rows of route ticks -- in whichever bottom corner has less drawn in it. Which
    one that is genuinely differs: the monotone sweeps rise to the right and leave the
    bottom right clear, while the concave recent-split ones fall to the right and leave the
    bottom left clear. Pass loc as (x, y) in axes coordinates to override.
    """
    if not standardized:
        return
    if loc is None:
        corner = _emptiest_corner(ax, candidates=("lower right", "lower left"))
        loc = (0.985, 0.02) if corner == "lower right" else (0.015, 0.02)
    ha = "right" if loc[0] > 0.5 else "left"
    # A background box, because the theory rules run the full height of the panel and would
    # otherwise strike through the text wherever a crossing lands near this corner.
    ax.text(loc[0], loc[1], "standardized GRM", transform=ax.transAxes,
            ha=ha, va="bottom", fontsize=8, style="italic", color="0.35", zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))


def _col(frame, name, standardized):
    """
    `name` or its _std twin, whichever this frame actually has.

    Falling back rather than raising is deliberate: a summary written before the
    standardized measurement existed has only the plain columns, and a standardized panel
    over such a summary should degrade to the unstandardized curve rather than fail. The
    theory columns have no _std twin by design -- the theory is a property of the
    demography, not of how the GRM was built -- so they always resolve to the plain name.
    """
    std = f"{name}_std"
    return std if standardized and std in frame.columns else name


def _val(row, name):
    """The scalar in `name` for a one-row frame, or NaN when absent."""
    if row is None or not len(row) or name not in row:
        return np.nan
    try:
        return float(row[name].iloc[0])
    except (TypeError, ValueError):
        return np.nan


def _theory_positions(row, kind, theory=None, level=None):
    """
    The three routes' x-positions for one quantity of one curve, as {route: x}.

    kind is "pickup" (the detectability transition, nu(x) = nu_c) or "cross" (the 0.9
    crossing). Reads crossings.csv's row first; falls back to theory.csv, which is where
    they live once stage 3 writes it, and which is also the frame the random-deme sweeps
    supply per replicate.
    """
    template = {"pickup": "x_c_{}", "cross": "x_cross_{}"}[kind]
    out = {}
    for route in ROUTES:
        v = np.nan
        name = template.format(route)
        if row is not None and len(row) and name in row:
            v = float(row[name].iloc[0])
        if not np.isfinite(v) and theory is not None and name in theory.columns:
            t = theory[theory["level"] == level] if level is not None else theory
            vals = t[name].to_numpy(dtype=float) if len(t) else np.array([])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                # Constant down a curve by construction, except on the per-replicate
                # sweeps, where the median over replicates is the summary that matches
                # how nu itself was defined there. Filtered first: an all-NaN route is a
                # legitimate state (no h, so no exact), not something to warn about.
                v = float(np.median(vals))
        out[route] = v
    return out


def _label_tick_rows(ax, tick_ys=(1.290, 1.110), fontsize=5.5):
    """
    Name the six tick sub-rows down the left margin.

    Six rows of bars are not self-describing: without labels the reader can see that the
    routes differ but not which row is which. The labels sit just outside the left spine in
    axes coordinates, so they do not move with the data.
    """
    for tick_y, group in zip(tick_ys, ("threshold", r"$\langle\hat u,u\rangle^2$")):
        for route in ROUTES:
            y = tick_y + ROUTE_ROW_DY[route]
            ax.text(-0.012, y, route, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=fontsize, color="0.45",
                    clip_on=False)
        ax.text(-0.075, tick_y + ROUTE_ROW_DY["approx"], group,
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=fontsize + 1.0, color="0.3", clip_on=False)


def _draw_theory_marks(ax, positions, color, vline_kw, draw_line=True, tick_y=1.0):
    """
    One quantity's three routes: a styled vertical line at the exact route, plus a tick per
    route just above the top spine.

    The line carries which quantity this is; the ticks resolve the individual routes. Where
    a route is unavailable -- no per-locus weights, so no exact -- its tick is simply absent
    rather than substituted.

    A shaded span across the three routes was tried and dropped: on the transition group the
    crude route sits a factor of two away, so the spans covered most of the panel and, across
    nine curves, stacked into blocks that hid the data they were annotating. The ticks carry
    the same information without painting over the curves.
    """
    vals = [positions.get(r, np.nan) for r in ROUTES]
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return

    ref = positions.get("exact", np.nan)
    if not np.isfinite(ref):
        ref = finite[0]      # exact unavailable: fall back so the quantity is still shown
    if draw_line:
        ax.axvline(ref, color=color, **vline_kw)
    for route in ROUTES:
        v = positions.get(route, np.nan)
        if np.isfinite(v):
            # Data coordinates, not axes fraction: the panel reserves headroom above
            # cos^2 = 1 for these rows, so they must sit at a fixed cos^2 value rather
            # than at a fraction of an axis whose top has moved.
            ax.plot([v], [tick_y + ROUTE_ROW_DY[route]], color=color, clip_on=False,
                    zorder=7, **ROUTE_TICK_KW[route])


def crossing_panel_from_summary(ax, curves, crossings, points=None, sweep=None,
                                cmap=None, cbar_label="", target=DEFAULT_TARGET,
                                levels=None, show_points=True, show_theory_curve=False,
                                show_theory_crossing=True, theory_table=None,
                                standardized=False, show_eig_threshold=True,
                                legend_title=None, legend_loc=None,
                                legend_ncol=1, show_legend=True, **_ignored):
    """
    Draw one panel from the pre-fitted summary files. Fits nothing.

    Stage 3 already did the fit, its confidence band and the theory inversion, so this
    only renders columns. Each curve gets a discrete colour and a legend entry, and carries
    three marks, with a tick per route (exact/approx/crude) above each theory line along the
    top spine:

        filled marker   the measured target crossing, on the curve at the target
        dash-dot line   the theoretical detection threshold (nu(x) = nu_c), at the exact route
        dashed line     the theoretical target crossing, at the exact route

    The detection threshold gets no empirical marker -- see the module docstring for why.
    See the note at ROUTE_TICK_KW for why the routes are ticks and not six separate lines.

    Args:
        curves: rows of curves.csv (x, y_fit, y_lo, y_hi, y_theory).
        crossings: rows of crossings.csv (crossing, pickup_x, theory_crossing_pred, and the
            x_c_*/x_cross_* columns when present).
        points: rows of points.csv for the scatter, or None to omit it.
        theory_table: rows of theory.csv, used for the per-route positions when crossings
            does not carry them, and for the random-deme sweeps, where the routes are
            summarised as the median over replicates. None for a theory-free sweep.
        show_eig_threshold: accepted and ignored; the eigenvalue-threshold marker is no
            longer drawn, the panel marks the fitted transition instead.
            Turn it off for the models where the single-spike fit behind it does not apply
            (the stepping-stones, where K ~ N leaves many axes detached at once).
        standardized: draw the curve, band, scatter and measured marks from the
            standardized-GRM columns (cos2_std, y_fit_std, crossing_std, pickup_x_std).
            The THEORY marks are unaffected and stay exactly as in the unstandardized
            panel: nu, nu_c and the crossings are properties of the demography and the
            locus weights, not of how the sample GRM was built, so the two versions of a
            figure carry identical lines and ticks and differ only in the measurement.
        sweep: filter all frames to this sweep name first.
        cbar_label: used as the legend title (kept under its old name so the figure scripts
            need no edit).
        show_theory_curve: also draw the continuous theoretical curve, not just its crossing.
        show_theory_crossing: draw the theoretical lines and route ticks at all. Off for the
            panels where the theory is not the point being made, and for the theory-free
            recent-split sweeps; the numbers are still returned either way.

    Returns the per-curve crossing rows as a list of dicts.
    """
    if sweep is not None:
        curves = curves[curves["sweep"] == sweep]
        crossings = crossings[crossings["sweep"] == sweep]
        if points is not None:
            points = points[points["sweep"] == sweep]
        if theory_table is not None and "sweep" in theory_table.columns:
            theory_table = theory_table[theory_table["sweep"] == sweep]

    if levels is None:
        levels = np.sort(curves["level"].unique())
    colors_ = level_colors(levels, cmap)

    out = []
    for level, color in zip(levels, colors_):
        c = curves[curves["level"] == level].sort_values("x")
        if c.empty:
            continue

        if show_points and points is not None:
            pt = points[points["level"] == level]
            pcol = "cos2_std" if standardized and "cos2_std" in pt.columns else "cos2"
            ax.scatter(pt["x"], pt[pcol], color=color, **POINT_KW)

        ax.fill_between(c["x"], c[_col(c, "y_lo", standardized)],
                        c[_col(c, "y_hi", standardized)], color=color, **BAND_KW)
        ax.plot(c["x"], c[_col(c, "y_fit", standardized)], "-", color=color,
                label=format_level(level), **FIT_KW)
        if show_theory_curve and c["y_theory"].notna().any():
            ax.plot(c["x"], c["y_theory"], ":", color=color, linewidth=1.0, zorder=3)

        row = crossings[crossings["level"] == level]
        crossing = _val(row, _col(row, "crossing", standardized))
        theory = _val(row, "theory_crossing_pred")
        at_edge = bool(row[_col(row, "crossing_at_edge", standardized)].iloc[0]) \
            if len(row) and _col(row, "crossing_at_edge", standardized) in row else False

        # The measured detection threshold, from the SAMPLE EIGENVALUES rather than the
        # cosine: below threshold the structure eigenvalue is pinned to the bulk edge, so
        # lambda/lambda_bulk = 1 is a real feature of the spectrum, which the cosine has no
        # equivalent of at finite N (see src/analysis/eigenvalue_threshold.py). Kept as a
        # returned diagnostic but no longer drawn: the transition on the panel is now the
        # fitted break below, which is measured from the cosine data itself and sits
        # exactly where the drawn curve leaves its plateau. Two markers for one quantity
        # invited the reading that the curve was picking up away from its own transition.
        x_c_emp = _val(row, "x_c_emp")

        # The fitted transition -- where the drawn curve leaves its plateau. Marked on the
        # curve so it sits exactly where the line lifts off, which is the whole point of
        # the threshold fit. The horizontal bar is its bootstrap CI: tau is a location on
        # the x axis, so its uncertainty is horizontal, and the vertical band beside it
        # answers a different question (how high the curve sits). Both come from the same
        # resamples.
        # Suppressed when the curve was already rising at the smallest swept x: tau is then
        # the left edge, a bound rather than a measurement, and a marker there would read
        # as a detected transition. Older summaries carry the superseded slope-based
        # pickup_x instead; _val gives NaN for whichever is absent.
        tau = _val(row, _col(row, "tau", standardized))
        below_range = bool(_val(row, _col(row, "tau_below_range", standardized)) == 1)
        if np.isfinite(tau) and not below_range and len(c):
            cx = c["x"].to_numpy(float)
            cy = c[_col(c, "y_fit", standardized)].to_numpy(float)
            y_tau = float(np.interp(np.log(tau), np.log(cx), cy))
            lo = _val(row, _col(row, "tau_lo", standardized))
            hi = _val(row, _col(row, "tau_hi", standardized))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.plot([lo, hi], [y_tau, y_tau], color=color, **TAU_CI_KW)
            ax.plot([tau], [y_tau], color=color, **TAU_MARKER_KW)
        pickup_x = tau if np.isfinite(tau) else _val(
            row, _col(row, "pickup_x", standardized))

        if np.isfinite(crossing):
            # On the curve, at the target: the marker is the crossing, so it cannot drift
            # away from the line the way a separate vertical rule could. A curve that only
            # grazes the target as the sweep ends gets a hollow marker -- its crossing is a
            # lower bound set by where the simulation stopped, not a measured detection
            # point, and it would otherwise read as if it were one.
            #
            # Horizontal bar first, so the marker sits on top of it: like the transition,
            # the crossing is a location on the x axis and its uncertainty is horizontal.
            # It comes from where the band edges reach the target, so it is the same
            # bootstrap that produced the band.
            c_lo = _val(row, _col(row, "crossing_lo", standardized))
            c_hi = _val(row, _col(row, "crossing_hi", standardized))
            if np.isfinite(c_lo) and np.isfinite(c_hi) and c_hi > c_lo:
                ax.plot([c_lo, c_hi], [target, target], color=color, **CROSS_CI_KW)
            if at_edge:
                ax.plot([crossing], [target],
                        **{**CROSS_MARKER_KW, "markerfacecolor": "none",
                           "markeredgecolor": color, "markeredgewidth": 1.4})
            else:
                ax.plot([crossing], [target], color=color, **CROSS_MARKER_KW)

        # The two theoretical quantities, each as a styled line at the exact route plus a
        # tick per route. The target crossing is drawn only where the fit also reaches the
        # target: a lone line there reads as a detection point the data never supports. The
        # transition has no such condition -- it is where the eigenvalue separates, which is
        # meaningful whether or not the curve later reaches the target.
        if show_theory_crossing:
            # Two tick rows at different heights, so each row reads with its own line group
            # rather than the two interleaving into one ambiguous comb.
            xc = _theory_positions(row, "pickup", theory_table, level)
            _draw_theory_marks(ax, xc, color, THEORY_PICKUP_VLINE_KW, tick_y=1.290)
            xx = _theory_positions(row, "cross", theory_table, level)
            if not np.isfinite(crossing):
                xx = {r: np.nan for r in ROUTES}
            _draw_theory_marks(ax, xx, color, THEORY_VLINE_KW, tick_y=1.110)
            if np.isfinite(crossing) and np.isfinite(theory) and \
                    not np.isfinite(xx.get("exact", np.nan)):
                # Older summaries carry only theory_crossing_pred; keep drawing it.
                ax.axvline(theory, color=color, **THEORY_VLINE_KW)

        out.append(dict(level=level, crossing=crossing,
                        crossing_lo=_val(row, _col(row, "crossing_lo", standardized)),
                        crossing_hi=_val(row, _col(row, "crossing_hi", standardized)),
                        theory=theory, at_edge=at_edge,
                        pickup=pickup_x, tau=tau,
                        tau_lo=_val(row, _col(row, "tau_lo", standardized)),
                        tau_hi=_val(row, _col(row, "tau_hi", standardized)),
                        x_c_emp=x_c_emp))

    ax.axhline(target, color="0.6", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xscale("log")
    # Headroom for the two route-tick rows, which are drawn just above cos^2 = 1 in axes
    # coordinates. Reserving it in the limits (rather than padding the title) keeps the
    # ticks inside the axes, so a title set later by the figure script cannot collide with
    # them and tight_layout does not crop them.
    ax.set_ylim(-0.04, 1.32 if show_theory_crossing else 1.04)
    if show_theory_crossing:
        _label_tick_rows(ax)
    ax.set_ylabel(r"squared overlap  $\langle \hat u, u\rangle^2$")
    if show_legend:
        _curve_legend(ax, legend_title if legend_title is not None else cbar_label,
                      legend_loc, ncol=legend_ncol)
    return out


def peak_panel_from_summary(ax, curves, crossings, points=None, sweep=None,
                            cmap=None, cbar_label="", target=DEFAULT_TARGET,
                            levels=None, show_points=True, mark_peak=True,
                            standardized=False,
                            legend_title=None, legend_loc=None,
                            show_legend=True, **_ignored):
    """
    Draw a non-monotone sweep: the concave scam fit and its peak, with no crossing markers.

    Where the response turns over, "the first x reaching the target" is not a well-defined
    summary; the peak is. The target line is kept as a reference level, but nothing is
    claimed to cross it.

    Returns per-curve dicts with level / peak_x / peak_y and their CIs.
    """
    if sweep is not None:
        curves = curves[curves["sweep"] == sweep]
        crossings = crossings[crossings["sweep"] == sweep]
        if points is not None:
            points = points[points["sweep"] == sweep]

    if levels is None:
        levels = np.sort(curves["level"].unique())
    colors_ = level_colors(levels, cmap)

    out = []
    for level, color in zip(levels, colors_):
        c = curves[curves["level"] == level].sort_values("x")
        if c.empty:
            continue

        if show_points and points is not None:
            pt = points[points["level"] == level]
            pcol = "cos2_std" if standardized and "cos2_std" in pt.columns else "cos2"
            ax.scatter(pt["x"], pt[pcol], color=color, **POINT_KW)

        ax.fill_between(c["x"], c[_col(c, "y_lo", standardized)],
                        c[_col(c, "y_hi", standardized)], color=color, **BAND_KW)
        ax.plot(c["x"], c[_col(c, "y_fit", standardized)], "-", color=color,
                label=format_level(level), **FIT_KW)

        row = crossings[crossings["level"] == level]
        px = _val(row, _col(row, "peak_x", standardized))
        py = _val(row, _col(row, "peak_y", standardized))
        if mark_peak and np.isfinite(px) and np.isfinite(py):
            ax.plot([px], [py], color=color, **CROSS_MARKER_KW)

        # No empirical transition marker here either, for the same reason as the crossing
        # panel; this demography also publishes no theory at all (see
        # models/split_stepping_stone.py).

        out.append(dict(level=level, peak_x=px, peak_y=py,
                        peak_x_lo=_val(row, _col(row, "peak_x_lo", standardized)),
                        peak_x_hi=_val(row, _col(row, "peak_x_hi", standardized)),
                        peak_y_lo=_val(row, _col(row, "peak_y_lo", standardized)),
                        peak_y_hi=_val(row, _col(row, "peak_y_hi", standardized))))

    ax.axhline(target, color="0.6", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(r"squared overlap  $\langle \hat u, u\rangle^2$")
    if show_legend:
        _curve_legend(ax, legend_title if legend_title is not None else cbar_label,
                      legend_loc)
    return out


def peak_legend(ax, target=DEFAULT_TARGET, loc="lower left"):
    """
    Key for what the marks mean on a non-monotone panel.

    Added as a *second* legend so it does not displace the per-curve one: matplotlib drops
    an existing legend when a new one is set, so the first is re-added as an artist.
    """
    handles = [
        Line2D([], [], color="0.35", linewidth=1.7, label="concave spline fit"),
        Patch(facecolor="0.35", alpha=0.25, edgecolor="none", label="99% confidence band"),
        Line2D([], [], color="0.35", label="peak overlap", **CROSS_MARKER_KW),
        Line2D([], [], color="0.6", linewidth=0.8, linestyle=":",
               label=rf"$\langle\hat u,u\rangle^2={target:g}$"),
    ]
    _add_second_legend(ax, handles, loc)


# Where the curve legend sits, and where the encoding key goes so the two never collide.
_OPPOSITE_LOC = {
    "lower right": "upper left", "lower left": "upper right",
    "upper right": "lower left", "upper left": "lower right",
    "center right": "upper left", "center left": "upper right",
}


def _add_second_legend(ax, handles, loc, force_loc=False):
    """
    Attach the encoding key without discarding or colliding with the per-curve legend.

    matplotlib drops an existing legend when a new one is set, so the first is re-added as
    an artist. The key is also moved to the opposite corner from the curve legend: both
    default to the same side otherwise, and on the concave panels they landed on top of
    each other.
    """
    existing = ax.get_legend()
    if not force_loc:
        # Pick whichever corner has the least drawn in it. The curve legend now lives
        # outside the axes, so the only thing to avoid is the data -- and which corner is
        # free differs per panel (a saturating sweep fills the top, a slow one the bottom),
        # so this is measured rather than assumed.
        loc = _emptiest_corner(ax)
    # A translucent background, unlike the per-curve legend: the key sits wherever the
    # curve legend is not, which on a saturating panel can be right on top of the curves.
    key = ax.legend(handles=handles, loc=loc, fontsize=8, frameon=True,
                    handlelength=1.6, labelspacing=0.3, framealpha=0.85,
                    facecolor="white", edgecolor="none")
    key.set_zorder(10)
    if existing is not None:
        ax.add_artist(existing)
        ax.add_artist(key)


def _emptiest_corner(ax, candidates=("upper left", "lower right", "lower left",
                                     "upper right")):
    """
    The corner quadrant containing the fewest plotted points.

    Counts the scatter and the fitted lines already on the axes, in axes coordinates, and
    returns the candidate corner with the lowest count. Ties keep the candidate order,
    which puts the conventional upper-left first.
    """
    import numpy as _np
    xs, ys = [], []
    for coll in ax.collections:                      # the replicate scatter
        off = coll.get_offsets()
        if off is not None and len(off):
            xs.append(_np.asarray(off[:, 0], float))
            ys.append(_np.asarray(off[:, 1], float))
    for ln in ax.lines:                              # the fitted curves
        if len(ln.get_xdata()) > 2:
            xs.append(_np.asarray(ln.get_xdata(), float))
            ys.append(_np.asarray(ln.get_ydata(), float))
    if not xs:
        return candidates[0]

    x = _np.concatenate(xs)
    y = _np.concatenate(ys)
    ok = _np.isfinite(x) & _np.isfinite(y)
    if not ok.any():
        return candidates[0]
    # Into axes coordinates, so the comparison is about screen position not data units.
    pts = ax.transAxes.inverted().transform(
        ax.transData.transform(_np.column_stack([x[ok], y[ok]])))
    ax_x, ax_y = pts[:, 0], pts[:, 1]

    counts = {}
    for corner in candidates:
        vert, horiz = corner.split()
        in_x = ax_x < 0.42 if horiz == "left" else ax_x > 0.58
        in_y = ax_y > 0.58 if vert == "upper" else ax_y < 0.42
        counts[corner] = int((in_x & in_y).sum())
    return min(candidates, key=lambda c: counts[c])


def _legend_loc_name(legend):
    """The string location a legend was created with, or None if it cannot be recovered."""
    codes = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right",
             5: "right", 6: "center left", 7: "center right", 8: "lower center",
             9: "upper center", 10: "center"}
    try:
        return codes.get(int(legend._loc))
    except Exception:
        return None


def peak_table(results, level_name, x_name):
    """Format the per-level peaks as text, the counterpart of crossing_table."""
    lines = [f"{level_name:>10}  {'peak overlap':>14}  {'68% CI':>20}  {x_name+' at peak':>16}  {'68% CI':>20}"]
    for r in results:
        cy = (f"[{r['peak_y_lo']:.3f}, {r['peak_y_hi']:.3f}]"
              if np.isfinite(r.get("peak_y_lo", np.nan)) else "-")
        cx = (f"[{r['peak_x_lo']:.3g}, {r['peak_x_hi']:.3g}]"
              if np.isfinite(r.get("peak_x_lo", np.nan)) else "-")
        lines.append(f"{r['level']:>10.4g}  {r['peak_y']:>14.3f}  {cy:>20}  "
                     f"{r['peak_x']:>16.4g}  {cx:>20}")
    return "peak of the fitted curve\n" + "\n".join(lines)


def encoding_legend(ax, target=DEFAULT_TARGET, loc="upper left", force_loc=False,
                    show_theory_crossing=True, show_band=True,
                    show_eig_threshold=True):
    """
    Key for what the marks mean on a crossing panel, beside the per-curve legend.

    force_loc keeps `loc` exactly as given instead of auto-placing opposite the curve
    legend -- needed where the auto choice would land the key on the curves themselves.
    show_theory_crossing must match what the panel drew, or the key describes a line that
    is not on it. show_band drops the band swatch where the band is too narrow on the page
    to be worth a key entry. show_eig_threshold is accepted and ignored: the hollow
    eigenvalue-threshold marker it used to key is no longer drawn.
    """
    handles = [
        Line2D([], [], color="0.35", linewidth=1.7, label="threshold fit"),
        Line2D([], [], color="0.35", marker="s", markersize=5.5, linewidth=1.6,
               markeredgecolor="white", markeredgewidth=0.8,
               label="measured transition (fit), 95% CI"),
    ]
    if show_band:
        handles.append(
            Patch(facecolor="0.35", alpha=0.25, edgecolor="none",
                  label="95% bootstrap band"))
    handles.append(
        Line2D([], [], color="0.35", linewidth=1.6,
               label=rf"measured $\langle\hat u,u\rangle^2={target:g}$, 95% CI",
               **CROSS_MARKER_KW))
    if show_theory_crossing:
        handles += [
            Line2D([], [], color="0.35", label="theory: detection threshold",
                   **THEORY_PICKUP_VLINE_KW),
            Line2D([], [], color="0.35",
                   label=rf"theory: $\langle\hat u,u\rangle^2={target:g}$",
                   **THEORY_VLINE_KW),
            # The rows are labelled on the axis itself (_label_tick_rows), so the key only
            # has to say what the marks are.
            Line2D([], [], color="0.35", label="theory route, one tick row each",
                   marker="|", markersize=8, markeredgewidth=1.4, linestyle="none"),
        ]
    _add_second_legend(ax, handles, loc, force_loc=force_loc)


def crossing_table(results, level_name, x_name):
    """
    Format the per-level crossings as text, for printing beside the figure.

    A row flagged at_edge is printed with ">" before the value and no ratio: the curve only
    reached the target as the sweep ended, so the number is a lower bound on the crossing
    rather than a measurement of it.
    """
    lines = [f"{level_name:>10}  {'measured':>13}  {'99% CI':>26}  {'theory':>12}  ratio"]
    for r in results:
        ci = (f"[{r['crossing_lo']:.3g}, {r['crossing_hi']:.3g}]"
              if np.isfinite(r["crossing_lo"]) else "-")
        edge = r.get("at_edge", False)
        meas = ("> " if edge else "") + f"{r['crossing']:.4g}"
        ratio = (r["crossing"] / r["theory"]
                 if np.isfinite(r["crossing"]) and np.isfinite(r["theory"])
                 and r["theory"] and not edge else np.nan)
        lines.append(f"{r['level']:>10.4g}  {meas:>13}  {ci:>26}  "
                     f"{r['theory']:>12.4g}  {ratio:.3f}")
    return f"{x_name} at target overlap\n" + "\n".join(lines)

# %%
