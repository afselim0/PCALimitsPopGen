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
    3. a square ON THE X AXIS at the measured detection threshold, with a capped error bar
       for its CI
    4. a dotted vertical line at the theoretical detection threshold (nu(x) = nu_c) and a
       dashed one at the theoretical target crossing, both at the APPROXIMATE route

Both measured marks sit on the x axis, because both are statements about a location in x
and nothing about their height is meaningful. The horizontal gap between a measured mark
and the theory line beside it is what the detection claim actually is.

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
styled lines: exact and approx land a fraction of a line width apart (see ROUTES), so
only the approximate route is drawn as a line, and only the L_eff ("crude") route is marked
separately -- as a tick below the x axis, where it reads as a coloured x-tick label rather
than as a rule across the data.

Curves are keyed by a discrete legend, not a continuous colourbar: the curve parameter takes
a handful of chosen values, not a continuum, and a legend states them exactly instead of
asking the reader to interpolate a colour ramp.

This module renders columns and holds no fitting or theory of its own -- stage 3
(src/sweeps_analysis/summarise.py) produces curves.csv, crossings.csv and points.csv.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import (cm, collections as mcoll, colors, legend as mlegend,
                        lines as mlines, text as mtext)
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

# Appended, not inserted: eleven analysis scripts share a filename with their plotting
# counterpart (PhaseTransitionSingleSplit.py, PhaseTransitionAfricanSplit.py, ...), so
# putting src/analysis ahead of this directory would make a plotting script importing a
# sibling by name get the analysis module instead.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src", "analysis"))
from crossing_fit import DEFAULT_TARGET, fit_with_band, theory_crossing
from plotting_style import (PANEL_LETTER_PT, TITLE_PT, LABEL_PT, TICK_PT,
                            LEGEND_PT, ANNOT_PT)

# The scatter is deliberately faint: it is context for the fit, and at 8 replicates x ~100
# x-values it will otherwise swamp the band drawn on top of it.
POINT_KW = dict(s=3, alpha=0.10, linewidths=0, zorder=1)
# Bands here are genuinely narrow (a typical one is ~0.015 in cos^2, a few points on the
# page), so they need enough alpha to register against the scatter rather than blending
# into it.
BAND_KW = dict(alpha=0.45, linewidth=0, zorder=2)
FIT_KW = dict(linewidth=1.1, zorder=3)
# The crossing is marked on the curve itself, at the SAME size as the tau square below it:
# both are measured locations on the x axis, and at 4 pt this marker was wider than its own
# CI bar and hid the interval it exists to locate. No white edge either, for the same
# reason -- a 1 pt ring on a small marker is half its radius. Neighbouring markers are
# separated by colour and by the curve each sits on.
CROSS_MARKER_KW = dict(marker="o", markersize=2.2, zorder=5, linestyle="none")
# The target crossing's CI: a CAPPED horizontal bar at the target level through the marker,
# drawn ABOVE it (zorder 6 > 5) so it stays visible. Capped and styled to match TAU_CI_KW --
# an uncapped segment at the target level read as a piece of the curve rather than as an
# interval with ends.
CROSS_CI_KW = dict(linewidth=0.8, capsize=2.2, capthick=0.8, alpha=1.0, zorder=6)

# --- the measured detection threshold, on the x axis ------------------------
#
# tau is a location in x and nothing about its height carries information, so it is drawn
# ON THE X AXIS rather than on the curve. On the curve its CI was a short bar buried in a
# thicket of converging lines and read as part of the fit; on the axis the marks form their
# own row, the CI has clear space to be seen in, and the two measured quantities (this and
# the target crossing) are directly comparable as positions along the same axis.
# No white marker edge here, unlike the crossing circle. The CI bar runs THROUGH this
# marker, and a white edge punches a gap in it on both sides that reads as two detached
# bars rather than one interval.
TAU_MARKER_KW = dict(marker="s", markersize=2.2, linestyle="none", zorder=7,
                     clip_on=False)
# A plain horizontal bar reads as a line segment of unknown extent; capped ends state where
# the interval stops. Drawn with errorbar's capsize, whose caps are vertical ticks.
# capsize sets the HALF-HEIGHT of the two vertical end bars. They must out-reach the
# square (markersize 2.2 pt) or the interval is invisible under it: on a log axis a
# realistic 10% CI spans only ~4% of the panel, which is barely wider than the marker.
TAU_CI_KW = dict(linewidth=0.8, capsize=2.2, capthick=0.8, alpha=1.0, zorder=8,
                 clip_on=False)
# Where that row sits, in AXES FRACTION: on the x axis line itself. Both measured marks are
# statements about a location in x, and putting this one on the axis makes it directly
# comparable with the target crossing while keeping it clear of the curves' rise, which is
# what it collided with when drawn on the curve.
TAU_ROW_Y = 0.0


def style_swept_axis(ax, label, tick_row=None):
    """
    Set the x-axis label, and lift the panel title clear of the L_eff row above the axes.

    The row now sits ABOVE the panel (TICK_ROW_Y > 1), so the x label has nothing to clear
    but the numeric tick labels and needs no extra pad -- it previously carried a 15 pt pad
    for the row when that row was underneath, which stranded it well below the panel. The
    title now has the row beneath it and needs SWEPT_TITLE_PAD instead.

    The pad is set on the Axes' title artist, not passed to set_title(), so it survives the
    figure scripts setting their title text AFTER this call (which all six of them do).
    tick_row is accepted and ignored, so existing callers keep working.
    """
    ax.set_xlabel(label)
    # Only FLAG the axes here. The title cannot be placed at this point: every figure script
    # calls set_title AFTER style_swept_axis, and set_title resets both the position and
    # _autotitlepos. save_figure applies TITLE_Y to every flagged axes just before drawing.
    ax._leff_row_above = True



THEORY_VLINE_KW = dict(linewidth=0.9, linestyle=(0, (4, 2)), zorder=4)
# The theoretical detection threshold, at nu(x) = nu_c: where the eigenvalue separates from
# the bulk and cos^2 leaves 0 in the N -> infinity limit. Dotted, against the target
# crossing's dashes; neither reads as the fitted curve's solid line.
THEORY_PICKUP_VLINE_KW = dict(linewidth=0.9, linestyle=(0, (1.2, 1.8)), zorder=4)

# --- which theoretical routes are drawn ------------------------------------
#
# Two quantities (the detectability transition nu(x)=nu_c, and the target crossing) x three
# routes (exact, approx, crude) is six x-positions per curve, and six styled lines is not a
# readable panel: measured on african_split_vary_N, exact and approx sit 0.27 points apart
# at the crossing on a 4-inch log axis, against a ~1.3 pt line width, so they simply
# overplot while a legend claims two distinct lines.
#
# So only two things are drawn. A VERTICAL LINE per quantity at the APPROXIMATE route --
# exact is indistinguishable from it at this scale, and drawing both would be a claim of
# resolution the page cannot support. And a TICK BELOW THE X AXIS for the crude (L_eff)
# route, which is the one that genuinely separates (4-11 pt) and is a different claim
# rather than a refinement of the same one.
#
# The L_eff row is not a coarser numerical inversion of the same overlap: it is where the
# manuscript's CLOSED-FORM expressions put the two quantities, from inverting
# nu(x) = sqrt(N/(L_eff(1-alpha))) analytically in each demography -- Fst_alpha for the
# single split and OOA3, M_alpha for the two stepping-stones. See the table in
# src/sweeps_analysis/theory_table.py's docstring. That is what makes the row worth a
# separate mark: read against the approximate line, it says how much the analyst who uses
# the formula gives up relative to solving the whole system.
ROUTES = ("exact", "approx", "crude")
# The route each quantity's vertical line is drawn at.
LINE_ROUTE = "approx"
# The route shown as a tick under the axis, and the label naming it there.
TICK_ROUTE = "crude"

# Both L_eff ticks share one row under the x axis -- they are the same kind of claim about
# the same axis -- and are separated by SHAPE, matching the shapes their full-strength
# counterparts use inside the panel: a vertical bar for the threshold (a line quantity), a
# dot for the target crossing (a marker quantity).
# The crude route's two marks. Neither may be the square or the circle: those are the
# MEASURED threshold and crossing inside the panel, and reusing them made the row above the
# panel read as a second copy of the measurement. A down triangle for the threshold and a
# diamond for the crossing stay distinct from each other and from both measured marks at
# 3-4 pt, which the old bar-and-dot pair did not -- the dot was simply a smaller circle.
TICK_KW = {
    "pickup": dict(marker="v", markersize=3.4, linestyle="none"),
    "cross":  dict(marker="D", markersize=2.6, linestyle="none"),
}
# In axes fraction, ABOVE the top of the panel. Under the axis the row had to clear the
# numeric x-tick labels, which forced the x-axis label down by AXIS_LABELPAD and left it
# stranded far below the panel. Above the axes there is nothing to clear, so the axis label
# sits at its normal distance and the marks still read against the same x positions.
TICK_ROW_Y = 1.055
# The panel title must clear the row now sitting above the axes; the default 6 pt pad
# would put the title straight through the marks. Title Y in axes fraction.
TITLE_Y = 1.10
# No extra pad needed now that the row is above the panel: the x label only has to clear the
# numeric tick labels, which is what the default already does.
AXIS_LABELPAD = None

# Where the per-curve key is anchored under its own panel, in axes fraction. A placeholder:
# it has to clear the tick numbers and the x label, whose heights are only known once the
# figure has been drawn, so reserve_curve_legend_row measures them and moves the key down to
# where it actually belongs. Starting it below the axes rather than at 0 keeps the first
# draw legible if a caller skips the layout step.
_CURVE_LEGEND_Y = -0.14
# Where each panel's key actually ended up (x, y in axes fraction), keyed by axes id, so a
# key that has to be rebuilt smaller (see _fit_curve_legend_row) lands back in the same place.
_key_anchor = {}


def _ylabel_x0(ax, renderer, inv):
    """The left edge, in inches, of the panel's y-axis label -- its tick numbers if none."""
    if ax.get_ylabel():
        return ax.yaxis.get_label().get_window_extent(renderer).transformed(inv).x0
    xs = [t.get_window_extent(renderer).transformed(inv).x0
          for t in ax.get_yticklabels() if t.get_text()]
    return min(xs) if xs else ax.get_window_extent(renderer).transformed(inv).x0

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
    The per-curve key, placed UNDER its own panel as a single horizontal row.

    Below rather than inside because these panels have no reliably empty corner: the curves
    sweep from bottom-left to top-right and the crossings sit along the top, so an inset key
    covers either the transition or the markers. Below rather than to the right because a
    side key eats ~1.3 in of a 6.5 in page -- with two panels that is 40% of the width spent
    on two short lists of numbers, and it squeezes each panel out of square. One row under
    the panel, starting at its left spine (on the same vertical as the y-axis label), costs
    one line of height instead.

    The title is set INLINE (an extra invisible handle carrying the title as its label)
    rather than with `title=`, which matplotlib always stacks ABOVE the entries: the key has
    to read as a single line, and a stacked title would make it two. `ncol` is ignored for
    the same reason -- the row is one row, so the column count is the number of entries.
    loc may still be given to force the key inside the axes, in which case the ordinary
    stacked-title form is used.
    """
    fs = plt.rcParams["legend.fontsize"]
    if loc is not None:
        return ax.legend(title=title, loc=loc, fontsize=fs, title_fontsize=fs,
                         frameon=False, handlelength=1.2, labelspacing=0.25, ncol=ncol,
                         columnspacing=1.0)
    handles, labels = ax.get_legend_handles_labels()
    if title:
        # A zero-length line draws nothing, and handlelength=0 for that one entry is not
        # expressible per-handle, so the title rides as a label with an invisible handle.
        handles = [Line2D([], [], linestyle="none", marker="none")] + list(handles)
        labels = [f"{title}:"] + list(labels)
    # Anchored just under the axes box; the caller (reserve_curve_legend_row) opens the
    # bottom margin to hold this row plus the x label and the shared encoding key.
    return _place_curve_legend(ax, handles, labels, fs)


def _place_curve_legend(ax, handles, labels, fontsize, columnspacing=None):
    """
    Draw the one-line key under `ax` at `fontsize`, and remember how to redraw it.

    Anchored at the left edge of the panel's Y-AXIS LABEL (set to the spine here and moved
    by reserve_curve_legend_row, which is where the label's position is known): the label is
    the leftmost thing the panel owns, so starting the key there is what makes the two read
    as one block. `columnspacing` defaults to the tightest legible setting;
    reserve_curve_legend_row then re-solves it so the row SPANS the panel
    (see _stretch_curve_legend).

    The handles and labels are stashed on the legend because changing either the size or the
    spacing means REBUILDING it, not restyling: handlelength and columnspacing are resolved
    into points when the legend is created, so setting them afterwards leaves the box
    essentially the width it was.
    """
    leg = ax.legend(handles, labels, loc="upper left",
                    bbox_to_anchor=(0.0, _CURVE_LEGEND_Y),
                    fontsize=fontsize, title_fontsize=fontsize, frameon=False,
                    handlelength=0.9, ncol=len(handles),
                    columnspacing=CURVE_KEY_COLUMNSPACING if columnspacing is None
                    else columnspacing,
                    handletextpad=0.35, borderaxespad=0.0)
    leg._curve_key_spec = (handles, labels, fontsize)
    return leg


def _stretch_curve_legend(ax, leg, target_in, fig, inv):
    """
    Widen the key's column gaps until the row spans `target_in` -- its panel's width.

    A key laid out at the tightest spacing ends wherever its labels happen to end, which on
    a panel with short labels (N: 100 200 400 800 1600) leaves a third of the row empty and
    makes the key read as belonging to the panel's left half. Distributing the slack into
    the gaps instead lines the last entry up with the panel's right spine.

    columnspacing is in FONT-SIZE units and applies to each of the ncol-1 gaps, so the extra
    width needed converts to a spacing increment directly -- one rebuild, no search. A key
    already at or over the target keeps its spacing: this only ever adds slack, it never
    compresses a row that has none to give.
    """
    handles, labels, fontsize = leg._curve_key_spec
    gaps = max(len(handles) - 1, 1)
    width = leg.get_window_extent(fig.canvas.get_renderer()).transformed(inv).width
    if width >= target_in:
        return leg
    # (inches of slack per gap) / (inches per font-size unit) = spacing units to add.
    extra = (target_in - width) / gaps / (fontsize / 72.0)
    return _place_curve_legend(ax, handles, labels, fontsize,
                               columnspacing=CURVE_KEY_COLUMNSPACING + extra)


def format_level(v):
    """
    Legend label for a curve parameter value: integers stay integers.

    Non-integers are cut to 3 significant digits. Some levels are exact ratios -- the
    sister sweep's harmonic phi runs through 1/15 -- and %g's 6 digits would print those
    as 0.0666667, which reads as a measured quantity rather than as the design point it is.
    """
    f = float(v)
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.3g}"


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

    ax.axhline(target, color="0.6", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_xscale("log")
    # The panel starts at the smallest swept value and ends at the largest. Without this
    # matplotlib autoscales to whatever was drawn, so a theoretical line sitting left of
    # the data opened a band of empty axis below the smallest Fst (or 1/M) ever simulated
    # -- inviting the reading that the sweep covered ground it never did. Marks outside the
    # range are suppressed rather than clipped (they would otherwise pile up on the spine
    # and read as a mark AT the smallest swept value); the numbers stay in crossings.csv.
    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(r"cosine similarity  $\langle \hat u, u\rangle^2$")

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


_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src", "sweeps_analysis", "configs")

_L_CACHE = {}


def sweep_L(sweep):
    """
    The fixed SNP count L for a named sweep, read from its config, or None if not found.

    The configs are the single source of truth for L -- every sweep pins it under `fixed:`
    and subsets to exactly that many polymorphic columns -- so the figure titles quote it
    from there rather than carrying a second copy that can drift out of date.
    """
    if not _L_CACHE:
        import glob
        import yaml
        for path in sorted(glob.glob(os.path.join(_CONFIG_DIR, "*.yaml"))):
            try:
                with open(path) as fh:
                    cfg = yaml.safe_load(fh) or {}
            except Exception:
                continue
            name = cfg.get("name")
            L = (cfg.get("fixed") or {}).get("L")
            if name and L:
                _L_CACHE[name] = int(L)
    return _L_CACHE.get(sweep)


_NTOT_CACHE = {}


def sweep_N_total(sweep):
    """
    The fixed total sample size for a named sweep, or None if the sweep does not pin one.

    Only the sweeps that hold N constant and DERIVE n-per-deme from it declare
    `sampling.fixed.N_total` (the deme-fraction and deme-stride sweeps); the rest fix
    n-per-deme and let N vary, and correctly get None here.

    Read from the config, like sweep_L, rather than from the measured N in the summary: the
    realised N differs by a few samples between cells because n-per-deme is an integer
    division (2D: 567-592), so a title quoting the measured value would report one cell's N
    as if it were the design. N_total is the quantity actually held fixed.
    """
    if not _NTOT_CACHE:
        import glob
        import yaml
        for path in sorted(glob.glob(os.path.join(_CONFIG_DIR, "*.yaml"))):
            try:
                with open(path) as fh:
                    cfg = yaml.safe_load(fh) or {}
            except Exception:
                continue
            name = cfg.get("name")
            n_tot = ((cfg.get("sampling") or {}).get("fixed") or {}).get("N_total")
            if name and n_tot:
                _NTOT_CACHE[name] = int(n_tot)
    return _NTOT_CACHE.get(sweep)


def L_title(sweep, prefix=""):
    """
    `prefix` with the sweep's fixed L folded in as a LaTeX 'L = m x 10^e' clause.

    The L clause goes INSIDE the prefix's trailing parenthesis when it has one, so a title
    carries one bracketed list of fixed parameters ("varying N  (phi=0.5, L=2x10^4)")
    rather than a bracket followed by a second comma-separated clause. A prefix with no
    bracket gets one of its own ("1D  (L=1x10^4)").

    Returns `prefix` unchanged when the config has no L, so a title never ends up with a
    dangling empty parenthesis.
    """
    L = sweep_L(sweep)
    if L is None:
        return prefix
    # Plain thousands-separated integer, not scientific notation: "L = 10,000" is read at
    # a glance where "L = 1 x 10^4" has to be decoded, and every L in this project is a
    # round SNP count of at most seven digits.
    clause = rf"$L = {int(round(L)):,}$".replace(",", "{,}")
    if not prefix:
        return clause
    stripped = prefix.rstrip()
    if stripped.endswith(")"):
        return f"{stripped[:-1]}, {clause})"
    return f"{stripped}  ({clause})"


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


# Every figure in the paper is written at one size and one resolution, set here rather
# than in the twenty scripts that call save_figure.
FIGURE_MAX_WIDTH_IN = 6.5      # single-column-to-full-page width of the manuscript
FIGURE_DPI = 500
# Widest a figure may be relative to its height. A 13 x 5.2 in row scaled to 6.5 in would be
# 2.5:1 and too squat to read at print size; 1.9:1 keeps two side-by-side panels close to
# the proportions they were designed at.
# Only a backstop against a figure declared far wider than tall. Scripts now declare their
# FINAL size directly (figsize(6.5, 2.95)), so this must not pull the height back up: at 1.9
# it silently overrode every sweep figure's requested height and re-opened the dead band
# between the x label and the key.
MAX_ASPECT = 2.6
# Type sizes come from ONE table in plotting_style, so a title is the same size in every
# figure and a legend the same size as every other legend. Nothing here redefines a size;
# these are aliases kept so existing imports keep working.
PANEL_LETTER_SIZE = PANEL_LETTER_PT
# The shared mark-definition key under a panel row is a legend like any other, and is set at
# the common legend size. It was previously shrunk to 7 pt to stop its six long entries from
# running wider than the page -- which made it unreadably small next to the 8 pt side keys.
# The width is now handled by laying it out in two rows of three (see
# figure_encoding_legend), which keeps it legible AND inside the page.
ENCODING_KEY_SIZE = LEGEND_PT

# Extra inches of clear space between two panels, on top of the room their axis labels and
# tick numbers need. With the curve keys moved under the panels the axes could run edge to
# edge, which left the two panels (and the two keys under them) reading as one band; this
# separates them.
PANEL_GAP_IN = 0.34
# The tightest column gap a curve key is laid out at, in font-size units. It is a FLOOR,
# not the final value: reserve_curve_legend_row widens the gaps so the row spans its panel.
CURVE_KEY_COLUMNSPACING = 0.6
# The strip at each side of the page a key may not cross. Not zero: bbox_inches="tight"
# trims to what was drawn, so a key ending exactly at the edge still reads as running off it.
PAGE_MARGIN_IN = 0.02



def figsize(width_in, height_in, max_width_in=FIGURE_MAX_WIDTH_IN, scale_text=False,
            max_aspect=MAX_ASPECT):
    """
    A figsize capped at `max_width_in` wide, with the type scaled to match the new page.

    Call this where the figure is CREATED, not at save time: matplotlib lays a figure out
    for the size it is given, so a figure built at 13 in and shrunk afterwards keeps 13
    in-worth of type and legend anchors on a 6.5 in page, and its legends collide with its
    axes. Building it at the final size lets tight_layout place everything for that page.

    Width sets the scale factor and the type scales with it. HEIGHT IS ALLOWED TO GROW back
    to `max_aspect` (width:height): a pure proportional squeeze turns a 13 x 5.2 in
    two-panel row into 6.5 x 2.6 in, where the panels are too squat to read and the side
    legends crowd the axes. Capping the aspect keeps each panel near the proportions a
    two-panel journal figure normally has. A figure already taller than the cap keeps its
    own aspect -- this only ever adds height, never removes it.

    scale_text is OFF by default: plotting_style now sets the manuscript's absolute point
    sizes (11 pt panel letters and legends, 10 pt titles, 8 pt axis labels), which are
    specified for the final 6.5 in page and must not be rescaled by how wide the figure was
    originally declared. Pass scale_text=True only for a figure that genuinely wants its
    type scaled with the canvas.
    """
    width_in = float(width_in)
    height_in = float(height_in)
    scale = min(1.0, max_width_in / width_in) if width_in else 1.0
    if scale_text and scale < 1.0:
        _scale_rc_text(scale)
    w = width_in * scale
    h = height_in * scale
    if max_aspect and w / max(h, 1e-9) > max_aspect:
        h = w / max_aspect
    return (w, h)


# The rcParams that carry a size in points and so must be scaled with the canvas.
_RC_POINT_KEYS = ("font.size", "axes.titlesize", "axes.labelsize",
                  "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
                  "legend.title_fontsize", "lines.markersize", "lines.linewidth",
                  "axes.linewidth", "xtick.major.size", "ytick.major.size",
                  "xtick.minor.size", "ytick.minor.size", "xtick.major.width",
                  "ytick.major.width", "xtick.minor.width", "ytick.minor.width",
                  "errorbar.capsize", "patch.linewidth", "grid.linewidth",
                  "axes.labelpad", "axes.titlepad", "xtick.major.pad", "ytick.major.pad")


def _scale_rc_text(scale):
    """Multiply every point-valued rcParam by `scale`, skipping symbolic sizes."""
    for key in _RC_POINT_KEYS:
        try:
            v = plt.rcParams[key]
        except KeyError:
            continue
        if isinstance(v, (int, float)):
            plt.rcParams[key] = v * scale


def save_figure(fig, figure_dir, stem, standardized=False, dpi=FIGURE_DPI):
    """
    Write <stem><suffix>.png to figure_dir and the matching .pdf to pdf_dir_for(figure_dir).

    `dpi` sets the PNG's resolution. The PDF takes no dpi: it comes out pure vector (paths
    plus Type 42 fonts, no /Image objects), so it has no resolution to set -- it renders at
    whatever the viewer or printer uses. Only rasterized artists would consult dpi there.

    The width cap is NOT applied here: it is applied where
    the figure is created, by figsize(), because matplotlib lays a figure out for the size
    it is given and resizing afterwards leaves the type and the legend anchors sized for
    the old page.

    Every sweep figure exists twice: once fitted to the overlap measured on the ordinary
    GRM and once on the standardized one, the latter suffixed _std. Both carry the same
    theoretical lines and ticks, so the pair is read by flipping between them.

    Returns the PNG path, which is the one worth printing: it is what gets looked at.
    """
    # Lift the title clear of the L_eff tick row on any panel that has one. Done here,
    # not in style_swept_axis, because the figure scripts set their titles after that call
    # and set_title resets the title's position and re-enables automatic placement.
    for ax in fig.axes:
        if getattr(ax, '_leff_row_above', False) and ax.get_title():
            ax.title.set_position((0.5, TITLE_Y))
            ax._autotitlepos = False
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
            ha=ha, va="bottom", fontsize=LABEL_PT, style="italic", color="0.35", zorder=8,
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


def _draw_theory_marks(ax, positions, color, vline_kw, kind, draw_line=True,
                       x_min=None):
    """
    One theoretical quantity for one curve: a vertical line at the approximate route, and a
    tick under the x axis at the crude (L_eff) route.

    `kind` is "pickup" or "cross" and selects the tick's shape. The exact route is computed
    upstream and stays in crossings.csv, but is not drawn: it lands well inside a line width
    of the approximate one (see the note at ROUTES), so a second line there would claim a
    distinction the page cannot show.

    `x_min` is the smallest swept value. A position below it is DROPPED, not clipped: the
    panel is clamped to the swept range, so a clipped mark would sit on the left spine and
    read as a prediction AT the smallest simulated Fst (or 1/M) rather than below it. A
    theory that fires outside the range is a real finding -- it stays in crossings.csv and
    in the printed table -- but the panel cannot honestly place it. This bites hardest at
    large N, where the predicted threshold falls below the swept grid.
    """
    if x_min is not None and np.isfinite(x_min):
        positions = {r: (v if np.isfinite(v) and v >= x_min else np.nan)
                     for r, v in positions.items()}

    ref = positions.get(LINE_ROUTE, np.nan)
    if not np.isfinite(ref):
        # Approximate unavailable: fall back to whichever route this curve does have, so the
        # quantity is still marked rather than silently missing.
        ref = next((positions[r] for r in ROUTES
                    if np.isfinite(positions.get(r, np.nan))), np.nan)
    if draw_line and np.isfinite(ref):
        ax.axvline(ref, color=color, **vline_kw)

    tick = positions.get(TICK_ROUTE, np.nan)
    if np.isfinite(tick):
        # Axes-fraction y, data-coordinate x: the row must stay pinned under the spine
        # whatever the y limits are.
        (line,) = ax.plot([tick], [TICK_ROW_Y], color=color, clip_on=False, zorder=7,
                          transform=ax.get_xaxis_transform(), **TICK_KW[kind])
        # Marks this axes as carrying an L_eff row, so style_swept_axis can pad the x label
        # by what is actually drawn rather than by what the caller declared.
        line._is_leff_tick = True


def _draw_tau_on_axis(ax, tau, lo, hi, color):
    """The measured detection threshold as a square in the top row, with a capped CI bar."""
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.errorbar([tau], [TAU_ROW_Y], xerr=[[tau - lo], [hi - tau]],
                    color=color, ecolor=color, fmt="none",
                    transform=ax.get_xaxis_transform(), **TAU_CI_KW)
    ax.plot([tau], [TAU_ROW_Y], color=color,
            transform=ax.get_xaxis_transform(), **TAU_MARKER_KW)


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
    only renders columns. Each curve gets a discrete colour and a legend entry, and carries:

        filled circle   the measured target crossing, on the curve at the target
        square on the   the measured detection threshold (the fitted transition), with a
        x axis            capped error bar for its 95% CI
        dotted line     the theoretical detection threshold (nu(x) = nu_c), approx route
        dashed line     the theoretical target crossing, approx route
        ticks under     the same two theoretical quantities on the crude (L_eff) route
        the x axis

    See the note at ROUTES for which routes are drawn and why the rest are not.

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
    # The swept range, accumulated over the curves actually drawn. The panel is clamped to
    # it below, and no theoretical mark is drawn outside it -- see _draw_theory_marks.
    x_min, x_max = np.inf, -np.inf
    for level, color in zip(levels, colors_):
        c = curves[curves["level"] == level].sort_values("x")
        if c.empty:
            continue
        x_min = min(x_min, float(c["x"].min()))
        x_max = max(x_max, float(c["x"].max()))

        if show_points and points is not None:
            pt = points[points["level"] == level]
            pcol = "cos2_std" if standardized and "cos2_std" in pt.columns else "cos2"
            ax.scatter(pt["x"], pt[pcol], color=color, **POINT_KW)

        ax.fill_between(c["x"], c[_col(c, "y_lo", standardized)],
                        c[_col(c, "y_hi", standardized)], color=color, **BAND_KW)
        ax.plot(c["x"], c[_col(c, "y_fit", standardized)], "-", color=color,
                label=format_level(level), **FIT_KW)
        if show_theory_curve and c["y_theory"].notna().any():
            ax.plot(c["x"], c["y_theory"], ":", color=color, linewidth=0.8, zorder=3)

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

        # The fitted transition -- where the drawn curve leaves its plateau. Drawn on the
        # x axis rather than on the curve: tau is a location in x, its uncertainty is
        # horizontal, and its height on the curve says nothing. On the axis its CI has the
        # clear space to actually be read (see TAU_MARKER_KW).
        # Suppressed when the curve was already rising at the smallest swept x: tau is then
        # the left edge, a bound rather than a measurement, and a marker there would read
        # as a detected transition. Older summaries carry the superseded slope-based
        # pickup_x instead; _val gives NaN for whichever is absent.
        tau = _val(row, _col(row, "tau", standardized))
        below_range = bool(_val(row, _col(row, "tau_below_range", standardized)) == 1)
        if np.isfinite(tau) and not below_range:
            _draw_tau_on_axis(ax, tau,
                              _val(row, _col(row, "tau_lo", standardized)),
                              _val(row, _col(row, "tau_hi", standardized)), color)
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
                ax.errorbar([crossing], [target],
                            xerr=[[crossing - c_lo], [c_hi - crossing]],
                            color=color, ecolor=color, fmt="none", **CROSS_CI_KW)
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
            xc = _theory_positions(row, "pickup", theory_table, level)
            _draw_theory_marks(ax, xc, color, THEORY_PICKUP_VLINE_KW, "pickup",
                               x_min=x_min)
            xx = _theory_positions(row, "cross", theory_table, level)
            if not np.isfinite(crossing):
                xx = {r: np.nan for r in ROUTES}
            _draw_theory_marks(ax, xx, color, THEORY_VLINE_KW, "cross", x_min=x_min)
            if np.isfinite(crossing) and np.isfinite(theory) and \
                    not np.isfinite(xx.get("exact", np.nan)) and theory >= x_min:
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

    ax.axhline(target, color="0.6", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_xscale("log")
    # The panel starts at the smallest swept value and ends at the largest. Without this
    # matplotlib autoscales to whatever was drawn, so a theoretical line sitting left of
    # the data opened a band of empty axis below the smallest Fst (or 1/M) ever simulated
    # -- inviting the reading that the sweep covered ground it never did. Marks outside the
    # range are suppressed rather than clipped (they would otherwise pile up on the spine
    # and read as a mark AT the smallest swept value); the numbers stay in crossings.csv.
    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        ax.set_xlim(x_min, x_max)
    # No headroom needed: both the measured-threshold row and the L_eff ticks sit on or
    # below the x axis, so the panel spans the data range alone.
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(r"cosine similarity  $\langle \hat u, u\rangle^2$")
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
    # The swept range, accumulated over the curves actually drawn. The panel is clamped to
    # it below, and no theoretical mark is drawn outside it -- see _draw_theory_marks.
    x_min, x_max = np.inf, -np.inf
    for level, color in zip(levels, colors_):
        c = curves[curves["level"] == level].sort_values("x")
        if c.empty:
            continue
        x_min = min(x_min, float(c["x"].min()))
        x_max = max(x_max, float(c["x"].max()))

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

    ax.axhline(target, color="0.6", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_xscale("log")
    # The panel starts at the smallest swept value and ends at the largest. Without this
    # matplotlib autoscales to whatever was drawn, so a theoretical line sitting left of
    # the data opened a band of empty axis below the smallest Fst (or 1/M) ever simulated
    # -- inviting the reading that the sweep covered ground it never did. Marks outside the
    # range are suppressed rather than clipped (they would otherwise pile up on the spine
    # and read as a mark AT the smallest swept value); the numbers stay in crossings.csv.
    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(r"cosine similarity  $\langle \hat u, u\rangle^2$")
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
    _add_second_legend(ax, peak_handles(target), loc)


def peak_handles(target=DEFAULT_TARGET):
    """
    The legend handles for a non-monotone (peak) panel: the fit and its maximum.

    Two entries, not four. The band hugging its own curve and the dotted target level are
    self-evident once the fit is named, and keying them crowded out the two statements the
    panel exists to make. `target` is kept in the signature so callers need no edit.
    """
    return [
        Line2D([], [], color="0.35", linewidth=1.1, label="smoothing spline fit"),
        Line2D([], [], color="0.35", label="maximum cosine similarity", **CROSS_MARKER_KW),
    ]


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
    key = ax.legend(handles=handles, loc=loc, fontsize=LEGEND_PT, frameon=True,
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


def _ci_handle(marker, markersize, label, color="0.35"):
    """
    A legend handle showing a marker on a CAPPED error bar, as the panel draws it.

    An ErrorbarContainer rather than a Line2D: matplotlib's legend renders a container's
    caps, so the handle carries the two vertical end bars that a plain line cannot show. The
    container is built free-standing (no axes), which is all the legend needs to draw it.

    The bar element must be a LineCollection, which is what errorbar itself produces; the
    legend handler copies collection properties off it and raises on a plain Line2D.
    """
    bar = mcoll.LineCollection([], colors=color, linewidths=0.8)
    caps = tuple(Line2D([], [], color=color, marker="|",
                        markersize=2 * TAU_CI_KW["capsize"],
                        markeredgewidth=TAU_CI_KW["capthick"], linestyle="none")
                 for _ in range(2))
    point = Line2D([], [], color=color, marker=marker, markersize=markersize,
                   linestyle="none")
    return ErrorbarContainer((point, caps, (bar,)), has_xerr=True, label=label)


def encoding_handles(target=DEFAULT_TARGET, show_theory_crossing=True, show_band=True):
    """
    The legend handles for a crossing panel, in the order the key reads them.

    Six statements about the panel, in two columns of three. Each COLUMN is one quantity --
    the detection threshold on the left, the target crossing on the right -- and each ROW is
    one route, so the label reads "<quantity> (<route>)" throughout: the close approximation,
    the crude approximation, and the I-spline fit through the replicates. Grouping by
    quantity rather than by route is what lets the key be read down a column: the three ways
    of placing the same thing sit together, and naming the route in parentheses is what makes
    them comparable at a glance.

    Matplotlib fills a legend COLUMN-MAJOR, so the list below is simply the left column
    followed by the right. Reordering these without accounting for the fill order silently
    transposes the key.

    A theory-free sweep keeps only the two measured entries, so the key is six or two
    entries here, never anything in between.
    """
    # The fit and its band are not keyed: a solid coloured line through a scatter and the
    # band hugging it are self-evident, and naming them crowded out the six statements the
    # key exists to make. show_band is kept in the signature so the figure scripts need no
    # edit, but no longer adds an entry.
    cross = rf"\langle\hat u,u\rangle^2={target:g}"
    # "I-spline fit", not "measured": both of these are read off the monotone spline fitted
    # to the replicates, so naming the fit says where the number came from rather than
    # implying it was observed directly at some swept value.
    #
    # Drawn as CAPPED error bars, not as a marker on a plain line: in the panel both of
    # these carry a capped CI (TAU_CI_KW / CROSS_CI_KW), and a Line2D handle cannot render
    # the caps -- it showed the marker on a bare segment, so the key was missing the very
    # vertical end bars that say where each interval stops.
    threshold_measured = _ci_handle(
        "s", 3.0, "detection threshold (I-spline fit)")
    cross_measured = _ci_handle(
        "o", 2.2, rf"${cross}$ crossing point (I-spline fit)")
    if not show_theory_crossing:
        return [threshold_measured, cross_measured]

    # Column-major: first three -> LEFT column (the detection threshold, three ways),
    # last three -> RIGHT column (the target crossing, the same three ways).
    return [
        Line2D([], [], color="0.35",
               label="detection threshold (close approximation)",
               **THEORY_PICKUP_VLINE_KW),
        # The crude route's marks are drawn as a row above the panel; the key is the only
        # place they are named, so its labels say both which quantity and which route.
        Line2D([], [], color="0.35",
               label="detection threshold (crude approximation)",
               **TICK_KW["pickup"]),
        threshold_measured,
        Line2D([], [], color="0.35",
               label=rf"${cross}$ crossing point (close approximation)",
               **THEORY_VLINE_KW),
        Line2D([], [], color="0.35",
               label=rf"${cross}$ crossing point (crude approximation)",
               **TICK_KW["cross"]),
        cross_measured,
    ]


def encoding_bottom_margin(handles, ncol=2, per_row=None, base=None, fig=None):
    """
    The figure bottom margin (a fraction) that holds `handles` laid out in `ncol` columns.

    Derived from the number of rows the key will actually occupy rather than fixed, so a
    panel whose key loses entries (a theory-free sweep drops three) does not leave a band
    of empty page under it.

    The key's height is a number of POINTS, so the reservation is computed in inches and
    then converted to a fraction of THIS figure. A constant fraction meant a taller figure
    reserved proportionally more, which is what opened a band of dead page between the x
    label and the key when the sweeps grew taller to fit 12 pt type.
    """
    rows = int(np.ceil(len(handles) / float(ncol)))
    row_in = (LEGEND_PT * 1.9) / 72.0 if per_row is None else per_row
    base_in = (LABEL_PT + TICK_PT) * 2.2 / 72.0 if base is None else base
    h = fig.get_size_inches()[1] if fig is not None else 3.15
    return (base_in + row_in * rows) / h


def figure_encoding_legend(fig, handles, ncol=2, y=None, fontsize=ENCODING_KEY_SIZE):
    """
    Put the encoding key under the panels, one COLUMN under each panel.

    One key for the whole figure rather than one per panel: the marks mean the same thing in
    every panel, and an in-axes key on a saturating sweep has nowhere to sit that is not on
    top of a curve.

    Its columns are drawn as SEPARATE legends, one per panel, each starting at the same x as
    that panel's curve key -- so the two blocks under the row line up with the two blocks
    above it and the whole bottom of the figure reads on two verticals. A single two-column
    fig.legend cannot do this: matplotlib sizes a legend's columns to their content and
    centres the block, so its second column lands wherever the first one's longest label
    happens to end, which matched neither panel. Centring the block on the figure was worse
    still -- bbox_inches="tight" crops to the artwork, so a canvas-centred key ends up
    visibly left of centre in the written file.

    The caller must leave room for it (fig.subplots_adjust(bottom=...)); a legend anchored
    outside the axes is invisible to tight_layout.

    Drawn at the common legend size (ENCODING_KEY_SIZE == LEGEND_PT), the same as the
    per-curve keys: a key that defines the marks is no less important than one naming the
    curves, and setting it smaller made it the least readable text in the figure while the
    side keys beside it were larger.

    Its entries are long ("<u,u>^2 = 0.8 crossing point (close approximation)"), so a single
    row runs far wider than the 6.5 in page -- and bbox_inches="tight" then EXPANDS the saved
    figure to contain the legend, silently defeating the width cap. Two columns of three is
    what fits. The width comes from the column count, not from shrinking the type.

    Returns the list of column legends.
    """
    # Anchored to the TOP of the band reserved by encoding_bottom_margin (i.e. just under
    # the axes), not to the figure's bottom edge: on a tall figure the reservation is a
    # small fraction and pinning the key to y~0 left a band of dead page above it.
    if y is None:
        # Clear everything that hangs below the axes box -- the tick numbers, the x label,
        # and (since the per-curve keys moved under the panels) those keys too. Measuring
        # the artists rather than adding a constant is what keeps this correct whichever of
        # them a given figure actually has.
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        r = fig.canvas.get_renderer()
        lows = [inv.transform_bbox(ax.xaxis.get_label().get_window_extent(r)).y0
                for ax in fig.axes if ax.get_xlabel()]
        lows += [inv.transform_bbox(leg.get_window_extent(r)).y0
                 for ax in fig.axes for leg in ax.findobj(mlegend.Legend)]
        y = (min(lows) if lows else fig.subplotpars.bottom) - 0.015
    # Column-major, matching the order encoding_handles builds: the first ceil(n/ncol)
    # handles are the left column, the rest the right.
    rows = int(np.ceil(len(handles) / float(ncol)))
    columns = [handles[i * rows:(i + 1) * rows] for i in range(ncol)]
    starts = _curve_key_starts(fig, ncol)
    legs = []
    for i, (column, x) in enumerate(zip(columns, starts)):
        if not column:
            continue
        # How far this column may run: to the next column's start, or to the page edge for
        # the last one. A column that starts at panel B's key and carries the longest label
        # ("<u,u>^2 = 0.8 crossing point (crude approximation)") overruns the page, and
        # bbox_inches="tight" then expands the saved figure to contain it.
        w = fig.get_size_inches()[0]
        limit = starts[i + 1] if i + 1 < len(starts) else 1.0 - PAGE_MARGIN_IN / w
        legs.append(_fitted_column_legend(fig, column, x, y, fontsize, limit))
    return legs


def _fitted_column_legend(fig, handles, x, y, fontsize, limit):
    """
    One column of the encoding key at `x`, shrunk if it would run past `limit`.

    Rebuilt rather than restyled, and floored at ANNOT_PT: handlelength resolves into points
    when the legend is created, so setting the texts smaller afterwards barely narrows the
    box, and dropping below the manuscript's smallest size to fit a label would trade one
    typographic rule for another.
    """
    def build(fs):
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y),
                         ncol=1, fontsize=fs, frameon=False,
                         handlelength=1.2, labelspacing=0.3, handletextpad=0.4)
        leg.set_zorder(10)
        return leg

    leg = build(fontsize)
    budget = (limit - x) * fig.get_size_inches()[0]
    inv = fig.dpi_scale_trans.inverted()
    fs = fontsize
    for _ in range(6):
        fig.canvas.draw()
        width = leg.get_window_extent(fig.canvas.get_renderer()).transformed(inv).width
        if width <= budget or fs <= ANNOT_PT:
            break
        fs = max(ANNOT_PT, fs * budget / width)
        leg.remove()
        leg = build(fs)
    return leg


def _curve_key_starts(fig, ncol):
    """
    The x (figure fraction) each encoding-key column starts at: its panel's curve key.

    Falls back to dividing the figure evenly when a panel has no curve key to align to, so a
    figure that suppressed those keys still gets sensibly spaced columns.
    """
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    starts = []
    for ax in fig.axes:
        keys = ax.findobj(mlegend.Legend)
        if keys:
            starts.append(inv.transform_bbox(keys[0].get_window_extent(r)).x0)
    starts = sorted(starts)[:ncol]
    while len(starts) < ncol:
        starts.append(len(starts) / float(ncol))
    return starts


def reserve_curve_legend_row(fig, n_panels=2, bottom=None, wspace=None,
                             left_in=None, top_in=None, gap_in=0.06):
    """
    Lay a sweep panel row out with each curve key on ONE line under its own panel.

    The keys are anchored outside the axes, so tight_layout cannot see them: it spreads the
    axes over the whole canvas and whatever sits under a panel is then drawn off the page,
    or on top of the figure-level encoding key. Calling this AFTER the keys are created is
    what reserves room for them.

    Two things happen here. The bottom margin is opened to hold the tick numbers, the x
    label and one legend row (plus whatever the caller asked for in `bottom`, which is the
    band the shared encoding key needs). And each key is then MOVED to just below its own
    panel's x label -- measured, not assumed, because the label's depth depends on the type
    sizes and on whether the panel carries a tick row.

    Nothing is reserved horizontally any more: with the keys underneath, the panels take
    the full page width and stay square.
    """
    if left_in is None:
        left_in = 0.50 * (LABEL_PT + TICK_PT) / 13.0
    if top_in is None:
        top_in = 0.28 * TITLE_PT / 8.0
    w, h = fig.get_size_inches()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()

    legs = [(ax, leg) for ax in fig.axes for leg in ax.findobj(mlegend.Legend)]
    row_in = max([leg.get_window_extent(r).transformed(inv).height
                  for _, leg in legs] or [0.0])
    # The x label plus its tick numbers, taken from the deepest panel: this is the drop
    # between the axes box and the top of the legend row.
    label_in = 0.0
    for ax in fig.axes:
        if not ax.get_xlabel():
            continue
        axes_y0 = ax.get_window_extent(r).transformed(inv).y0
        lab_y0 = ax.xaxis.get_label().get_window_extent(r).transformed(inv).y0
        label_in = max(label_in, axes_y0 - lab_y0)

    below_in = label_in + gap_in + row_in
    bottom_frac = (bottom or 0.0) + below_in / h
    if wspace is None:
        # The gap holds the next panel's y label and tick numbers -- the keys no longer sit
        # in it -- plus a clear separation, so the two panels and their two one-line keys
        # read as two blocks rather than as one continuous band across the page.
        panel_w = (w - left_in) / max(n_panels, 1)
        wspace = (0.42 + PANEL_GAP_IN) / max(panel_w, 0.1)
    fig.subplots_adjust(left=left_in / w, right=1.0 - 0.04 / w,
                        top=1.0 - top_in / h, bottom=bottom_frac, wspace=wspace)

    # Now that the axes have their final height, put each key a fixed gap under its own
    # panel's x label. bbox_to_anchor is in AXES fraction, so the offset has to be
    # re-expressed against the (new) axes height for each panel.
    fig.canvas.draw()
    for ax, leg in legs:
        b = ax.get_window_extent(r).transformed(inv)
        y = -(label_in + gap_in) / max(b.height, 1e-9)
        # x in AXES fraction, at the left edge of the y-axis LABEL rather than at the spine:
        # the label is the leftmost thing the panel owns, so the key lines up with it, and
        # the ~0.46 in it gains is what keeps a six-entry key inside the page.
        x = (_ylabel_x0(ax, r, inv) - b.x0) / max(b.width, 1e-9)
        _key_anchor[id(ax)] = (x, y)
        leg.set_bbox_to_anchor((x, y), transform=ax.transAxes)

    _fit_curve_legend_row(fig, legs, n_panels, left_in, w, r, inv)
    return wspace


def _fit_curve_legend_row(fig, legs, n_panels, left_in, fig_w, renderer, inv):
    """
    Keep each one-line key inside the page, growing rightwards from its panel's left edge.

    A key is as wide as its entries, so a six-level one (the sister sweep's harmonic phi,
    whose labels run to 0.0667) is wider than its panel and runs into the next panel's key
    or off the page -- and bbox_inches="tight" then EXPANDS the saved figure to contain it,
    silently defeating the 6.5 in cap.

    The key starts at its panel's y-axis label and grows right, so the room it has is the
    distance from there to the NEXT panel's y-axis label (or, for the last panel, the page
    edge). If it does not fit, rebuild it smaller, down to but never past ANNOT_PT --
    shrinking below the manuscript's smallest size to fit a key would trade one rule for
    another. Rebuild rather than restyle: handlelength and columnspacing are resolved into
    points when the legend is created, so setting the texts smaller afterwards leaves the
    box essentially the width it was.

    Then the opposite problem: a key whose labels are short ends well before its panel does,
    and reads as belonging to the panel's left half. _stretch_curve_legend spreads the
    leftover width into the column gaps so the row spans the panel.
    """
    starts = sorted(_ylabel_x0(ax, renderer, inv) for ax, _ in legs)
    shrunk = False
    for ax, leg in legs:
        x0 = _ylabel_x0(ax, renderer, inv)
        rights = [other for other in starts if other > x0 + 1e-6]
        budget = (min(rights) if rights else fig_w - PAGE_MARGIN_IN) - x0
        spec = getattr(leg, "_curve_key_spec", None)
        if spec is None:
            continue
        anchor = _key_anchor.get(id(ax), (0.0, _CURVE_LEGEND_Y))
        fs = plt.rcParams["legend.fontsize"]
        for _ in range(6):
            # Re-fetch the renderer each pass: the redraw below invalidates the one the
            # caller measured with, and a stale renderer reports the OLD size.
            width = leg.get_window_extent(fig.canvas.get_renderer()).transformed(inv).width
            if width <= budget or fs <= ANNOT_PT:
                break
            fs = max(ANNOT_PT, fs * budget / width)
            leg = _place_curve_legend(ax, spec[0], spec[1], fs)
            leg.set_bbox_to_anchor(anchor, transform=ax.transAxes)
            fig.canvas.draw()
            shrunk = True

        # Then spread the row out to fill the panel. The target is from where the key
        # actually starts (its y label) to the panel's RIGHT spine, so the last entry lines
        # up with the end of the axis the key describes -- never past `budget`, which is
        # what keeps it clear of the next panel and the page edge.
        target = min(budget, ax.get_window_extent(
            fig.canvas.get_renderer()).transformed(inv).x1 - x0)
        leg = _stretch_curve_legend(ax, leg, target, fig, inv)
        leg.set_bbox_to_anchor(anchor, transform=ax.transAxes)
        shrunk = True
    if shrunk:
        fig.canvas.draw()


def reserve_side_legend(fig, n_panels=2, side_legend_in=None, bottom=None,
                        wspace=None, left_in=None, top_in=None, pad_in=0.22):
    """
    Lay a sweep panel row out. Kept as the name every sweep figure calls.

    The curve keys used to sit to the RIGHT of each panel, and this reserved their width.
    They now sit under their own panel on a single line (see _curve_legend), so there is
    nothing to reserve at the side and this delegates to reserve_curve_legend_row.
    side_legend_in and pad_in are accepted and ignored so no caller needs an edit.
    """
    return reserve_curve_legend_row(fig, n_panels=n_panels, bottom=bottom, wspace=wspace,
                                    left_in=left_in, top_in=top_in)


def encoding_legend(ax, target=DEFAULT_TARGET, loc="upper left", force_loc=False,
                    show_theory_crossing=True, show_band=True,
                    show_eig_threshold=True):
    """
    Key for what the marks mean, as a second legend inside one panel's axes.

    Kept for callers that want the key in-axes; the sweep figures now put it under the row
    with figure_encoding_legend instead. show_eig_threshold is accepted and ignored: the
    hollow eigenvalue-threshold marker it used to key is no longer drawn.
    """
    _add_second_legend(ax, encoding_handles(target, show_theory_crossing, show_band),
                       loc, force_loc=force_loc)


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
