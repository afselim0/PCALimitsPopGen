#%%
"""
Draw one overlap-vs-swept-parameter panel from the pre-fitted summary files.

The shared panel behind the overlap figures. Each curve of the sweep (a value of N, phi or
n-per-deme -- the *curve parameter*) is drawn across the *swept parameter* on the x axis
(Fst or 1/M) as:

    1. the shape-constrained scam fit through every replicate, with its 95% confidence band
    2. a filled marker sitting on the curve where it reaches the target cos^2 (0.9) -- the
       measured detection point, on the line rather than beside it
    3. a dashed vertical line at the same crossing predicted by theory

The comparison between measurement and theory is the horizontal gap between the marker and
the dashed line, which is what the detection claim actually is.

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
THEORY_VLINE_KW = dict(linewidth=1.3, linestyle=(0, (4, 2)), zorder=4)

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


def crossing_panel_from_summary(ax, curves, crossings, points=None, sweep=None,
                                cmap=None, cbar_label="", target=DEFAULT_TARGET,
                                levels=None, show_points=True, show_theory_curve=False,
                                show_theory_crossing=True,
                                legend_title=None, legend_loc=None,
                                legend_ncol=1, show_legend=True, **_ignored):
    """
    Draw one panel from the pre-fitted summary files. Fits nothing.

    Stage 3 already did the scam fit, its confidence band and the theory inversion, so this
    only renders columns. Each curve gets a discrete colour and a legend entry; the measured
    crossing is a filled marker sitting on the curve at the target level, and the theoretical
    crossing a dashed vertical line in the same colour.

    Args:
        curves: rows of curves.csv (x, y_fit, y_lo, y_hi, y_theory).
        crossings: rows of crossings.csv (crossing, crossing_y, theory_crossing_pred).
        points: rows of points.csv for the scatter, or None to omit it.
        sweep: filter all three frames to this sweep name first.
        cbar_label: used as the legend title (kept under its old name so the figure scripts
            need no edit).
        show_theory_curve: also draw the continuous theoretical curve, not just its crossing.
        show_theory_crossing: draw the dashed vertical line at the theoretical crossing.
            Off for the panels where the theory is not the point being made; the number is
            still returned and printed in the crossings table either way.

    Returns the per-curve crossing rows as a list of dicts.
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
            ax.scatter(pt["x"], pt["cos2"], color=color, **POINT_KW)

        ax.fill_between(c["x"], c["y_lo"], c["y_hi"], color=color, **BAND_KW)
        ax.plot(c["x"], c["y_fit"], "-", color=color, label=format_level(level), **FIT_KW)
        if show_theory_curve and c["y_theory"].notna().any():
            ax.plot(c["x"], c["y_theory"], ":", color=color, linewidth=1.0, zorder=3)

        row = crossings[crossings["level"] == level]
        crossing = float(row["crossing"].iloc[0]) if len(row) else np.nan
        theory = float(row["theory_crossing_pred"].iloc[0]) if len(row) else np.nan
        at_edge = bool(row["crossing_at_edge"].iloc[0]) if len(row) and             "crossing_at_edge" in row else False
        if np.isfinite(crossing):
            # On the curve, at the target: the marker is the crossing, so it cannot drift
            # away from the line the way a separate vertical rule could. A curve that only
            # grazes the target as the sweep ends gets a hollow marker -- its crossing is a
            # lower bound set by where the simulation stopped, not a measured detection
            # point, and it would otherwise read as if it were one.
            if at_edge:
                ax.plot([crossing], [target],
                        **{**CROSS_MARKER_KW, "markerfacecolor": "none",
                           "markeredgecolor": color, "markeredgewidth": 1.4})
            else:
                ax.plot([crossing], [target], color=color, **CROSS_MARKER_KW)
            # Theory drawn only where the fit also reaches the target: a lone dashed line
            # reads as a detection point the data never supports. The number is in the
            # table either way.
            if show_theory_crossing and np.isfinite(theory):
                ax.axvline(theory, color=color, **THEORY_VLINE_KW)

        out.append(dict(level=level, crossing=crossing,
                        crossing_lo=float(row["crossing_lo"].iloc[0]) if len(row) else np.nan,
                        crossing_hi=float(row["crossing_hi"].iloc[0]) if len(row) else np.nan,
                        theory=theory, at_edge=at_edge))

    ax.axhline(target, color="0.6", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(r"squared overlap  $\langle \hat u, u\rangle^2$")
    if show_legend:
        _curve_legend(ax, legend_title if legend_title is not None else cbar_label,
                      legend_loc, ncol=legend_ncol)
    return out


def peak_panel_from_summary(ax, curves, crossings, points=None, sweep=None,
                            cmap=None, cbar_label="", target=DEFAULT_TARGET,
                            levels=None, show_points=True, mark_peak=True,
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
            ax.scatter(pt["x"], pt["cos2"], color=color, **POINT_KW)

        ax.fill_between(c["x"], c["y_lo"], c["y_hi"], color=color, **BAND_KW)
        ax.plot(c["x"], c["y_fit"], "-", color=color, label=format_level(level), **FIT_KW)

        row = crossings[crossings["level"] == level]
        px = float(row["peak_x"].iloc[0]) if len(row) and "peak_x" in row else np.nan
        py = float(row["peak_y"].iloc[0]) if len(row) and "peak_y" in row else np.nan
        if mark_peak and np.isfinite(px) and np.isfinite(py):
            ax.plot([px], [py], color=color, **CROSS_MARKER_KW)

        out.append(dict(level=level, peak_x=px, peak_y=py,
                        peak_x_lo=float(row["peak_x_lo"].iloc[0]) if len(row) else np.nan,
                        peak_x_hi=float(row["peak_x_hi"].iloc[0]) if len(row) else np.nan,
                        peak_y_lo=float(row["peak_y_lo"].iloc[0]) if len(row) else np.nan,
                        peak_y_hi=float(row["peak_y_hi"].iloc[0]) if len(row) else np.nan))

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
                    show_theory_crossing=True, show_band=True):
    """
    Key for what the marks mean on a crossing panel, beside the per-curve legend.

    force_loc keeps `loc` exactly as given instead of auto-placing opposite the curve
    legend -- needed where the auto choice would land the key on the curves themselves.
    show_theory_crossing must match what the panel drew, or the key describes a line that
    is not on it. show_band drops the band swatch where the band is too narrow on the page
    to be worth a key entry.
    """
    handles = [
        Line2D([], [], color="0.35", linewidth=1.7, label="monotonic spline fit"),
    ]
    if show_band:
        handles.append(
            Patch(facecolor="0.35", alpha=0.25, edgecolor="none",
                  label="99% confidence band"))
    handles += [
        Line2D([], [], color="0.35",
               label=rf"measured $\langle\hat u,u\rangle^2={target:g}$",
               **CROSS_MARKER_KW),
    ]
    if show_theory_crossing:
        handles.append(
            Line2D([], [], color="0.35",
                   label=rf"theoretical $\langle\hat u,u\rangle^2={target:g}$",
                   **THEORY_VLINE_KW))
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
