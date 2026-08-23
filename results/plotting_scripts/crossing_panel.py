#%%
"""
Draw one overlap-vs-swept-parameter panel from per-replicate measurements.

The shared panel behind the overlap figures. Each curve of the sweep (a value of N, phi, or
n-per-deme -- the *curve parameter*) contributes a cloud of per-replicate cos^2 values
across the *swept parameter* on the x axis (Fst or 1/M), and is drawn as three things:

    1. a solid regression line with a shaded confidence band, fitted through every
       replicate rather than through their medians (crossing_fit.fit_with_band)
    2. a thin solid vertical line at the swept value where that fit reaches the target
       cos^2 (default 0.9) -- the measured detection point
    3. a dashed vertical line at the same crossing predicted by theory

Replacing the old "median line + prediction line at every x" pair: the comparison between
measurement and theory is now a comparison between two vertical lines, which is what the
detection claim actually is, and the band shows how well the crossing is resolved.

The theoretical crossing is supplied by the caller as a function of the swept value,
because nu(x) differs per demography; crossing_fit.theory_crossing inverts it numerically
against src/threshold_and_overlap.

Colour encodes the curve parameter through a colourbar rather than a legend, since ten
curves would otherwise need ten legend entries per panel.

Crossings are recomputed here so a figure can be redrawn from the replicate file alone;
they match the *_crossings.csv the analysis writes, which is the place to read them as
numbers.
"""
import os
import sys

import numpy as np
from matplotlib import cm, colors

# Appended, not inserted: eleven analysis scripts share a filename with their plotting
# counterpart (SingleSplitSweeps.py, SisterPCAlignment.py, ...), so putting src/analysis
# ahead of this directory would make a plotting script importing a sibling by name get the
# analysis module instead.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src", "analysis"))
from crossing_fit import DEFAULT_TARGET, fit_with_band, theory_crossing

POINT_KW = dict(s=4, alpha=0.18, linewidths=0, zorder=1)
# The band is frequently narrower than the line itself at 10-20 replicates, so it needs
# enough alpha to register as more than a smudge where it is thin.
BAND_KW = dict(alpha=0.30, linewidth=0, zorder=2)
FIT_KW = dict(linewidth=1.6, zorder=3)
EMP_VLINE_KW = dict(linewidth=1.0, zorder=4)
THEORY_VLINE_KW = dict(linewidth=1.3, linestyle=(0, (4, 2)), zorder=4)


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
                                levels=None, norm=None, log_levels=False,
                                show_points=True, show_theory_curve=False):
    """
    Draw one panel from the pre-fitted summary files. Fits nothing.

    The stage-3 summariser (src/sweeps_analysis/summarise.py) already did the Richards fit,
    the bootstrap band and the theory inversion, so this function only renders columns.
    That is the whole point of the split: the figure no longer depends on the theory
    module, on crossing_fit, or on the per-replicate cloud being reloaded.

    Args:
        curves: rows of curves.csv (x, y_fit, y_lo, y_hi, y_theory, fit_kind).
        crossings: rows of crossings.csv (crossing, theory_crossing_pred).
        points: rows of points.csv for the scatter, or None to omit it.
        sweep: filter all three frames to this sweep name first, when they hold several.
        show_theory_curve: also draw the continuous theoretical curve, not just its crossing.

    Returns the per-curve crossing rows as a list of dicts, matching crossing_panel's.
    """
    if sweep is not None:
        curves = curves[curves["sweep"] == sweep]
        crossings = crossings[crossings["sweep"] == sweep]
        if points is not None:
            points = points[points["sweep"] == sweep]

    if levels is None:
        levels = np.sort(curves["level"].unique())
    if norm is None:
        norm = (colors.LogNorm(vmin=levels.min(), vmax=levels.max()) if log_levels
                else colors.Normalize(vmin=levels.min(), vmax=levels.max()))
    cmap = cm.viridis if cmap is None else cmap

    out = []
    for level in levels:
        c = curves[curves["level"] == level].sort_values("x")
        if c.empty:
            continue
        color = cmap(norm(level))

        if show_points and points is not None:
            pt = points[points["level"] == level]
            ax.scatter(pt["x"], pt["cos2"], color=color, **POINT_KW)

        ax.fill_between(c["x"], c["y_lo"], c["y_hi"], color=color, **BAND_KW)
        ax.plot(c["x"], c["y_fit"], "-", color=color, **FIT_KW)
        if show_theory_curve and c["y_theory"].notna().any():
            ax.plot(c["x"], c["y_theory"], ":", color=color, linewidth=1.0, zorder=3)

        row = crossings[crossings["level"] == level]
        crossing = float(row["crossing"].iloc[0]) if len(row) else np.nan
        theory = float(row["theory_crossing_pred"].iloc[0]) if len(row) else np.nan
        if np.isfinite(crossing):
            ax.axvline(crossing, color=color, **EMP_VLINE_KW)
            # Drawn only where the fit also reaches the target: a lone dashed line reads as
            # a detection point the data never supports. The number is in the table either
            # way. Same rule as crossing_panel.
            if np.isfinite(theory):
                ax.axvline(theory, color=color, **THEORY_VLINE_KW)

        out.append(dict(level=level, crossing=crossing,
                        crossing_lo=float(row["crossing_lo"].iloc[0]) if len(row) else np.nan,
                        crossing_hi=float(row["crossing_hi"].iloc[0]) if len(row) else np.nan,
                        theory=theory))

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


def encoding_legend(ax, target=DEFAULT_TARGET, loc="upper left"):
    """The shared key for what the three line styles mean, drawn in neutral grey."""
    ax.plot([], [], "-", color="0.35", linewidth=1.6, label="fit to replicates")
    ax.fill_between([], [], [], color="0.35", alpha=0.30, linewidth=0, label="68% band")
    ax.plot([], [], "-", color="0.35", linewidth=1.0,
            label=rf"measured $\langle\hat u,u\rangle^2={target:g}$")
    ax.plot([], [], color="0.35", **{**THEORY_VLINE_KW, "zorder": 5}, label="theory")
    ax.legend(loc=loc, fontsize=8, frameon=False)


def crossing_table(results, level_name, x_name):
    """Format the per-level crossings as text, for printing beside the figure."""
    lines = [f"{level_name:>10}  {'measured':>12}  {'68% CI':>26}  {'theory':>12}  ratio"]
    for r in results:
        ci = (f"[{r['crossing_lo']:.3g}, {r['crossing_hi']:.3g}]"
              if np.isfinite(r["crossing_lo"]) else "-")
        ratio = (r["crossing"] / r["theory"]
                 if np.isfinite(r["crossing"]) and np.isfinite(r["theory"]) and r["theory"]
                 else np.nan)
        lines.append(f"{r['level']:>10.4g}  {r['crossing']:>12.4g}  {ci:>26}  "
                     f"{r['theory']:>12.4g}  {ratio:.3f}")
    return f"{x_name} at target overlap\n" + "\n".join(lines)

# %%
