#%%
"""
Spatial-axis recovery vs migration under sparse deme sampling, with the maximum realizable
similarity drawn as a ceiling on each curve.

This is PhaseTransitionRandomSS with one addition. Sampling a fraction phi of the demes does
not merely add noise -- it changes which axis the theoretical covariance HAS, so even a
perfect estimate cannot reach the full-lattice spatial axis that stands in for geographic
position. Each curve therefore gets a horizontal DOTTED line at

    ceiling(phi) = mean over random deme configurations of <u_phi, u_1>^2

in its own colour: the value the measured curve can approach but not exceed. At phi = 1 the
ceiling is exactly 1 and the line sits on the top of the panel, which is the right degenerate
case rather than a special case.

A  1D stepping stone, K=25, total N fixed at 200
B  2D stepping stone, K=9 grid, total N fixed at 600

The measured curves, their bands and their markers come from the same pre-fitted summary
files PhaseTransitionRandomSS reads; the ceilings come from
src/analysis/SparseSamplingCeiling.py. This script draws only -- it fits nothing and
computes no theory.

Reads results/sweep_summaries/stepping_stone_random/{curves,crossings,points}.csv
      results/analysis/SparseSamplingCeiling/ceiling_mean.csv
Writes results/figures/SparseSamplingCeiling.pdf
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from crossing_panel import (DEFAULT_TARGET, LEGEND_PT, PANEL_LETTER_SIZE,
                            crossing_panel_from_summary, crossing_table, encoding_handles,
                            encoding_bottom_margin, figsize, figure_encoding_legend,
                            L_title, level_colors, load_summary, reserve_side_legend,
                            save_figure, style_swept_axis, sweep_N_total, variant_note)
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries",
                           "stepping_stone_random")
CEILING_PATH = os.path.join(_PROJECT_ROOT, "results", "analysis", "SparseSamplingCeiling",
                            "ceiling_mean.csv")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP = cm.plasma
TARGET = DEFAULT_TARGET
# (sweep name, panel label, the dimension its ceilings are stored under)
SWEEPS = [("ss1d_vary_deme_fraction", "1D", 1), ("ss2d_vary_deme_fraction", "2D", 2)]

# Figure fraction reserved for the shared curve key, and where it sits -- just above the
# encoding key, which figure_encoding_legend places at the very bottom.
CURVE_KEY_HEIGHT = 0.07
# Gaps (figure fraction): x labels -> curve key, and curve key -> encoding key.
CURVE_KEY_GAP = 0.02
ENCODING_KEY_GAP = 0.025

# Dotted, thin, and under the curves: the ceiling is a reference level, not a measurement.
CEILING_KW = dict(linestyle=(0, (1.2, 1.8)), linewidth=0.9, zorder=2.5)


def _n_prefix(sweep, label):
    """`label` with the sweep's fixed total N, the quantity this design holds constant."""
    n = sweep_N_total(sweep)
    return label if n is None else rf"{label}  ($N={n}$)"


def draw_ceilings(ax, ceilings, dimension, levels, cmap=CMAP):
    """One horizontal dotted line per phi, at that phi's mean realizable similarity."""
    c = ceilings[ceilings["dimension"] == dimension]
    colors = level_colors(levels, cmap)
    drawn = {}
    for level, color in zip(levels, colors):
        row = c[c["phi"] == level]
        if row.empty:
            continue
        # mean, matching what the drawn CURVE converges to. Replicates are never collapsed
        # before fitting (CLAUDE.md), so the least-squares fit through all 16 replicate
        # points lands on their MEAN, not their median -- verified: 1D phi=0.2 fit_max
        # 0.8917 against a replicate mean of 0.8906 and a median of 0.9447. The ceiling is
        # drawn over that fitted curve, so it has to be the same statistic; a median
        # ceiling sat 0.055 above the curve it was meant to bound.
        y = float(row["mean"].iloc[0])
        ax.axhline(y, color=color, **CEILING_KW)
        drawn[level] = y
    return drawn


def _centre_encoding_columns(fig, legs, axes):
    """
    Re-anchor each encoding-key column so it is centred under its own panel.

    figure_encoding_legend starts each column at that panel's own curve key
    (_curve_key_starts). This figure suppresses those per-panel keys in favour of one shared
    key, so that lookup finds nothing and falls back to x = 0 and 0.5 -- both columns hard
    against the left of their half of the page. Centring on the panel's axes box instead is
    what the fallback was standing in for.
    """
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    r = fig.canvas.get_renderer()
    for leg, ax in zip(legs, axes):
        box = ax.get_position()
        w = inv.transform_bbox(leg.get_window_extent(r)).width
        leg.set_bbox_to_anchor((box.x0 + box.width / 2 - w / 2,
                                leg.get_bbox_to_anchor().transformed(inv).y0),
                               transform=fig.transFigure)


def make_figure(summary_dir=SUMMARY_DIR, ceiling_path=CEILING_PATH, figure_dir=FIGURE_DIR,
                target=TARGET, standardized=False, name="SparseSamplingCeiling"):
    setup_plotting_style()
    curves, crossings, points, theory = load_summary(summary_dir)
    ceilings = pd.read_csv(ceiling_path)

    # The eigenvalue threshold marker is off here: K ~ N in the stepping stones, so many
    # structure axes are detached at once and the single-spike fit behind it does not apply.
    fig, axes = plt.subplots(1, 2, figsize=figsize(6.5, 4.35))
    results, all_ceilings = [], {}
    for ax, (sweep, label, dimension) in zip(axes, SWEEPS):
        res = crossing_panel_from_summary(
            ax, curves, crossings, points, sweep=sweep, cmap=CMAP,
            cbar_label="fraction of\ndemes sampled", target=target,
            show_theory_crossing=False, theory_table=theory,
            standardized=standardized, show_eig_threshold=False,
            # One shared key for the whole figure instead: both panels sweep the SAME phi
            # values, so a per-panel key printed the identical five entries twice.
            show_legend=False)
        levels = sorted(curves[curves["sweep"] == sweep]["level"].unique())
        all_ceilings[label] = draw_ceilings(ax, ceilings, dimension, levels)
        style_swept_axis(ax, r"$1/M$")
        # Name the quantity explicitly. crossing_panel's shared default is the generic
        # <u_hat, u>^2; here both sides matter -- the observed PC of the SPARSE sample
        # against the FULL-lattice axis -- and the ceiling only reads correctly once the
        # axis says which two vectors are being compared.
        ax.set_ylabel(r"cosine similarity  "
                      r"$\langle \hat{u}_\phi, u_1\rangle^2$")
        ax.set_title(L_title(sweep, _n_prefix(sweep, label)))
        results.append(res)

    for ax in axes:
        variant_note(ax, standardized)

    for ax, letter in zip(axes, "AB"):
        # Offset in POINTS, not axes fraction: at 12 pt the fractional offset put the letter
        # on the "using L_eff" row and the panel title.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -34 / 72, 26 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="top")

    # The shared encoding key, plus one entry naming the ceiling -- it is the only mark on
    # the panel the standard key does not already describe.
    # encoding_handles returns artists carrying their own `label`, so the added entries are
    # built the same way rather than as (handle, label) pairs.
    handles = encoding_handles(target=target, show_theory_crossing=False)
    # The shared key names the crossing generically; spell out the same two vectors the y
    # axis now names, so the marker and the axis agree.
    for h in handles:
        lab = h.get_label()
        if "crossing point" in lab:
            h.set_label(rf"$\langle \hat{{u}}_\phi, u_1\rangle^2={target:g}$ "
                        r"crossing point (I-spline fit)")
    # The level itself, keyed before the ceiling: the two horizontal references are read
    # together, and the denser dots match how axhline draws it inside the panel. "line"
    # against the marker's "crossing point (I-spline fit)" is what separates the horizontal
    # LINE at 0.8 from the POINT where each curve reaches it.
    handles.append(Line2D([], [], color="0.6", linewidth=0.6, linestyle=":",
                          label=rf"$\langle \hat{{u}}_\phi, u_1\rangle^2={target:g}$ "
                                r"line"))
    handles.append(Line2D([], [], color="0.35",
                          label=r"maximum realizable $\langle u_\phi, u_1\rangle^2$",
                          **CEILING_KW))
    # Room for TWO stacked keys under the panels, read top-down in the order they are
    # needed: WHICH CURVE IS WHICH first, directly under the axes, then what the marks on
    # them mean below that.
    bottom = encoding_bottom_margin(handles, fig=fig) + CURVE_KEY_HEIGHT
    reserve_side_legend(fig, n_panels=2, bottom=bottom)

    # The single curve key, centred across both panels and sitting just under them. Its y
    # is MEASURED from the lowest artist hanging off the axes (the x labels), not set as a
    # constant: a constant left a band of dead page between the panels and the key.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    r = fig.canvas.get_renderer()
    lows = [inv.transform_bbox(ax.xaxis.get_label().get_window_extent(r)).y0
            for ax in fig.axes if ax.get_xlabel()]
    curve_y = (min(lows) if lows else fig.subplotpars.bottom) - CURVE_KEY_GAP

    c_handles, c_labels = axes[0].get_legend_handles_labels()
    enc_y = curve_y
    if c_handles:
        spacer = Line2D([], [], linestyle="none", marker="none")
        leg = fig.legend([spacer] + c_handles, ["fraction of demes sampled:"] + c_labels,
                         loc="upper center", bbox_to_anchor=(0.5, curve_y),
                         ncol=len(c_handles) + 1, fontsize=LEGEND_PT, frameon=False,
                         columnspacing=0.9, handlelength=1.4, handletextpad=0.5,
                         borderpad=0.0)
        # Measure where the curve key actually ends and start the encoding key below THAT.
        # figure_encoding_legend's own y=None cannot do this: it measures artists attached
        # to the AXES, and this key is figure-level, so it was invisible to that scan and
        # both rows were drawn at the same height, on top of each other.
        fig.canvas.draw()
        enc_y = inv.transform_bbox(leg.get_window_extent(r)).y0 - ENCODING_KEY_GAP

    # Its two columns are placed under the two panels explicitly: _curve_key_starts aligns
    # them to the per-panel curve keys, and this figure has none (one shared key instead),
    # so its fallback would put them at x = 0 and 0.5, hard against the left edge.
    legs = figure_encoding_legend(fig, handles, y=enc_y)
    _centre_encoding_columns(fig, legs, axes)

    out = save_figure(fig, figure_dir, name, standardized)
    print(f"wrote {out}")
    for (_, label, _), res in zip(SWEEPS, results):
        print(f"\n{label}")
        print(crossing_table(res, "phi", "1/M"))
        print("  max realizable similarity by deme fraction:")
        for level, y in sorted(all_ceilings[label].items()):
            print(f"    phi={level:g}: {y:.4f}")
    return fig


def make_all(summary_dir=SUMMARY_DIR, ceiling_path=CEILING_PATH, figure_dir=FIGURE_DIR,
             target=TARGET):
    """Both versions: the ordinary GRM and the standardized one, same ceilings."""
    return [make_figure(summary_dir, ceiling_path, figure_dir, target, standardized=s)
            for s in (False, True)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary-dir", default=SUMMARY_DIR)
    p.add_argument("--ceiling", default=CEILING_PATH)
    p.add_argument("--figure-dir", default=FIGURE_DIR)
    p.add_argument("--name", default="SparseSamplingCeiling")
    p.add_argument("--standardized", action="store_true")
    args = p.parse_args()
    make_figure(args.summary_dir, args.ceiling, args.figure_dir,
                standardized=args.standardized, name=args.name)


if __name__ == "__main__":
    main()

# %%
