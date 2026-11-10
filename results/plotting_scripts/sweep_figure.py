"""Layout for a pair of cosine-similarity-vs-swept-parameter panels at GENETICS print size.

Two panels side by side sharing the y axis, each with its curve key underneath, and one
shared encoding key (theory lines, L_eff ticks, fitted marks) across the bottom. Drawing of
the marks themselves is crossing_panel.crossing_panel_from_summary / peak_panel_from_summary.
"""
import numpy as np
from matplotlib.lines import Line2D

import genetics_style as gs
from crossing_panel import (DEFAULT_TARGET, THEORY_PICKUP_VLINE_KW, THEORY_VLINE_KW,
                            TICK_KW, _ci_handle, crossing_panel_from_summary,
                            peak_panel_from_summary)

LEFT, GAP, RIGHT = 9.0, 3.0, 1.5      # mm: y-label column, between panels, right edge
TOP = 5.0                             # mm above the axes: L_eff tick row and panel letter
XAXIS_H = 9.0                         # mm below the axes: tick labels, x label, gap
LINE_H = 2.75                         # mm per legend line at 6 pt
YLABEL = r"cosine similarity $\langle \hat u, u\rangle^2$"


def curve_key_h():
    return LINE_H * 2 + 1.0          # title line + one row of entries


def encoding_key_h(show_theory, n_extra=0):
    n = (6 if show_theory else 2) + n_extra
    return (LINE_H + 0.3) * int(np.ceil(n / 2)) + 1.0


def short_level(label):
    """Two significant figures for non-integer levels (1/15 -> 0.067)."""
    v = float(label)
    return label if v == int(v) else f"{v:.2g}"


def encoding_handles(target=DEFAULT_TARGET, show_theory=True, cross=None):
    """The shared key: threshold column, then target-crossing column (column-major fill)."""
    cross = cross or rf"$\langle\hat u,u\rangle^2={target:g}$"
    thr_fit = _ci_handle("s", 2.6, "threshold (I-spline fit)")
    cross_fit = _ci_handle("o", 2.6, f"{cross} (I-spline fit)")
    if not show_theory:
        return [thr_fit, cross_fit]
    grey = "0.35"
    return [
        Line2D([], [], color=grey, label="threshold (close approx.)", **THEORY_PICKUP_VLINE_KW),
        Line2D([], [], color=grey, label="threshold (crude approx.)", **TICK_KW["pickup"]),
        thr_fit,
        Line2D([], [], color=grey, label=f"{cross} (close approx.)", **THEORY_VLINE_KW),
        Line2D([], [], color=grey, label=f"{cross} (crude approx.)", **TICK_KW["cross"]),
        cross_fit,
    ]


def pair_height(n_levels, width_mm, ax_h, show_theory=True, n_extra=0):
    """Height (mm) of the pair block, from the top of the panel letters to the bottom key."""
    return TOP + ax_h + XAXIS_H + curve_key_h() + encoding_key_h(show_theory, n_extra)


def draw_pair(fig, panels, x_label, width_mm, ax_h, top_mm, letters="ab",
              show_theory=True, target=DEFAULT_TARGET, peak=False,
              ylabel=YLABEL, cross_label=None, extra_handles=(), panel_hook=None):
    """
    Draw two panels whose letter row starts `top_mm` below the figure top.

    panels: two dicts with keys curves, crossings, points, theory, sweep, title (curve-key
    title) and optionally cmap. Returns (axes, per-panel crossing rows).
    """
    H = fig.get_size_inches()[1] / gs.MM
    w = (width_mm - LEFT - GAP - RIGHT) / 2
    bottom = H - top_mm - TOP - ax_h
    axes = [gs.axes_mm(fig, LEFT, bottom, w, ax_h),
            gs.axes_mm(fig, LEFT + w + GAP, bottom, w, ax_h)]

    results = []
    for ax, p in zip(axes, panels):
        draw = peak_panel_from_summary if peak else crossing_panel_from_summary
        kw = {} if peak else dict(theory_table=p.get("theory"),
                                  show_theory_crossing=show_theory)
        res = draw(ax, p["curves"], p["crossings"], p["points"], sweep=p["sweep"],
                   cmap=p.get("cmap", gs.cm.viridis), target=target, show_legend=False, **kw)
        results.append(res)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        if panel_hook is not None:
            panel_hook(ax, p)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    entries = [(h, [short_level(x) for x in l])
               for h, l in (ax.get_legend_handles_labels() for ax in axes)]
    gs.matched_row_keys(axes, entries, XAXIS_H, [p["title"] for p in panels])
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)

    for ax, letter, dx in zip(axes, letters, (LEFT, GAP)):
        x0 = ax.get_position().x0 * width_mm
        gs.label_panel_mm(fig, letter, x0 - dx + 0.3, H - top_mm)

    # Shared mark key: first half (threshold marks) under a, second half under b
    if not peak:
        handles = encoding_handles(target, show_theory, cross_label) + list(extra_handles)
        half = int(np.ceil(len(handles) / 2))
        centers = [(a.get_position().x0 + a.get_position().width / 2) * width_mm for a in axes]
        gs.grid_key(fig, [handles[:half], handles[half:]], centers,
                    bottom - XAXIS_H - curve_key_h(), pitch_mm=LINE_H + 0.3)
    return axes, results
