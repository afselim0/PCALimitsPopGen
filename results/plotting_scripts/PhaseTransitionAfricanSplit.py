#%%
"""
PC alignment with the recently-split sister axis (YRI1/YRI2), as a detection threshold in Fst.

A  cosine similarity vs Fst at nine sample sizes N, each sister held at 20% of N (H = 0.20).
B  cosine similarity vs Fst at ten sister sampling designs, N = 1000 throughout. Each design
   takes a pair (phi_1, phi_2) from {0.05, 0.10, 0.15, 0.20} -- every unordered pair,
   phi_1 = phi_2 included -- and gives YRI1 and YRI2 those fractions of N, with CEU and CHB
   splitting the rest. H is the harmonic mean of the two fractions, which is dominated by
   the smaller one: (0.05, 0.20) gives H = 0.08, much closer to (0.05, 0.05)'s 0.05 than to
   (0.20, 0.20)'s 0.20 despite four times the samples in the larger sister. That is the
   point of the panel.

Panel A holds H at 0.20 rather than 0.05: at 0.05 the exact nu topped out at 1.67 across
the whole simulated Fst range, so every curve plateaued near cosine similarity 0.78 and never
reached the 0.9 target -- the sweep had no crossing to report.

The fit (flat below the fitted transition, monotone increasing above), its 95% band and the theoretical
crossing are computed upstream by src/sweeps_analysis/summarise.py and read here as
columns; this script draws only. A filled marker sits on the curve at the target, a square on the
x axis marks the fitted detection threshold, and dashed / dotted vertical lines mark the
theoretical positions. The theory uses the exact
heteroskedastic route (eps is genuinely nonzero on this demography, tau/tau_bar spans
0.81-1.17), anchored on the exact nu recorded per cell.

Reads results/sweep_summaries/african_split/{curves,crossings,points}.csv
    (sweeps african_split_vary_N and african_split_vary_harmonic_phi)
Writes results/figures/PhaseTransitionAfricanSplit.pdf
"""
import os

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import pandas as pd
from matplotlib import cm

from crossing_panel import (reserve_side_legend, PANEL_LETTER_SIZE, figsize, DEFAULT_TARGET, L_title, crossing_panel_from_summary,
                            crossing_table, encoding_handles, figure_encoding_legend,
                            encoding_bottom_margin, load_summary, save_figure,
                            style_swept_axis, variant_note)
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "african_split")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis
CMAP_H = cm.plasma
TARGET = DEFAULT_TARGET


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET,
                standardized=False):
    setup_plotting_style()
    curves, crossings, points, theory = load_summary(summary_dir)

    # No eigenvalue threshold marker. Not because the edge is wrong -- rank K-1 = the 4th
    # highest eigenvalue is a clean edge here (sub-threshold lambda/lambda_bulk = 1.009,
    # as good as the split's 1.008) -- but because converting the fitted nu_c into an Fst
    # is ill-conditioned for this axis. nu = eps_0 + s*Fst carries a pedestal eps_0 = 0.215
    # that is 31-53% of nu_c, so d log x_c / d log nu_c runs 1.5-2.2 and an 8-13% error in
    # nu_c becomes a 26-54% error in Fst. In nu space the estimate is 0.79 +/- 0.02; in Fst
    # space 0.67 +/- 0.09. nu_c_emp is in crossings.csv for anyone who wants it.
    fig, axes = plt.subplots(1, 2, figsize=figsize(6.5, 4.35))

    res_N = crossing_panel_from_summary(
        axes[0], curves, crossings, points, sweep="african_split_vary_N",
        cmap=CMAP_N, cbar_label=r"$N$", target=target, theory_table=theory, standardized=standardized,
        show_eig_threshold=False)
    res_H = crossing_panel_from_summary(
        axes[1], curves, crossings, points, sweep="african_split_vary_harmonic_phi",
        cmap=CMAP_H, cbar_label=r"$\tilde{\phi}$", target=target, theory_table=theory, standardized=standardized,
        show_eig_threshold=False)

    # Only the x label is set here: the y label is the panel's own (cosine similarity),
    # set by crossing_panel_from_summary.
    for ax in axes:
        style_swept_axis(ax, r"$F_{ST}$")
    axes[0].set_title(L_title("african_split_vary_N",
                              r"varying $N$  ($\tilde{\phi}=0.20$)"))
    axes[1].set_title(L_title("african_split_vary_harmonic_phi",
                              r"varying $\phi_1,\phi_2$  ($N=1000$)"))

    for ax in axes:
        variant_note(ax, standardized)

    for ax, letter in zip(axes, "AB"):
        # Offset in POINTS, not axes fraction: at 12 pt the fractional offset put the
        # letter on the "using L_eff" row and the panel title.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -34 / 72, 26 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="top")

    # tight_layout knows nothing about the figure-level key, so the bottom margin is
    # opened for it after the fact; wspace holds panel B's y-axis label clear of A.
    handles = encoding_handles(target=target)
    # reserve_side_legend, not a bare wspace: the per-panel keys are anchored OUTSIDE
    # the axes, so tight_layout cannot see them and lays the axes across the full
    # width -- the key then lands on the neighbouring panel. This reserves the
    # legend's width in inches so each panel keeps its own square.
    reserve_side_legend(fig, n_panels=2, bottom=encoding_bottom_margin(handles, fig=fig))
    figure_encoding_legend(fig, handles)
    out = save_figure(fig, figure_dir, "PhaseTransitionAfricanSplit", standardized)
    print(f"wrote {out}")
    print(crossing_table(res_N, "N", "Fst"))
    print(crossing_table(res_H, "H", "Fst"))
    return fig


def make_all(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    """Both versions: the ordinary GRM and the standardized one, same theory marks."""
    return [make_figure(summary_dir, figure_dir, target, standardized=s)
            for s in (False, True)]


if __name__ == "__main__":
    make_all()

# %%
