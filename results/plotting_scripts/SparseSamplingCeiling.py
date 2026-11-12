"""
Supplementary figure: spatial-axis recovery under sparse random deme sampling, with the
maximum realizable similarity drawn as a ceiling on each curve.

a  1D stepping stone, K = 25, total N fixed at 200; fraction of demes sampled 0.2-1.
b  2D stepping stone, 9 x 9, total N fixed at 600.
Each curve's dotted horizontal line is ceiling(phi) = mean over random deme configurations
of <u_phi, u_1>^2: the value even a perfect estimate of the sparse sample's own axis reaches
against the full-lattice axis (exactly 1 at phi = 1). The mean, not the median, matches the
fitted curve, which is a least-squares fit through every replicate.

Reads results/sweep_summaries/stepping_stone_random/{curves,crossings,points}.csv
      results/analysis/SparseSamplingCeiling/ceiling_mean.csv
Writes results/figures/supplement/SparseSamplingCeiling.pdf
"""
import os

import pandas as pd
from matplotlib.lines import Line2D

import genetics_style as gs
from crossing_panel import DEFAULT_TARGET, crossing_table, level_colors, load_summary
from sweep_figure import draw_pair, pair_height

SUMMARY_DIR = os.path.join(gs._ROOT, "results", "sweep_summaries", "stepping_stone_random")
CEILING_PATH = os.path.join(gs._ROOT, "results", "analysis", "SparseSamplingCeiling",
                            "ceiling_mean.csv")
WIDTH, AX_H = 130.0, 50.0
# (sweep, label, dimension the ceilings are stored under)
SWEEPS = [("ss1d_vary_deme_fraction", "1D", 1), ("ss2d_vary_deme_fraction", "2D", 2)]
CEILING_KW = dict(linestyle=(0, (1.2, 1.8)), linewidth=0.8, zorder=2.5)


def make_figure(summary_dir=SUMMARY_DIR, ceiling_path=CEILING_PATH, target=DEFAULT_TARGET):
    gs.setup()
    curves, crossings, points, theory = load_summary(summary_dir)
    ceilings = pd.read_csv(ceiling_path)
    drawn = {}

    def draw_ceilings(ax, p):
        c = ceilings[ceilings["dimension"] == p["dimension"]]
        levels = sorted(curves[curves["sweep"] == p["sweep"]]["level"].unique())
        for level, color in zip(levels, level_colors(levels, gs.cm.viridis)):
            row = c[c["phi"] == level]
            if len(row):
                y = float(row["mean"].iloc[0])
                ax.axhline(y, color=color, **CEILING_KW)
                drawn.setdefault(p["label"], {})[level] = y

    panels = [dict(curves=curves, crossings=crossings, points=points, theory=theory,
                   sweep=sweep, label=label, dimension=dim,
                   title="fraction of demes sampled") for sweep, label, dim in SWEEPS]
    pair = r"\langle \hat{u}_\phi, u_1\rangle^2"
    extra = [Line2D([], [], color="0.6", linewidth=0.6, linestyle=":",
                    label=rf"${pair}={target:g}$ line"),
             Line2D([], [], color="0.35", label=r"maximum realizable $\langle u_\phi, u_1\rangle^2$",
                    **CEILING_KW)]

    fig = gs.blank_figure(WIDTH, pair_height(5, WIDTH, AX_H, show_theory=False, n_extra=2))
    _, results = draw_pair(fig, panels, r"$1/M$", WIDTH, AX_H, top_mm=0, show_theory=False,
                           target=target, ylabel=rf"squared cosine similarity ${pair}$",
                           cross_label=rf"${pair}={target:g}$", extra_handles=extra,
                           panel_hook=draw_ceilings)
    print(f"wrote {gs.save(fig, 'SparseSamplingCeiling', 'supplement')}")
    for (_, label, _), res in zip(SWEEPS, results):
        print(f"\n{label}\n{crossing_table(res, 'phi', '1/M')}")
        for level, y in sorted(drawn.get(label, {}).items()):
            print(f"    ceiling phi={level:g}: {y:.4f}")
    return fig


if __name__ == "__main__":
    make_figure()
