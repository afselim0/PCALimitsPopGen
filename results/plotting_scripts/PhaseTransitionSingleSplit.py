"""
Main Fig. 2: PC1 recovery across the two-population split, as a detection threshold in Fst.

a  cosine similarity vs Fst at five sample sizes N (phi = 0.5).
b  cosine similarity vs Fst at five sampling fractions phi (N = 200).

Fits, bands and theoretical positions come from src/sweeps_analysis/summarise.py; this
script only draws them (see crossing_panel.crossing_panel_from_summary).

Reads results/sweep_summaries/single_split/{curves,crossings,points}.csv
Writes results/figures/main/PhaseTransitionSingleSplit.pdf (single column)
"""
import os

import genetics_style as gs
from crossing_panel import crossing_table, load_summary
from sweep_figure import draw_pair, pair_height

SUMMARY_DIR = os.path.join(gs._ROOT, "results", "sweep_summaries", "single_split")
WIDTH, AX_H = gs.WIDTH["single"] / gs.MM, 34.0


def make_figure(summary_dir=SUMMARY_DIR):
    gs.setup()
    curves, crossings, points, theory = load_summary(summary_dir)
    common = dict(curves=curves, crossings=crossings, points=points, theory=theory)
    panels = [dict(common, sweep="split_vary_N", title=r"$N$", count=True),
              dict(common, sweep="split_vary_phi", title=r"$\phi$")]

    fig = gs.blank_figure(WIDTH, pair_height(5, WIDTH, AX_H))
    _, (res_N, res_phi) = draw_pair(fig, panels, r"$F_{ST}$", WIDTH, AX_H, top_mm=0)
    print(f"wrote {gs.save(fig, 'PhaseTransitionSingleSplit', 'main')}")
    print(crossing_table(res_N, "N", "Fst"))
    print(crossing_table(res_phi, "phi", "Fst"))
    return fig


if __name__ == "__main__":
    make_figure()
