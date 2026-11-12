"""
Main Fig. 5: spatial-axis recovery in the 1D and 2D stepping-stone models vs 1/M.

a  1D (K = 25 demes), five values of n per deme, every deme sampled.
b  2D (9 x 9 demes), same n grid; cosine similarity of the Procrustes-aligned PC1-PC2 plane.

Reads results/sweep_summaries/stepping_stone_even/{curves,crossings,points}.csv
Writes results/figures/main/PhaseTransitionSS.pdf (single column)
"""
import os

import genetics_style as gs
from crossing_panel import crossing_table, load_summary
from sweep_figure import draw_pair, pair_height

SUMMARY_DIR = os.path.join(gs._ROOT, "results", "sweep_summaries", "stepping_stone_even")
WIDTH, AX_H = gs.WIDTH["single"] / gs.MM, 34.0
SWEEPS = [("ss1d_vary_n", "1D"), ("ss2d_vary_n", "2D")]


def make_figure(summary_dir=SUMMARY_DIR):
    gs.setup()
    curves, crossings, points, theory = load_summary(summary_dir)
    panels = [dict(curves=curves, crossings=crossings, points=points, theory=theory,
                   sweep=sweep, title=r"$n$ per deme", count=True) for sweep, _ in SWEEPS]

    fig = gs.blank_figure(WIDTH, pair_height(5, WIDTH, AX_H))
    _, results = draw_pair(fig, panels, r"$1/M$", WIDTH, AX_H, top_mm=0)
    print(f"wrote {gs.save(fig, 'PhaseTransitionSS', 'main')}")
    for (_, label), res in zip(SWEEPS, results):
        print(f"\n{label}\n{crossing_table(res, 'n', '1/M')}")
    return fig


if __name__ == "__main__":
    make_figure()
