"""
Shared drawing for the standardized supplementary figures: a pair of cosine-similarity
panels from a sweep summary, with the theory at the actual number of SNPs kept.

Standardized SNP columns have flat locus weights, so L_eff = L and there is no separate
"using L_eff" route to mark: the tick row is dropped and the lines are keyed as "theory".
"""
import numpy as np

import genetics_style as gs
from crossing_panel import crossing_table, load_summary
from sweep_figure import draw_pair, pair_height

WIDTH, AX_H = gs.WIDTH["single"] / gs.MM, 34.0
CRUDE_COLUMNS = ("x_c_crude", "x_cross_crude")
COUNTS = ("N", "n (1D)", "n (2D)")


def make_figure(name, summary_dir, sweeps, x_label):
    """sweeps: [(sweep name, curve-key title, table label)]; labels in COUNTS are sample counts."""
    gs.setup()
    curves, crossings, points, theory = load_summary(summary_dir)
    for frame in (crossings, theory):
        for col in CRUDE_COLUMNS:
            if frame is not None and col in frame:
                frame[col] = np.nan
    panels = [dict(curves=curves, crossings=crossings, points=points, theory=theory,
                   sweep=sweep, title=title, count=label in COUNTS) for sweep, title, label in sweeps]

    fig = gs.blank_figure(WIDTH, pair_height(5, WIDTH, AX_H, n_extra=-2))
    _, results = draw_pair(fig, panels, x_label, WIDTH, AX_H, top_mm=0, crude=False)
    print(f"wrote {gs.save(fig, name, 'supplement')}")
    for (_, _, label), res in zip(sweeps, results):
        print(crossing_table(res, label, x_label.strip("$")))
    return fig
