"""
Main Fig. 3: the YRI1/YRI2 sister split inside OOA3, and its detection threshold in Fst.

a  demesdraw preview of OOA3 with YRI split into YRI1/YRI2 at t_split ~ 25 generations
   (Fst(YRI1, YRI2) = 1e-3), log time axis.
b  the sister-axis PC (selected by overlap with the predicted axis), N = 1000,
   phi_1 = phi_2 = 0.05; YRI1/YRI2 coloured, everyone else grey, vertical jitter only.
c  the same at phi_1 = phi_2 = 0.20.
d  cosine similarity vs Fst at five N (harmonic mean phi~ = 0.20).
e  cosine similarity vs Fst at six (phi_1, phi_2) designs with N = 1000, keyed by phi~.

Reads results/analysis/OOA3Demography/{pc1,meta}.csv and
      results/sweep_summaries/african_split/{curves,crossings,points}.csv
Writes results/figures/main/PhaseTransitionAfricanSplit.pdf (single column)
"""
import copy
import os
import sys

import demes
import demesdraw
import numpy as np
import pandas as pd
import stdpopsim

import genetics_style as gs
from crossing_panel import crossing_table, load_summary
from sweep_figure import LEFT, RIGHT, draw_pair, pair_height

ANALYSIS_DIR = os.path.join(gs._ROOT, "results", "analysis", "OOA3Demography")
SUMMARY_DIR = os.path.join(gs._ROOT, "results", "sweep_summaries", "african_split")
sys.path.insert(0, os.path.join(gs._ROOT, "src", "simulations"))
from demographies.african_split import demography as ooa3_demography

WIDTH, AX_H = gs.WIDTH["single"] / gs.MM, 34.0
LABEL = "fst1e-3"                     # the marginal split shown in a-c
TOP_H = 38.0                          # mm for the demography row
STRIP_H = 8.0                         # mm, height of each PC strip
COLOR_ANCESTRAL = "0.6"
# Placement in demesdraw's x units (haploid Ne): YRI1/YRI2 adjacent, CEU/CHB outside
DEME_POSITIONS = {"YRI": 0, "YRI1": -15000, "YRI2": 15000, "CEU": -52000, "CHB": 58000}
DEME_COLOURS = {"YRI": COLOR_ANCESTRAL, "CEU": COLOR_ANCESTRAL, "CHB": COLOR_ANCESTRAL,
                "YRI1": gs.POP1, "YRI2": gs.POP2}


def _preview_graph(dem, t_split):
    """
    demes.Graph for the preview, with YRI's tube stopping at t_split.

    msprime's to_demes() clips YRI to the CEU+CHB merge (or extends it to 0), so the graph is
    built from the 3-population OOA backbone and patched: YRI ends at t_split, and YRI1/YRI2
    descend from it with their CEU/CHB migration rates read from `dem`.
    """
    species = stdpopsim.get_species("HomSap")
    backbone = copy.deepcopy(species.get_demographic_model("OutOfAfrica_3G09").model)
    backbone["YRI"].initially_active = True
    d = backbone.to_demes().asdict()

    yri_size = None
    for deme in d["demes"]:
        if deme["name"] == "YRI":
            deme["epochs"][-1]["end_time"] = t_split
            yri_size = deme["epochs"][-1]["end_size"]

    kept = []
    for m in d.get("migrations", []):
        if m["source"] == "YRI" or m["dest"] == "YRI":
            if m["start_time"] <= t_split:
                continue
            m = dict(m, end_time=max(m["end_time"], t_split))
        kept.append(m)
    d["migrations"] = kept

    for name in ("YRI1", "YRI2"):
        pid = dem[name].id
        d["demes"].append(dict(name=name, ancestors=["YRI"], proportions=[1.0],
                               start_time=t_split, epochs=[dict(end_time=0, end_size=yri_size)]))
        for other in ("CEU", "CHB"):
            oid = dem[other].id
            d["migrations"].append(dict(source=name, dest=other, start_time=t_split, end_time=0,
                                        rate=float(dem.migration_matrix[pid, oid])))
            d["migrations"].append(dict(source=other, dest=name, start_time=t_split, end_time=0,
                                        rate=float(dem.migration_matrix[oid, pid])))
    return demes.Graph.fromdict(d)


def panel_demography(ax, t_split):
    graph = _preview_graph(ooa3_demography(t_split), t_split)
    demesdraw.tubes(graph, ax=ax, log_time=True, num_lines_per_migration=2,
                    positions=DEME_POSITIONS, colours=DEME_COLOURS, labels="xticks",
                    fill=True)
    # YRI is the unsampled split ancestor; its tick label only crowds YRI1/YRI2
    ax.set_xticks([DEME_POSITIONS[p] for p in ("CEU", "YRI1", "YRI2", "CHB")])
    ax.set_xticklabels(["CEU", "YRI1", "YRI2", "CHB"], fontsize=gs.TICK_PT)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_ylabel("time ago (generations)")
    for line in ax.lines:
        line.set_linewidth(min(line.get_linewidth(), 0.8))
    for coll in ax.collections:
        coll.set_linewidth(0.5)
    for patch in ax.patches:
        patch.set_linewidth(0.6)
        if hasattr(patch, "set_mutation_scale"):
            patch.set_mutation_scale(5)


def panel_pc(ax, d):
    """Sister-axis PC strip: YRI1/YRI2 coloured, others grey, jitter carries no quantity."""
    pop, pc = d.population.values, d.pc1.values
    jitter = np.random.default_rng(0).uniform(-1, 1, len(pc))
    other = ~np.isin(pop, ["YRI1", "YRI2"])
    ax.scatter(pc[other], jitter[other], s=1.2, color=gs.BACKGROUND, edgecolors="none",
               zorder=1, rasterized=True)
    for p, color in (("YRI1", gs.POP1), ("YRI2", gs.POP2)):
        sel = pop == p
        ax.scatter(pc[sel], jitter[sel], s=1.5, color=color, alpha=0.9, edgecolors="none",
                   zorder=2, rasterized=True)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    # baseline 1.6 mm above the strip, level with the panel letter beside it
    ax.text(0.5, 1 + 1.6 / STRIP_H, rf"$\phi_1=\phi_2={d.phi.iloc[0]:g}$,  "
            rf"$\langle\hat u,u\rangle^2={d.overlap.iloc[0]:.2f}$",
            transform=ax.transAxes, ha="center", va="baseline", fontsize=gs.TICK_PT)
    return int(d["rank"].iloc[0])


def make_figure(analysis_dir=ANALYSIS_DIR, summary_dir=SUMMARY_DIR):
    gs.setup()
    pc1 = pd.read_csv(os.path.join(analysis_dir, "pc1.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("label")
    curves, crossings, points, theory = load_summary(summary_dir)

    H = TOP_H + pair_height(6, WIDTH, AX_H)
    fig = gs.blank_figure(WIDTH, H)

    # Top row: demography (a) beside two stacked PC strips (b, c), letters on one line
    dem_w, strip_x = 30.0, LEFT + 30.0 + 7.0
    strip_w = WIDTH - strip_x - RIGHT
    strip_tops = (H - 4.0, H - 21.0)
    ax_dem = gs.axes_mm(fig, LEFT, strip_tops[1] - STRIP_H, dem_w,
                        H - 3.0 - (strip_tops[1] - STRIP_H))
    panel_demography(ax_dem, float(meta.loc[LABEL, "t_split"]))
    gs.label_panel_mm(fig, "a", 0.3, H)

    for i, (phi, top) in enumerate(zip(sorted(pc1.phi.unique()), strip_tops)):
        ax = gs.axes_mm(fig, strip_x, top - STRIP_H, strip_w, STRIP_H)
        rank = panel_pc(ax, pc1[(pc1.label == LABEL) & (pc1.phi == phi)])
        gs.label_panel_mm(fig, "bc"[i], strip_x - 5.5, top + 4.0)
    ax.set_xlabel(f"PC{rank + 1} (YRI1/YRI2 axis)")

    panels = [dict(curves=curves, crossings=crossings, points=points, theory=theory,
                   sweep="african_split_vary_N", title=r"$N$"),
              dict(curves=curves, crossings=crossings, points=points, theory=theory,
                   sweep="african_split_vary_harmonic_phi", title=r"$\tilde{\phi}$")]
    _, (res_N, res_H) = draw_pair(fig, panels, r"$F_{ST}$", WIDTH, AX_H, top_mm=TOP_H,
                                  letters="de")
    print(f"wrote {gs.save(fig, 'PhaseTransitionAfricanSplit', 'main')}")
    print(crossing_table(res_N, "N", "Fst"))
    print(crossing_table(res_H, "H", "Fst"))
    return fig


if __name__ == "__main__":
    make_figure()
