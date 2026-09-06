#%%
"""
OOA3 demography and the YRI1/YRI2 sister-split PC, at two split depths (2x3).

Rows: the two Fst(YRI1, YRI2) targets, 1e-3 then 1e-2, whichever t_split the analysis
matched them to (log time axis throughout, so both are legible on one scale despite the
shared axis reaching back through the full OOA history).

Columns:
    1  demesdraw preview of the demography at that t_split, with Fst(YRI1, YRI2) labelled
    2  sample PC (by overlap with the predicted sister axis), N=1000, L=70,000 polymorphic
       SNPs, phi_1=phi_2=0.05, YRI1/YRI2 colored red/blue, everyone else light grey.
       Each PC panel is titled with its overlap against the predicted sister axis.
    3  the same at phi_1=phi_2=0.20


Reads results/analysis/OOA3Demography, writes results/figures/DemoAfricanSplit.pdf
"""
import copy
import importlib.util
import os
import sys

import demes
import demesdraw
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
import stdpopsim

from crossing_panel import PANEL_LETTER_SIZE, figsize, save_figure
from plotting_style import ANNOT_PT, setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "OOA3Demography")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "simulations"))
from demographies.african_split import demography as ooa3_demography

COLOR_YRI1, COLOR_YRI2 = 'crimson', 'royalblue'
COLOR_OTHER = '0.8'
# Half-height of the PC jitter band, on the +/-1.12 axis set in panel_pc.
JITTER_PC = 1.0
# Height:width of the PC strips. They carry one axis, so they are short by design.
PC_BOX_ASPECT = 0.30
COLOR_ANCESTRAL = '0.6'
LABELS = ["fst1e-3", "fst1e-2"]   # weakest split first, so panel A is the marginal case

# Fixed manually: demesdraw's automatic layout doesn't know YRI1/YRI2 should sit next to
# each other, and its default line density (10 per migration edge) is too busy at this
# figure size. Offsets are in the same units as population size (haploid Ne) and only need
# to clear each deme's half-width, which doesn't depend on t_split.
DEME_POSITIONS = {'YRI': 0, 'YRI1': -15000, 'YRI2': 15000, 'CEU': -52000, 'CHB': 58000}
DEME_COLOURS = {'YRI': COLOR_ANCESTRAL, 'CEU': COLOR_ANCESTRAL, 'CHB': COLOR_ANCESTRAL,
                'YRI1': COLOR_YRI1, 'YRI2': COLOR_YRI2}


def _preview_graph(dem, t_split):
    """
    demes.Graph for the preview, with YRI's tube correctly stopping at t_split.

    dem.to_demes() cannot do this directly: YRI is both the target of the base OOA3
    model's CEU+CHB merge (t=5600) and the ancestral population of the YRI1/YRI2 split
    (t_split), and msprime's to_demes() derives a deme's rendered start_time from the most
    recent event that references it as a merge *target* -- so it clips YRI's tube to end at
    5600 regardless of the later split, unless YRI.initially_active is forced True, in
    which case it goes the other way and draws the tube all the way to the present. Neither
    setting is correct; only the *simulated* demography (dem itself, which does not go
    through to_demes()) is unaffected by this.

    Instead, take the graph from the pre-split 3-population backbone (CEU/CHB/YRI, which
    converts correctly on its own), then patch it by hand: trim YRI's final epoch and any
    migration touching it to end at t_split instead of 0, and add YRI1/YRI2 as its
    descendants from t_split to the present, with their CEU/CHB migration rates read
    directly from dem (where they mirror YRI's by construction -- see demography()).
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
            m = dict(m)
            m["end_time"] = max(m["end_time"], t_split)
        kept.append(m)
    d["migrations"] = kept

    for name in ("YRI1", "YRI2"):
        pid = dem[name].id
        d["demes"].append(dict(
            name=name, ancestors=["YRI"], proportions=[1.0], start_time=t_split,
            epochs=[dict(end_time=0, end_size=yri_size)]))
        for other in ("CEU", "CHB"):
            oid = dem[other].id
            d["migrations"].append(dict(source=name, dest=other, start_time=t_split,
                                        end_time=0, rate=float(dem.migration_matrix[pid, oid])))
            d["migrations"].append(dict(source=other, dest=name, start_time=t_split,
                                        end_time=0, rate=float(dem.migration_matrix[oid, pid])))

    return demes.Graph.fromdict(d)


def _sci_tex(x):
    """Fst as a TeX power of ten: 0.001006 -> 10^{-3}, 0.0203 -> 2\\times 10^{-2}."""
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10.0 ** exponent
    # These are round design targets, so the mantissa is 1 to within rounding and printing
    # it ("1 x 10^-3") would only add noise.
    if abs(mantissa - 1.0) < 0.05:
        return rf'10^{{{exponent}}}'
    return rf'{mantissa:.0f}\times 10^{{{exponent}}}'


def panel_demography(ax, t_split, Fst):
    """demesdraw tube plot, log time axis so both depths are legible on one scale."""
    dem = ooa3_demography(t_split)
    graph = _preview_graph(dem, t_split)
    demesdraw.tubes(graph, ax=ax, log_time=True, num_lines_per_migration=2,
                    positions=DEME_POSITIONS, colours=DEME_COLOURS)
    # YRI is the unsampled split ancestor -- zero width at present, and its tick label
    # only crowds the space between its two children.
    ax.set_xticklabels(['' if t.get_text() == 'YRI' else t.get_text()
                        for t in ax.get_xticklabels()],
                       fontsize=ANNOT_PT * 0.85)
    ax.set_title(rf'$t_{{\mathrm{{split}}}}={t_split:.0f}$ generations ago' + '\n' +
                rf'$F_{{ST}}={_sci_tex(Fst)}$')


def panel_pc(ax, pc1_row):
    """Sister-axis PC scatter: YRI1/YRI2 in red/blue, everyone else light grey, mild jitter.

    The axis label reports the rank the analysis actually selected (pc1.csv's `rank`
    column), which is normally 2 -- i.e. PC3, behind the three deep OOA splits.

    Returns the overlap against the predicted sister axis, for the panel title.
    """
    pop = pc1_row.population.values
    pc1 = pc1_row.pc1.values
    rank = int(pc1_row["rank"].iloc[0])
    rng = np.random.default_rng(0)
    # A shallow band, not the full panel: the y axis carries no quantity, and jitter
    # spread over the whole box reads as a second axis of variation.
    jitter = rng.uniform(-JITTER_PC, JITTER_PC, len(pc1))

    other = ~np.isin(pop, ["YRI1", "YRI2"])
    ax.scatter(pc1[other], jitter[other], s=2.0, color=COLOR_OTHER, alpha=0.6,
              edgecolors='none', zorder=1)
    for p, color in [("YRI1", COLOR_YRI1), ("YRI2", COLOR_YRI2)]:
        sel = pop == p
        ax.scatter(pc1[sel], jitter[sel], s=2.5, color=color, alpha=0.85,
                  edgecolors='none', zorder=2, label=p)
    ax.set_yticks([])
    # Clipped right around the jitter band, with the left spine removed: there is no y
    # quantity for either the range or the spine to describe. Combined with the short box
    # aspect set in make_figure, this puts the x axis just under the points.
    ax.set_ylim(-1.12, 1.12)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(f'PC{rank + 1} (YRI1/YRI2 axis)')
    return float(pc1_row.overlap.iloc[0])


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    pc1 = pd.read_csv(os.path.join(analysis_dir, "pc1.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("label")

    phis = sorted(pc1.phi.unique())

    # Two blocks (one per t_split), each a tall demography panel beside a SHORT PC strip.
    # The strip gets its own short gridspec row rather than a short box inside a tall one:
    # squeezing it with set_box_aspect left a void under it the height of the demography.
    fig = plt.figure(figsize=figsize(9.0, 5.6))
    outer = fig.add_gridspec(len(LABELS), 1, hspace=0.55)
    axes = np.empty((len(LABELS), 3), dtype=object)
    for i in range(len(LABELS)):
        inner = outer[i].subgridspec(2, 3, height_ratios=[0.001, 1.0], wspace=0.30)
        axes[i, 0] = fig.add_subplot(inner[:, 0])
        for j in (1, 2):
            axes[i, j] = fig.add_subplot(inner[1, j])

    for i, label in enumerate(LABELS):
        t_split = float(meta.loc[label, "t_split"])
        Fst = float(meta.loc[label, "Fst"])

        panel_demography(axes[i, 0], t_split, Fst)

        for j, phi in enumerate(phis, start=1):
            d = pc1[(pc1.label == label) & (pc1.phi == phi)]
            overlap = panel_pc(axes[i, j], d)
            axes[i, j].set_title(rf'$\phi_1=\phi_2={phi:g}$' + '\n' +
                                 rf'$\langle\hat u,u\rangle^2 = {overlap:.3f}$')

    # axes[0, 1].legend(loc='upper right', fontsize=8, frameon=False, markerscale=2)

    # Each ROW is one t_split, so the figure has two labelled units, not six. The letter
    # sits beside the demography panel that opens its row.
    for ax, letter in zip(axes[:, 0], "AB"):
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -26 / 72, 16 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight='bold', va='top', ha='left')

    # The PC strips are short and sit on the FLOOR of their block, level with the bottom
    # of the demography beside them, so every x axis in the figure lines up. They show one
    # axis -- PC3 -- with jitter standing in for nothing at all, so height spent on them
    # only invites reading the vertical spread as a second dimension.
    for ax in axes[:, 1:].flat:
        ax.set_box_aspect(PC_BOX_ASPECT)
        ax.set_anchor("S")

    out = save_figure(fig, figure_dir, "DemoAfricanSplit")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
