#%%
"""
OOA3 demography and the YRI1/YRI2 sister-split PC, at three split depths (3x3).

Rows: recent, intermediate, deep t_split (very different scales, so the demography preview
uses a log time axis throughout).

Columns:
    1  demesdraw preview of the demography at that t_split
    2  sample PC1 (by overlap with the predicted sister axis), N=10000, phi_1=phi_2=0.05,
       YRI/YRI2 colored red/blue, everyone else light grey
    3  the same at phi_1=phi_2=0.20

Reads results/analysis/OOA3Demography, writes results/figures/OOA3Demography.pdf
"""
import importlib.util
import os
import sys

import demesdraw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "OOA3Demography")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "simulations"))
from demographies.african_split import demography as ooa3_demography

COLOR_YRI, COLOR_YRI2 = 'crimson', 'royalblue'
COLOR_OTHER = '0.8'
LABELS = ["recent", "intermediate", "deep"]


def panel_demography(ax, t_split):
    """demesdraw tube plot, log time axis so all three depths are legible on one scale."""
    dem = ooa3_demography(t_split)
    graph = dem.to_demes()
    demesdraw.tubes(graph, ax=ax, log_time=True)
    ax.set_title(rf'$t_{{\mathrm{{split}}}}={t_split:.0f}$')


def panel_pc(ax, pc1_row):
    """PC1 scatter: YRI/YRI2 in red/blue, everyone else in light grey, mild jitter."""
    pop = pc1_row.population.values
    pc1 = pc1_row.pc1.values
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-1, 1, len(pc1))

    other = ~np.isin(pop, ["YRI", "YRI2"])
    ax.scatter(pc1[other], jitter[other], s=3, color=COLOR_OTHER, alpha=0.6,
              edgecolors='none', zorder=1)
    for p, color in [("YRI", COLOR_YRI), ("YRI2", COLOR_YRI2)]:
        sel = pop == p
        ax.scatter(pc1[sel], jitter[sel], s=5, color=color, alpha=0.85,
                  edgecolors='none', zorder=2, label=p)
    ax.set_yticks([])
    ax.set_xlabel('PC1 (YRI1/YRI2 axis)')


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    pc1 = pd.read_csv(os.path.join(analysis_dir, "pc1.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).set_index("label")

    phis = sorted(pc1.phi.unique())

    fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.5))

    for i, label in enumerate(LABELS):
        t_split = float(meta.loc[label, "t_split"])
        Fst = float(meta.loc[label, "Fst"])

        panel_demography(axes[i, 0], t_split)
        if i == 0:
            axes[i, 0].set_title(axes[i, 0].get_title() + f'\n({label})')
        else:
            axes[i, 0].set_title(axes[i, 0].get_title() + f'\n({label}, '
                                 rf'$F_{{ST}}={Fst:.2e}$)')

        for j, phi in enumerate(phis, start=1):
            d = pc1[(pc1.label == label) & (pc1.phi == phi)]
            panel_pc(axes[i, j], d)
            if i == 0:
                axes[i, j].set_title(rf'$\phi_1=\phi_2={phi:g}$')

    axes[0, 1].legend(loc='upper right', fontsize=8, frameon=False, markerscale=2)

    for i, letter in enumerate("ABC"):
        axes[i, 0].text(-0.12, 1.0, letter, transform=axes[i, 0].transAxes,
                        fontsize=13, fontweight='bold', va='bottom', ha='right')

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "OOA3Demography.pdf")
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
