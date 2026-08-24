#%%
"""
PC alignment for a recently split sister pair, as a detection threshold in Fst.

A  PC alignment vs Fst at nine sample sizes N, with the sisters held at phi_1 = phi_2 = 0.05.
B  PC alignment vs Fst at ten sampling balances, N = 1000 and the sisters sharing 200
   samples between them, split from (5, 195) through (100, 100).

Each curve is fitted through every replicate (not through their medians) with a bootstrap
band. The comparison against theory is a pair of vertical lines rather than a prediction
drawn at every x: a thin solid line at the Fst where the fit reaches squared overlap 0.9,
and a dashed line at the Fst where the theory does.

The theoretical crossing uses the exact heteroskedastic machinery, not the flat sqrt(gamma)
threshold. That distinction matters here: eps_a is nonzero on this demography (tau/tau_bar
spans 0.81-1.17), and at low Fst the exact nu_c sits *above* nu, so the correct prediction
is 0 -- while sqrt(gamma) sits *below* nu and the flat formula would draw a large
false-positive plateau. nu itself is the exact value from the full coalescent-time matrix,
which the analysis already computed; it is not re-derived here.

Why the exact nu and not the split-attributable part. Inverting the overlap law on the 434
cells with |cos| > 0.5 gives nu_implied/nu_exact = 1.018 -- BBP fed the exact eigenvalue
recovers the eigenvalue that produced the observed overlap, to 2%. Against the
pedestal-removed value it is off by 27%. So the additive pedestal contributes real
recoverable signal and belongs in nu.

The measured curves plateau at a nonzero value at low Fst. That plateau is set by the
estimator (a fixed-rank read within the n_1+n_2 rows the axis is supported on) and is not
part of the theoretical curve -- which is exactly why the fit has a free floor rather than
one pinned at zero: the floor absorbs it instead of dragging the transition.

Panel B is the substantive one. Every curve in it spends the *same* 200 samples on the
sister pair; only the balance differs. Because H is a harmonic mean it is capped by
2*min(phi_1, phi_2), so the lopsided designs waste their larger sister: the (5, 195) design
needs a far deeper split to reach the same overlap that (100, 100) reaches, from identical
sequencing effort. The colourbar is therefore labelled by H rather than by a sample count,
since H -- not the number of sister samples -- is what sets detectability.

Reads results/analysis/SisterPCAlignment/{vary_N,vary_phi}_replicates.csv, writes
results/figures/SisterPCAlignment.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from crossing_panel import crossing_panel, crossing_table, encoding_legend
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
from threshold_and_overlap import approx_nu_c, approx_overlap

ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SisterPCAlignment")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis
CMAP_H = cm.plasma
TARGET = 0.9


def theory_for(df, level_key):
    """
    curve -> (Fst -> predicted squared overlap), for the dashed vertical lines.

    nu is very nearly linear in Fst at fixed design, so this fits a line through the
    curve's own (Fst, nu) pairs -- the exact nu the analysis recorded from the full
    coalescent-time matrix -- and feeds it to the heteroskedastic overlap. Fitting the line
    rather than assuming nu = N*H*Fst keeps the additive pedestal: nu does not pass through
    the origin, and dropping the intercept is what makes the approximate form understate it.
    """
    def make(level):
        d = df[df[level_key] == level]
        if d.empty:
            return None
        cells = d.groupby("Fst").first().reset_index()
        if len(cells) < 2:
            return None
        slope, intercept = np.polyfit(cells.Fst.to_numpy(float), cells.nu.to_numpy(float), 1)

        gamma_raw = float(cells.gamma_raw.median())
        mu2 = float(cells.mu2.median())
        delta = float(cells.delta.median())
        phi1, phi2 = float(cells.phi1.iloc[0]), float(cells.phi2.iloc[0])
        # The two sisters plus everything else: only the sister proportions matter for this
        # axis, and the remainder carries no weight on it.
        phi = np.array([phi1, phi2, max(1.0 - phi1 - phi2, 1e-12)])
        eps = np.zeros(3)
        nu_c = approx_nu_c(phi, eps, gamma_raw, mu2, delta)
        return lambda fst: approx_overlap(slope * fst + intercept, phi, eps,
                                          gamma_raw, mu2, delta, nu_c=nu_c)
    return make


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR, target=TARGET):
    setup_plotting_style()
    vary_N = pd.read_csv(os.path.join(analysis_dir, "vary_N_replicates.csv"))
    vary_phi = pd.read_csv(os.path.join(analysis_dir, "vary_phi_replicates.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    H_N = vary_N.H.iloc[0]
    N_phi = int(vary_phi.N.iloc[0])
    sister_total = int(vary_phi.n_sister1.iloc[0] + vary_phi.n_sister2.iloc[0])

    res_N = crossing_panel(axes[0], vary_N, "N", "Fst", CMAP_N, r"$N$", target=target,
                           theory_overlap=theory_for(vary_N, "N"))
    res_H = crossing_panel(axes[1], vary_phi, "H", "Fst", CMAP_H, r"$H$", target=target,
                           theory_overlap=theory_for(vary_phi, "H"))

    for ax in axes:
        ax.set_xlabel(r"$F_{ST}$")
    axes[0].set_title(rf"varying $N$  ($H={H_N:g}$)")
    axes[1].set_title(rf"varying $\phi_1{{:}}\phi_2$  ($N={N_phi}$, $n_1{{+}}n_2={sister_total}$)")
    encoding_legend(axes[0], target=target, loc="lower right")

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SisterPCAlignment.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    print(crossing_table(res_N, "N", "Fst"))
    print(crossing_table(res_H, "H", "Fst"))
    return fig


if __name__ == "__main__":
    make_figure()

# %%
