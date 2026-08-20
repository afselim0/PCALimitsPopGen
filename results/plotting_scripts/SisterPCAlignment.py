#%%
"""
PC alignment for a recently split sister pair, against the zero-parameter prediction.

A  PC alignment vs Fst at nine sample sizes N, with the sisters held at phi_1 = phi_2 = 0.05.
B  PC alignment vs Fst at ten sampling balances, N = 1000 and the sisters sharing 200
   samples between them, split from (5, 195) through (100, 100).

Solid lines are the measured median over replicates; dashed lines are the BBP prediction.

The prediction uses the **exact** nu from the full coalescent-time matrix, not N*H*Fst,
which assumes eps_a = 0 and understates it by nu_exact = 1.379*(N H Fst) + 0.344.

No floor is applied to the prediction -- see bbp_overlap. The measured curves plateau at a
nonzero value at low Fst; that plateau is set by the estimator (argmax over candidate
eigenvectors, within the n_1+n_2 rows the axis is supported on) and is not part of the
theoretical curve. Adding it in quadrature inflates the prediction by ~0.12 and roughly
doubles the error, so it is left off.

Why the exact nu and not the split-attributable part. Inverting the overlap law on the 434
cells with |cos| > 0.5 gives nu_implied/nu_exact = 1.018 -- BBP fed the exact eigenvalue
recovers the eigenvalue that produced the observed overlap, to 2%. Against the
pedestal-removed value it is off by 27%. So the additive pedestal contributes real
recoverable signal and belongs in nu.

Why the support floor. The sister axis is exactly zero outside the sister rows, so the
relevant ambient dimension is n_1+n_2, not N; and the axis is located by argmax over
candidate eigenvectors, which lifts the floor further. See support_overlap_floor.

Residual: the agreement degrades at small H (the implied/exact ratio drifts 0.79 to 1.10
across the H range), which is unmodelled and tracks the sister-block dimension.

Panel B is the substantive one. Every curve in it spends the *same* 200 samples on the
sister pair; only the balance differs. Because H is a harmonic mean it is capped by
2*min(phi_1, phi_2), so the lopsided designs waste their larger sister: at the deepest split
the (5, 195) design gives |cos| ~ 0.16 while (100, 100) gives ~0.80, from the same
sequencing effort. The colourbar is therefore labelled by H rather than by a sample count,
since H -- not the number of sister samples -- is what sets detectability.

Reads results/analysis/SisterPCAlignment, writes results/figures/SisterPCAlignment.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "SisterPCAlignment")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

CMAP_N = cm.viridis
CMAP_H = cm.plasma


def bbp_overlap(nu, gamma):
    """
    |cos| = sqrt((1-gamma/nu^2)/(1+gamma/nu)), unfloored.

    No floor is applied. The measured cosine saturates at a nonzero value when there is no
    signal -- the estimator picks the best-matching eigenvector out of many, in the subspace
    the axis is supported on -- but that is a property of the *estimator*, not of the
    covariance. Adding it to a prediction that already goes to zero double-counts: measured
    against the sweep it inflates every curve by ~0.12 (RMSE 0.123 -> 0.189, bias +0.02 ->
    +0.13). The floor belongs in the interpretation of the measurement's plateau, not in the
    theoretical curve.
    """
    nu = np.asarray(nu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ov = np.where(nu > np.sqrt(gamma), (1 - gamma / nu ** 2) / (1 + gamma / nu), 0.0)
    return np.sqrt(np.clip(ov, 0.0, 1.0))


def panel(ax, df, key, cmap, label, title):
    """
    One sweep: measured median as a solid line, BBP prediction dashed, per level.

    Levels are coloured by the swept parameter on a shared normalization, so the colourbar
    reads directly onto the curves.
    """
    levels = np.sort(df[key].unique())
    norm = colors.Normalize(vmin=levels.min(), vmax=levels.max())

    for level in levels:
        d = df[df[key] == level].sort_values("Fst")
        color = cmap(norm(level))
        ax.plot(d.Fst, d.cos_median, "-", color=color, linewidth=1.5, zorder=3)
        ax.plot(d.Fst, bbp_overlap(d.nu, d.gamma), "--", color=color,
                linewidth=1.1, alpha=0.85, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{ST}$")
    ax.set_ylabel(r"Cosine similarity: $|\langle \hat u_1, u_1\rangle|$")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title(title)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(label)
    cbar.outline.set_visible(False)
    return norm


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    vary_N = pd.read_csv(os.path.join(analysis_dir, "vary_N.csv"))
    vary_phi = pd.read_csv(os.path.join(analysis_dir, "vary_phi.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    n_reps = int(vary_N.num_replicates.iloc[0])
    L_eff_N = vary_N.L_eff.median()
    L_eff_phi = vary_phi.L_eff.median()
    H_N = vary_N.H.iloc[0]
    N_phi = int(vary_phi.N.iloc[0])
    sister_total = int(vary_phi.n_sister1.iloc[0] + vary_phi.n_sister2.iloc[0])

    panel(axes[0], vary_N, "N", CMAP_N, r"$N$",
          rf"varying $N$  ($H={H_N:g}$, $L_{{\mathrm{{eff}}}}\approx{L_eff_N:.0f}$)")
    panel(axes[1], vary_phi, "H", CMAP_H, r"$H$",
          rf"varying $\phi_1{{:}}\phi_2$  ($N={N_phi}$, "
          rf"$n_1{{+}}n_2={sister_total}$, $L_{{\mathrm{{eff}}}}\approx{L_eff_phi:.0f}$)")

    axes[0].plot([], [], "-", color="0.35", linewidth=1.5,
                 label=f"simulation (median of {n_reps})")
    axes[0].plot([], [], "--", color="0.35", linewidth=1.1,
                 label=r"BBP prediction (exact $\nu$)")
    # Lower left: the curves occupy the upper left at high Fst and the legend collided there.
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "SisterPCAlignment.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
