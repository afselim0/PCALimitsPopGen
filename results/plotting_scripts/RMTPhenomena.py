#%%
"""
Three RMT phenomena on a two-population split, at N = 1e3 and L_input = 1e5.

A  Eigenvalue spreading. The 398 bulk eigenvalues of one spectrum (grey histogram) against
   the Marchenko-Pastur density computed two ways: with the raw SNP count L, and with the
   variance-effective count L_eff = (sum v)^2 / sum v^2, v = p(1-p). Only the L_eff curve
   matches the observed bulk -- the GRM is built from unnormalized genotypes, so each
   column enters with weight v_l and the aspect ratio is N/L_eff, not N/L. The thick black
   line marks the 398 identical theoretical eigenvalues, which as a density is a single
   spike.

B  Spiked eigenvalue transition. The gap lambda_1 / lambda_2 against Fst. Below threshold
   the spike is absorbed in the bulk and the ratio sits near 1; above it the spike detaches
   and the ratio climbs.

C  Eigenvector inconsistency. |<v1_hat, v1>| against Fst, with PC1 scatters at three Fst
   values chosen for no / partial / near-perfect recovery.

Both B and C mark the two candidate thresholds, Fst_c = 1/(2*phi*(1-phi)*sqrt(N*L)) with
raw L and with L_eff, so the panels show which one the transition actually respects.

Reads results/analysis/RMTPhenomena, writes results/figures/RMTPhenomena.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import ScaledTranslation

from plotting_style import okabe_ito, setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RMTPhenomena")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

COLOR_MAIN = okabe_ito[1]        # vermillion: the measured curve
COLOR_EFF = okabe_ito[0]         # blue: theory with L_eff
COLOR_RAW = okabe_ito[3]         # orange: theory with raw L
COLOR_POP1, COLOR_POP2 = 'crimson', 'royalblue'


def sci_latex(x, digits=1):
    """Format x as a LaTeX 'm \\times 10^{e}' string rather than matplotlib's '1.2e-03'."""
    exponent = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    mantissa = x / 10.0 ** exponent
    return rf'{mantissa:.{digits}f} \times 10^{{{exponent}}}'


def mp_density(x, gamma, scale=1.0):
    """MP density for a floor at `scale`, supported on scale * (1 -/+ sqrt(gamma))^2."""
    lo, hi = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2
    u = np.asarray(x, dtype=float) / scale
    out = np.zeros_like(u)
    inside = (u > lo) & (u < hi)
    out[inside] = (np.sqrt((hi - u[inside]) * (u[inside] - lo))
                   / (2 * np.pi * gamma * u[inside]))
    return out / scale


def panel_A(ax, spectrum, meta):
    """Bulk histogram against the two MP densities."""
    bulk = spectrum.eigenvalue.values
    N = int(meta.N)
    gamma_raw = N / float(spectrum.L_raw.iloc[0])
    gamma_eff = N / float(spectrum.L_eff.iloc[0])
    # The floor is where the theoretical eigenvalues sit and what both MP densities are
    # centered on. The MP density has mean 1 (in floor units), so the bulk *mean* estimates
    # it; the median does not -- the density is left-skewed, and at this gamma the median
    # sits ~2.5% low, which would visibly offset the line from both curves' centers.
    floor = float(bulk.mean())

    ax.hist(bulk, bins=28, density=True, color='0.78', edgecolor='none',
            label=f'{bulk.size} bulk eigenvalues')

    # Each curve is drawn over its own support, so the narrower raw-L density is visibly
    # confined well inside the observed bulk rather than being clipped to it.
    L_raw = float(spectrum.L_raw.iloc[0])
    L_eff = float(spectrum.L_eff.iloc[0])
    for gamma, color, label in [
            (gamma_raw, COLOR_RAW, rf'raw $L$  ({L_raw:.0f})'),
            (gamma_eff, COLOR_EFF, rf'effective $L$  ({L_eff:.0f})')]:
        grid = np.linspace(floor * (1 - np.sqrt(gamma)) ** 2,
                           floor * (1 + np.sqrt(gamma)) ** 2, 800)
        ax.plot(grid, mp_density(grid, gamma, scale=floor), color=color, linewidth=2.3,
                label=label)

    ax.axvline(floor, color='black', linewidth=2.5,
               label=f'{bulk.size} identical theoretical eigenvalues')
    ax.set_xlabel(r'sample eigenvalue $\hat\lambda$')
    ax.set_ylabel('density')
    ax.set_ylim(bottom=0)
    ax.set_title('Eigenvalue spreading')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), fontsize=11,
              frameon=False)


def _threshold_lines(ax, meta):
    """The two candidate critical Fst, dotted."""
    ax.axvline(float(meta.Fst_c_raw), color=COLOR_RAW, linestyle=':', linewidth=1.6,
               label=r'$F_{ST}^{\mathrm{critical}}$ (raw $L$)')
    ax.axvline(float(meta.Fst_c_eff), color=COLOR_EFF, linestyle=':', linewidth=1.6,
               label=r'$F_{ST}^{\mathrm{critical}}$ ($L_{\mathrm{eff}}$)')


def panel_B(ax, sweep, meta):
    """Eigenvalue gap against Fst."""
    ax.plot(sweep.Fst, sweep.gap_median, '-', color=COLOR_MAIN, linewidth=1.5)
    _threshold_lines(ax, meta)
    ax.set_xscale('log')
    ax.set_xlabel(r'$F_{ST}$')
    ax.set_ylabel(r'$\hat\lambda_1\,/\,\hat\lambda_2$')
    ax.set_title('Spiked eigenvalue transition')
    # Below the axes, matching panel A. Panel C omits its legend entirely and is read
    # against this one, so it has to sit where both panels can be compared against it.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), fontsize=11,
              frameon=False)


def panel_C(ax, sweep, meta):
    """Cosine similarity against Fst."""
    ax.plot(sweep.Fst, sweep.cos_median, '-', color=COLOR_MAIN, linewidth=1.5)
    _threshold_lines(ax, meta)
    ax.set_xscale('log')
    ax.set_xlabel(r'$F_{ST}$')
    ax.set_ylabel(r'$|\langle \hat v_1, v_1\rangle|$')
    ax.set_ylim(-0.04, 1.04)
    ax.set_title('Eigenvector inconsistency')
    # No legend: the two threshold lines are the same as panel B's and are read off it.


def add_insets(fig, ax_C, pc1):
    """
    PC1 scatters under panel C, one per representative Fst.

    Placed in figure coordinates below C's axes box. The vertical offset clears C's own
    x-axis label and its legend; the inset titles sit above each box, so the gap has to
    accommodate both or the two collide.
    """
    fig.canvas.draw()                       # finalize C's position before reading it
    fig_w_in, fig_h_in = fig.get_size_inches()
    pos = ax_C.get_position()

    groups = [g for _, g in pc1.groupby('target_cos')]
    groups.sort(key=lambda g: float(g.Fst.iloc[0]))

    inset_w = pos.width / 3 * 0.80
    inset_gap = pos.width * 0.10
    inset_h = inset_w * fig_w_in / fig_h_in      # equal physical size -> square
    # Panel C has no legend under it, so the insets can sit closer than the offset needed
    # to clear panels A and B's below-axes legends.
    inset_bottom = pos.y0 - 0.115 - inset_h

    for j, g in enumerate(groups):
        left = pos.x0 + j * (inset_w + inset_gap)
        ax_in = fig.add_axes([left, inset_bottom, inset_w, inset_h])
        pc1_vals = g.pc1.values
        pop = g.population.values
        rng = np.random.default_rng(j)
        jitter = rng.uniform(-1, 1, len(pc1_vals))
        for p, color in [(0, COLOR_POP1), (1, COLOR_POP2)]:
            sel = pop == p
            ax_in.scatter(pc1_vals[sel], jitter[sel], s=1.6, color=color,
                          alpha=0.75, edgecolors='none')
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_xlabel('PC1', fontsize=8, labelpad=2)
        ax_in.set_title(rf'$F_{{ST}}={sci_latex(float(g.Fst.iloc[0]))}$',
                        fontsize=8.5, pad=3)
        for spine in ax_in.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    sweep = pd.read_csv(os.path.join(analysis_dir, "sweep.csv"))
    spectrum = pd.read_csv(os.path.join(analysis_dir, "spectrum.csv"))
    pc1 = pd.read_csv(os.path.join(analysis_dir, "pc1.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).iloc[0]

    fig = plt.figure(figsize=(11.0, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[4.0, 1.9], hspace=0.62, wspace=0.34)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    panel_A(ax_A, spectrum, meta)
    panel_B(ax_B, sweep, meta)
    panel_C(ax_C, sweep, meta)

    for ax, letter in zip((ax_A, ax_B, ax_C), "ABC"):
        ax.text(-0.05, 0.95, letter, transform=ax.transAxes + ScaledTranslation(
            -18 / 72, 22 / 72, fig.dpi_scale_trans),
            fontsize=13, fontweight='bold', va='bottom', ha='left')

    add_insets(fig, ax_C, pc1)

    os.makedirs(figure_dir, exist_ok=True)
    out = os.path.join(figure_dir, "RMTPhenomena.pdf")
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
