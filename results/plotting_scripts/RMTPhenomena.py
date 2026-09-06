#%%
"""
Three RMT phenomena on a two-population split, at N = 400 and L_input = 2e4.

A  Eigenvalue spreading. The bulk eigenvalues of one spectrum (grey histogram) against
   the Marchenko-Pastur density computed two ways: with the raw SNP count L, and with the
   variance-effective count L_eff = (sum v)^2 / sum v^2, v = p(1-p). Only the L_eff curve
   matches the observed bulk -- the GRM is built from unnormalized genotypes, so each
   column enters with weight v_l and the aspect ratio is N/L_eff, not N/L. The thick black
   line marks the identical theoretical eigenvalues, which as a density is a single spike.

B  BBP phase transition. The gap lambda_1 / lambda_2 against Fst, one spectrum per
   Fst (no replicates). Below threshold the spike is absorbed in the bulk and the ratio
   sits near 1; above it the spike detaches and the ratio climbs.

C  Eigenvector inconsistency. The measured <v1_hat, v1>^2 against Fst at N = 400, with
   its 95% confidence band, against the theoretical curve. All three are READ from
   results/sweep_summaries/single_split -- the same fit, band and theory the single-split
   sweep figure draws -- so this panel cannot disagree with figure 3. PC1 scatters sit below at three Fst
   values: very small, just above the detection threshold, and very large.

Both B and C mark the two candidate thresholds, Fst_c = 1/(2*phi*(1-phi)*sqrt(N*L)) with
raw L and with L_eff, so the panels show which one the transition actually respects.

Reads results/analysis/RMTPhenomena, writes results/figures/RMTPhenomena.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import ScaledTranslation

from crossing_panel import PANEL_LETTER_SIZE, figsize, save_figure
from plotting_style import ANNOT_PT, okabe_ito, setup_plotting_style


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RMTPhenomena")
# Panel C reads its fitted curve, band and theory from the sweep pipeline's own output.
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries", "single_split")
SWEEP_LEVEL = 400
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

COLOR_MAIN = okabe_ito[1]        # vermillion: panel C's fitted cosine-similarity curve
COLOR_GAP = okabe_ito[2]         # green: panel B's measured eigenvalue-gap curve --
                                  # a different quantity (a ratio, not a fit) from C's,
                                  # so it gets its own color rather than sharing COLOR_MAIN
COLOR_EFF = okabe_ito[0]         # blue: theory with L_eff
COLOR_RAW = okabe_ito[3]         # orange: theory with raw L
COLOR_POP1, COLOR_POP2 = 'crimson', 'royalblue'
# Half-height of the PC1 jitter band in the insets, whose y limits are +/-1.
INSET_JITTER = 0.18
# Inset height as a fraction of the square it would otherwise be -- see add_insets.
INSET_ASPECT = 1.0


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
        ax.plot(grid, mp_density(grid, gamma, scale=floor), color=color, linewidth=1.3,
                label=label)

    ax.axvline(floor, color='black', linewidth=1.4,
               label=f'{bulk.size} identical theoretical $\\lambda$')
    ax.set_xlabel(r'sample eigenvalue $\hat\lambda$')
    ax.set_ylabel('density')
    ax.set_ylim(bottom=0)
    ax.set_title('Eigenvalue spreading')
    # Below the panel, in the gridspec row reserved under A. Anchored to that row's own
    # axes rather than floated at a figure coordinate, so the layout accounts for it --
    # the earlier bbox_to_anchor=(0.5, -0.20) was invisible to tight_layout and drifted
    # into the page margin when the canvas narrowed.
    return ax.get_legend_handles_labels()


def _threshold_lines(ax, meta):
    """The two candidate critical Fst, dotted."""
    ax.axvline(float(meta.Fst_c_raw), color=COLOR_RAW, linestyle=':', linewidth=1.2,
               label=r'$F_{ST}^{\mathrm{critical}}$ (raw $L$)')
    ax.axvline(float(meta.Fst_c_eff), color=COLOR_EFF, linestyle=':', linewidth=1.2,
               label=r'$F_{ST}^{\mathrm{critical}}$ (effective $L$)')


def panel_B(ax, gap, meta):
    """Eigenvalue gap against Fst."""
    ax.plot(gap.Fst, gap.gap, '-', color=COLOR_GAP, linewidth=1.1,
            label='median ratio of first and\nsecond largest eigenvalues')
    _threshold_lines(ax, meta)
    ax.set_xscale('log')
    ax.set_xlabel(r'$F_{ST}$')
    ax.set_ylabel(r'$\hat\lambda_1\,/\,\hat\lambda_2$')
    ax.set_title('BBP phase transition')
    # The two threshold lines are keyed HERE, under B, and not in C -- both panels draw
    # the same pair, so naming them once is enough, and B is where the eigenvalue
    # transition they mark is actually read off. Returned for the shared row under the
    # panels rather than placed in-axes: two long math labels do not fit beside the curve.
    return ax.get_legend_handles_labels()


def panel_C(ax, cos_sweep, meta, summary=None):
    """
    Squared cosine similarity against Fst: the fitted measurement against the theory.

    Everything drawn here is READ, not computed. The single_split sweep already fits this
    exact curve at N = 400 and publishes it in results/sweep_summaries/single_split:

        curves.csv  y_fit with its 95% band (y_lo, y_hi) -- the shared threshold fit
                    through every replicate, the same model behind every sweep figure
        theory.csv  cos2_exact / cos2_crude, the prediction per x on the two routes
        points.csv  the replicate scatter behind the fit

    So this panel is the same measurement and the same theory as figure 3, at one level,
    rather than a second computation of them that could drift out of agreement. The band is
    the sweep's own confidence band, which is what makes it a credible interval on the
    curve rather than a decoration.

    Falls back to the medians in cos_sweep.csv when the summary is unavailable, so the
    panel still draws without the sweep pipeline.
    """
    if summary is not None:
        curve, theory, pts = summary
        if pts is not None and len(pts):
            ax.scatter(pts.x, pts.cos2, s=3, alpha=0.30, linewidths=0,
                       color=COLOR_MAIN, zorder=1)
        # Unkeyed: a band hugging its own line reads as the line's uncertainty without
        # being named, and the key is better spent on the measured/theory contrast.
        ax.fill_between(curve.x, curve.y_lo, curve.y_hi, color=COLOR_MAIN,
                        alpha=0.30, linewidth=0, zorder=2)
        ax.plot(curve.x, curve.y_fit, '-', color=COLOR_MAIN, linewidth=1.1, zorder=3,
                label='I-Spline fit to\nobserved similarity')
        # The exact Ding & Yang route, the same one the sweep figures draw against.
        ax.plot(theory.x, theory.cos2_exact, '--', color='0.35', linewidth=1.0,
                zorder=4, label='theory')
    else:
        d = cos_sweep.sort_values('Fst')
        ax.plot(d.Fst, d.median_cos.to_numpy(float) ** 2, '-', color=COLOR_MAIN,
                linewidth=1.5, label='I-Spline fit to\nobserved similarity')
        ax.plot(d.Fst, d.analytical_cos.to_numpy(float) ** 2, '--', color='0.35',
                linewidth=1.3, label='theory')

    fit_handles = ax.get_legend_handles_labels()
    _threshold_lines(ax, meta)          # drawn, but keyed under B
    ax.set_xscale('log')
    ax.set_xlabel(r'$F_{ST}$')
    ax.set_ylabel(r'cosine similarity $\langle\hat v_1, v_1\rangle^2$')
    ax.set_ylim(-0.04, 1.04)
    ax.set_title('Eigenvector inconsistency')
    # Only the fit and the theory. The threshold lines this panel also draws are keyed
    # under B, so naming them again here would repeat two long entries. Bottom-right, and
    # pushed PAST the right spine (x=1.16): the fit and theory curves saturate
    # along the top and their rise cuts through the panel's interior, so a key anchored
    # inside the axes sat on the lines. The overhang is small enough to stay clear of the
    # page edge, which bbox_inches="tight" expands to include.
    # Returned for the row beneath the panels, like A's and B's: two 12 pt entries do not
    # fit inside a panel this narrow without landing on the curve.
    return fit_handles


def load_sweep_summary(level, sweep='split_vary_N',
                       summary_dir=SUMMARY_DIR):
    """
    The fitted curve, theory and replicate scatter for one level of the single_split sweep.

    Returns (curves, theory, points) filtered to `level`, or None when the summary is
    absent -- panel C then falls back to the medians it already reads.
    """
    try:
        curve = pd.read_csv(os.path.join(summary_dir, "curves.csv"))
        theory = pd.read_csv(os.path.join(summary_dir, "theory.csv"))
    except FileNotFoundError:
        return None
    sel = lambda d: d[(d.sweep == sweep) & (d.level == level)].sort_values("x")
    curve, theory = sel(curve), sel(theory)
    if curve.empty or theory.empty:
        return None
    try:
        points = sel(pd.read_csv(os.path.join(summary_dir, "points.csv")))
    except FileNotFoundError:
        points = None
    return curve, theory, points


def add_insets(fig, gs_cell, pc1):
    """
    PC1 scatters, one per representative Fst, in the gridspec cell under panel C.

    Laid out as a nested gridspec inside `gs_cell` rather than floated at hand-tuned
    figure coordinates. The old version placed them with `ax_C.get_position()` offsets
    measured on the pre-6.5in canvas, so on the narrower page they detached from the row
    and left a band of empty page; a nested gridspec is measured by the layout engine and
    tracks whatever height the row is actually given.
    """
    # Fixed left-to-right order (small, threshold, large Fst) rather than sorted by value,
    # since the insets are meant to read as a progression across the sweep.
    order = ['small', 'threshold', 'large']
    groups = [pc1[pc1.inset == key] for key in order]

    # A dedicated top row for the shared "PC1 at three Fst values" label, above the insets
    # rather than below them as an x label: it now names the whole row of three boxes at a
    # glance instead of reading like a caption for the last (large-Fst) box alone.
    outer = gs_cell.subgridspec(2, 1, height_ratios=[0.28, 1.0], hspace=0.1)
    ax_label = fig.add_subplot(outer[0, 0])
    ax_label.axis("off")
    # Same size as the legends (LEGEND_PT, not ANNOT_PT) and nudged down from the cell top
    # so its baseline lines up with "median ratio..." in B's key, which the earlier fixed
    # top alignment across panels also nudged down (bbox_to_anchor y > 1.0).
    ax_label.text(0.5, 0.86, r'PC1 at three $F_{ST}$ values',
                 fontsize=plt.rcParams['legend.fontsize'],
                 ha='center', va='top', transform=ax_label.transAxes)
    inner = outer[1, 0].subgridspec(1, 3, wspace=0.28)

    for j, g in enumerate(groups):
        ax_in = fig.add_subplot(inner[0, j])
        pc1_vals = g.pc1.values
        pop = g.population.values
        rng = np.random.default_rng(j)
        # A thin jitter band, not the full inset height: these panels show a single PC, and
        # jitter spread over the whole box reads as a second axis of variation.
        jitter = rng.uniform(-INSET_JITTER, INSET_JITTER, len(pc1_vals))
        for p, color in [(0, COLOR_POP1), (1, COLOR_POP2)]:
            sel = pop == p
            ax_in.scatter(pc1_vals[sel], jitter[sel], s=1.6, color=color,
                          alpha=0.75, edgecolors='none')
        ax_in.set_ylim(-1, 1)
        ax_in.set_box_aspect(INSET_ASPECT)
        # A fixed box aspect otherwise centers the axes within its (now taller) cell,
        # which would push the box down and leave blank space above the title instead of
        # letting the whole inset+title block sit flush against the cell's top.
        ax_in.set_anchor('N')
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        # Just the value, not "F_ST = value": three of these sit across one panel's width,
        # and the repeated symbol made the titles wider than their own boxes and overlap.
        # The quantity is named once, in the shared row label above (ax_label).
        ax_in.set_title(rf'${sci_latex(float(g.Fst.iloc[0]), digits=0)}$',
                        fontsize=ANNOT_PT * 0.85, pad=2)
        for spine in ax_in.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)


def make_figure(analysis_dir=ANALYSIS_DIR, figure_dir=FIGURE_DIR):
    setup_plotting_style()
    gap = pd.read_csv(os.path.join(analysis_dir, "gap.csv"))
    cos_sweep = pd.read_csv(os.path.join(analysis_dir, "cos_sweep.csv"))
    spectrum = pd.read_csv(os.path.join(analysis_dir, "spectrum.csv"))
    pc1 = pd.read_csv(os.path.join(analysis_dir, "pc1.csv"))
    meta = pd.read_csv(os.path.join(analysis_dir, "meta.csv")).iloc[0]

    fig = plt.figure(figsize=figsize(11.0, 7.6))
    # Row 1 carries what sits under each panel: A's and B's keys, and C's PC1 insets
    # (C's own key is in-axes -- see below). Row 2 is unused since C's insets moved up
    # into row 1 alongside B's key; bbox_inches="tight" trims the resulting blank strip.
    gs = fig.add_gridspec(3, 3, height_ratios=[4.6, 1.3, 1.1], hspace=0.55, wspace=0.55)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    handles_A = panel_A(ax_A, spectrum, meta)
    handles_B = panel_B(ax_B, gap, meta)
    handles_C = panel_C(ax_C, cos_sweep, meta, load_sweep_summary(SWEEP_LEVEL))

    # A's and B's keys go in the row beneath their own panel, on an invisible axes so the
    # legend is a laid-out artist rather than a floating figure-coordinate annotation.
    ax_key_A = fig.add_subplot(gs[1, 0])
    ax_key_A.axis("off")
    ax_key_A.legend(*handles_A, loc='upper center', frameon=False,
                    fontsize=plt.rcParams['legend.fontsize'], handlelength=1.3,
                    labelspacing=0.35, borderaxespad=0.0)

    # B's key label wraps to 2 lines and runs wide enough to crowd A's key at its shared
    # column boundary -- nudged right of center (bbox_to_anchor x > 0.5) to clear it.
    ax_key_B = fig.add_subplot(gs[1, 1])
    ax_key_B.axis("off")
    ax_key_B.legend(*handles_B, loc='upper center', bbox_to_anchor=(0.62, 1.09),
                    frameon=False, fontsize=plt.rcParams['legend.fontsize'],
                    handlelength=1.3, labelspacing=0.35, borderaxespad=0.0)

    # C's key moves in-axes, at the panel's own bottom-right -- the fit and theory curves
    # saturate along the top and rise through the interior but leave the bottom-right
    # clear, and an in-axes key needs no space reserved for it in the row below at all.
    # Shrunk to 7 pt (below the common LEGEND_PT) since it now sits over the data rather
    # than in reserved whitespace, where the common size would crowd the curves. Pushed
    # further right than the default corner inset (bbox_to_anchor x > 1) to clear the
    # curve's rise, which otherwise runs under the key's left edge.
    ax_C.legend(*handles_C, loc='lower right', bbox_to_anchor=(1.16, 0.0),
               frameon=False, fontsize=7, handlelength=1.3, labelspacing=0.35,
               borderaxespad=0.3)

    # C's insets now span rows 1+2 (C's own key no longer needs row-1 space, having moved
    # in-axes above), grown to fill that combined height so their bottom lands level with
    # A's key bottom instead of stopping at B's key row alone.
    gs_C_insets = gs[1:, 2]

    for ax, letter in zip((ax_A, ax_B, ax_C), "ABC"):
        ax.text(-0.05, 0.95, letter, transform=ax.transAxes + ScaledTranslation(
            -18 / 72, 22 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight='bold',
            va='bottom', ha='left')

    # Square panels, set on the axes box rather than left to the figure aspect: the
    # bottom row (A's and B's keys, C's insets) takes height the figure shape cannot
    # account for, so at a 6.4 in-tall canvas the three panels came out tall rectangles.
    for ax in (ax_A, ax_B, ax_C):
        ax.set_box_aspect(1.0)

    add_insets(fig, gs_C_insets, pc1)


    out = save_figure(fig, figure_dir, "RMTPhenomena")
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    make_figure()

# %%
