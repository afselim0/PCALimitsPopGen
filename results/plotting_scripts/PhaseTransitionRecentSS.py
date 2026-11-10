#%%
"""
Spatial-axis recovery vs migration for the recent-split demography: a panmictic population
A plus a K x K grid B that split off t_split generations ago.

A  n-per-deme-B varied at a fixed 50/50 split between the structured and panmictic halves.
B  phi (B's share of the total sample) varied at n-per-deme-B = 4.

phi here is the *total sample fraction* drawn from the structured demes -- not the
fraction-of-demes-sampled phi in PhaseTransitionRandomSS, which is a different sweep
entirely. Every deme is always sampled here; phi only trades n_A against B's total.

The target axis is the (1,0)/(0,1) spatial pair, located by overlap rather than by rank:
it is not reliably the leading eigenvalue of the full A+B system once A carries a large
share of the sample.

x is 1/M, applied in the analysis rather than here. The fit and the band come from
src/sweeps_analysis/summarise.py -- this script draws only.

Unlike the other overlap figures, this one reports a PEAK rather than a detection crossing.
cos^2 is not monotone in 1/M here: it rises to ~0.93 near 1/M ~ 0.3 and then decays. The
summariser therefore fits these sweeps with an *unconstrained* penalized smooth (mgcv's
"tp") rather than the shape-constrained one used elsewhere -- only smoothness is imposed,
so the maximum is read off wherever the data puts it -- and marks each curve at that
maximum. The target line is kept as a reference level, not as something claimed to be
crossed.

The bottom row reads OBSERVATION -> MECHANISM:

C  how much of the spatial signal successively wider PC windows recover
D  the theoretical spectrum's separation and scale, which is why C looks as it does

C comes from src/analysis/RecentSpatialParticipation.py and is a MEASUREMENT on the
simulated genotypes. With p_k = ||U_inf^T u_k||^2 the squared projection of sample PC k
onto the two-dimensional equilibrium spatial plane, it plots the cumulative
p_r = sum_{k=2..r} p_k for r = 3, 4, 5 and 21, each formed within a replicate before any
averaging and with the PCs kept in EIGENVALUE order (never selected by overlap). The sum
starts at k=2 because PC1 is the A-vs-B mode. Near panel A's maximum all four windows sit
together just under 2 -- two PCs already hold the whole plane -- and on the weak-migration
side they fan apart: the signal is conserved and redistributed over more PCs, not lost.
p_3 is panel A's own measurement doubled, and p_21 is the S that the same analysis reports
as a scalar; both are checked numerically and printed beside the figure.

D comes from the stored phase-type sweep (src/analysis/RecentSpatialSpectrum.py) and adds
no new computation. It carries two facts on twin axes -- the relative gap (nu_spatial -
nu_next)/nu_spatial on the left, nu_spatial itself on the right, log -- for the transient
and equilibrium spectra. Colour is the scenario, line style is the quantity. The transient
eigenvalue RISES (1.30 -> 6.01) while its gap to the next mode COLLAPSES (0.510 -> 0.052),
and the equilibrium gap stays near-flat (0.506 -> 0.554) so the collapse cannot be blamed
on anything generic about a stepping-stone lattice. Converging modes, not a vanishing
spike, is what C measures as the signal spreading over more PCs.

The retired panel showed the full top-5 spectrum with the detection threshold nu_c; its
numbers (including nu_c and the margin nu_spatial/nu_c, which falls 1.47 -> 1.04) stay in
RecentSpatialSpectrum/ for the manuscript text.

The equilibrium reference is the no-split stepping stone, which has no population A, so it
is computed B-only while the transient spectrum uses the full sample (B's demes plus A).
The two are therefore indexed DIFFERENTLY -- the transient spectrum is led by the A-vs-B
mode and its spatial pair sits at ranks 1, 2, while the equilibrium one has no A mode and
its pair leads at 0, 1 -- so the analysis aligns them on the spatial pair rather than
rank-for-rank. The pair is identified by principal cosine against the equilibrium DCT modes
(never by rank) and is degenerate to 3.5e-15, so it is drawn as a single curve.

All of A's marker, C and D use the same cell (phi=0.5, n_B=4), so the vertical dotted line
at panel A's measured maximum applies to all of them.

Reads results/sweep_summaries/split_stepping_stone/{curves,crossings,points}.csv
    (sweeps split_ss_vary_N and split_ss_vary_phi)
    results/analysis/RecentSpatialSpectrum/{spectrum,gaps}.csv
    results/analysis/RecentSpatialParticipation/{participation,summary}.csv
Writes results/figures/PhaseTransitionRecentSS.pdf (and a PNG preview in results/png_figures)
"""
import os

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from matplotlib import cm

from crossing_panel import (reserve_side_legend, PANEL_LETTER_SIZE, figsize, DEFAULT_TARGET, L_title, figure_encoding_legend,
                            load_summary, peak_handles, peak_panel_from_summary,
                            peak_table, save_figure, style_swept_axis, variant_note)
from plotting_style import setup_plotting_style

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_DIR = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries",
                           "split_stepping_stone")
SPECTRUM_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RecentSpatialSpectrum")
PARTICIPATION_DIR = os.path.join(_PROJECT_ROOT, "results", "analysis",
                                 "RecentSpatialParticipation")
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")

# Room under the panels for the shared encoding key.
BOTTOM_MARGIN = 0.20

CMAP = cm.viridis
CMAP_PHI = cm.plasma
TARGET = DEFAULT_TARGET

# Panels C/D. The spatial pair is the quantity the whole figure is about, so it is the one
# saturated colour; every other structural mode is grey, since their individual identities
# do not matter -- only that the pair is sinking into them.
SPATIAL_COLOR = "#c2185b"
SPEC_KW = dict(linewidth=0.9, zorder=4)
# The vertical marker at the measured maximum of the panel-A fit, repeated in C and D.
PEAK_VLINE_KW = dict(color="0.35", linewidth=0.8, linestyle=(0, (1.2, 1.8)), zorder=2)
# Panel D. COLOUR = scenario, LINE STYLE = quantity -- the two are orthogonal so that a
# reader can decode either axis without consulting the other half of the key.
TRANSIENT_COLOR = SPATIAL_COLOR      # the same red the spatial pair carried before
EQUILIBRIUM_COLOR = "black"
GAP_KW = dict(linewidth=0.9, linestyle="-", zorder=4)     # left axis, relative gap
NU_KW = dict(linewidth=0.9, linestyle=(0, (3.5, 2)), zorder=4)   # right axis, eigenvalue

# Panel C's cumulative windows. A SEQUENTIAL ramp, light to dark as the window widens, so
# the nesting (p_3 <= p_5 <= p_10 <= p_21) reads as an ordering rather than four unrelated
# colours. All four are solid measured windows; only the theoretical p_2 line is dashed.
CUM_WINDOWS = (3, 5, 10)
CUM_COLORS = ("#80cbc4", "#26a69a", "#00695c")
CUM_WIDE = 21
CUM_WIDE_COLOR = "#00332e"     # darker than the ramp: the widest MEASURED window
# SOLID, like the rest of the ramp -- it is a measured window, not a reference. Dashed is
# reserved for the theoretical p_2 line, so style alone separates measurement from theory.
CUM_WIDE_KW = dict(linewidth=0.9, linestyle="-", zorder=5)
# p_2: how much of the equilibrium spatial plane the NON-EQUILIBRIUM theoretical pair
# spans. No estimation in it, so it is the ceiling the measured windows approach -- black
# dashed, to read as theory against the measured teal.
THEORY_P2_KW = dict(color="black", linewidth=0.9, linestyle=(0, (4, 2)), zorder=6)
BAND_KW_D = dict(alpha=0.22, linewidth=0, zorder=2)


def split_time(summary_dir=SUMMARY_DIR, default=500.0):
    """
    The split time of the sweeps behind this figure, for the y label.

    It is a property of the SIMULATIONS, so it appears in neither the sweep configs nor the
    summary CSVs the figure reads -- only in the per-replicate tables upstream. Both sweeps
    of this demography share one value; a mismatch would mean the panels are not comparable,
    so that is checked rather than assumed, and the first replicate table that carries the
    column wins. `default` applies only if none is readable.
    """
    found = {}
    for sweep in ("split_ss_vary_N", "split_ss_vary_phi"):
        path = os.path.join(_PROJECT_ROOT, "results", "sweeps", sweep, "replicates.csv")
        try:
            col = pd.read_csv(path, usecols=["t_split"])["t_split"]
        except (FileNotFoundError, ValueError):
            continue
        vals = col.unique()
        if len(vals):
            found[sweep] = float(vals[0])
    if not found:
        return default
    if len(set(found.values())) > 1:
        raise ValueError(f"split_stepping_stone sweeps disagree on t_split: {found}")
    return next(iter(found.values()))


def load_spectrum(spectrum_dir=None):
    """
    (spectrum, gaps) from RecentSpatialSpectrum, or (None, None) if it has not been run.

    Returning None rather than raising keeps the figure buildable from the sweep summaries
    alone: panels C and D are then skipped and A/B are drawn as before, which is what makes
    this script safe to run before the analysis has finished.

    The default is read from the module global at CALL time, not bound as a default
    argument: a default argument is evaluated once at def time, so re-pointing
    SPECTRUM_DIR (which is how the dry run tests these panels) would silently have no
    effect and the panels would be skipped as though the analysis were missing.
    """
    if spectrum_dir is None:
        spectrum_dir = SPECTRUM_DIR
    spec = os.path.join(spectrum_dir, "spectrum.csv")
    gaps = os.path.join(spectrum_dir, "gaps.csv")
    if not (os.path.exists(spec) and os.path.exists(gaps)):
        return None, None
    return pd.read_csv(spec), pd.read_csv(gaps)


def load_participation(participation_dir=None):
    """
    (participation, summary) from RecentSpatialParticipation, or (None, None) if absent.

    Same call-time default and same graceful degradation as load_spectrum: panel D is
    skipped rather than raising, so the figure still builds from whatever has been run.
    """
    if participation_dir is None:
        participation_dir = PARTICIPATION_DIR
    part = os.path.join(participation_dir, "participation.csv")
    summ = os.path.join(participation_dir, "summary.csv")
    ranks = os.path.join(participation_dir, "per_rank_replicates.csv")
    if not (os.path.exists(part) and os.path.exists(summ) and os.path.exists(ranks)):
        return None, None, None
    return pd.read_csv(part), pd.read_csv(summ), pd.read_csv(ranks)


def mechanism_panel(ax, spec, gaps, phi=0.5, n_B=4, peak_x=None):
    """
    Panel D: why panel C's signal spreads -- the spatial pair's separation collapsing while
    its own eigenvalue does not.

    Two facts on twin axes, encoded so that COLOUR = scenario and LINE STYLE = quantity:

        left, linear   relative gap (nu_spatial - nu_next)/nu_spatial
                       transient solid red, equilibrium solid grey
        right, LOG     the spatial eigenvalue nu_spatial itself
                       transient dashed red, equilibrium dashed grey

    The right axis is log because the equilibrium eigenvalue runs to ~121 while the
    transient tops out at ~6; on a linear axis the transient curve would sit flat on the
    bottom spine and the comparison would be unreadable.

    The grey curves do two different jobs, which is why both are drawn. As a GAP they are a
    control: the equilibrium spectrum's relative gap is near-flat (0.506 -> 0.554), so the
    transient collapse (0.510 -> 0.052) cannot be blamed on anything generic about a
    stepping-stone lattice. As an EIGENVALUE they are a ceiling: the divergence between
    dashed red and dashed grey is how much spatial structure the t_split=500 split has not
    had time to accumulate.

    Read together: the transient eigenvalue RISES (1.30 -> 6.01) while its gap to the next
    mode COLLAPSES. Nothing is sinking into the bulk; the modes are converging on each
    other, which is what panel C measures as the signal spreading over more PCs.

    Everything is read from the stored phase-type sweep (RecentSpatialSpectrum), which also
    supplies the numbers quoted in the manuscript text; this panel adds no new computation.
    The spatial pair is identified there by principal cosine against the equilibrium DCT
    modes rather than by rank, and it is exactly degenerate (max relative difference 3.5e-15
    across the sweep), so the pair is drawn as one curve.
    """
    g = gaps[(gaps.n_B == n_B) & (gaps.phi == phi)].sort_values("inv_M")
    sp = spec[(spec.n_B == n_B) & (spec.phi == phi)]
    if g.empty or sp.empty:
        return None

    # The equilibrium spatial eigenvalue. The equilibrium system is B-only and has no
    # A-vs-B mode, so its pair leads at ranks 0-1 while the transient pair sits at 1-2; the
    # analysis already aligned the two on the pair, so the equilibrium value is carried on
    # the is_spatial rows' nu_eq column rather than at a fixed rank here.
    eq_nu = (sp[sp.is_spatial].groupby("inv_M").nu_eq.mean().sort_index())

    ax.plot(g.inv_M, g.gap, color=TRANSIENT_COLOR, **GAP_KW)
    ax.plot(g.inv_M, g.gap_eq, color=EQUILIBRIUM_COLOR, **GAP_KW)

    ax2 = ax.twinx()
    ax2.plot(g.inv_M, g.nu_spatial, color=TRANSIENT_COLOR, **NU_KW)
    ax2.plot(eq_nu.index, eq_nu.values, color=EQUILIBRIUM_COLOR, **NU_KW)

    if peak_x is not None and np.isfinite(peak_x):
        ax.axvline(peak_x, **PEAK_VLINE_KW)

    ax.set_xscale("log")
    ax.set_xlim(g.inv_M.min(), g.inv_M.max())
    ax.set_ylim(0.0, 0.6)
    ax.set_ylabel(r"relative gap  "
                  r"$(\nu_{\mathrm{spatial}}-\nu_{\mathrm{next}})/\nu_{\mathrm{spatial}}$")
    # The repo style turns the right spine off globally (plotting_style sets
    # axes.spines.right = False), which is right for a single-axis panel but leaves this
    # twin axis' ticks and labels floating with no axis line to belong to. Turned back on
    # for BOTH axes objects: the twin draws its own spine, and the underlying axes' right
    # spine sits in the same place, so leaving that one off would show a half-length line.
    ax2.spines["right"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"spatial eigenvalue  $\nu_{\mathrm{spatial}}$")
    return ax2


def cumulative_panel(ax, per_rank_rep, part, peak_x=None, gaps=None,
                     phi=0.5, n_B=4):
    """
    Panel C: cumulative leakage of the spatial plane into successively wider PC windows.

    p_r = sum_{k=2..r} p_k, with p_k = ||U_inf^T u_k||^2 the squared projection of sample PC
    k onto the two-dimensional equilibrium spatial plane. The sum starts at k=2 because PC1
    is the A-vs-B mode and carries no spatial signal (measured p_1 <= 0.009).

    Four measured windows, all solid on one light-to-dark ramp: CUM_WINDOWS (PCs 2-3, 2-5,
    2-10) plus CUM_WIDE (PCs 2-21) as the darkest. The ramp skips 2-4: 2-3, 2-5 and 2-10
    sample the spreading at three genuinely different scales, while 2-4 sat on top of 2-5.
    Dashing is reserved for the black theoretical p_2 line, so line style separates
    measurement from theory and colour alone orders the windows. They are NESTED by construction, so p_3 <= p_4 <= p_5 <= p_21 everywhere;
    what the panel shows is WHERE they separate. Near panel A's maximum all four sit
    together just under 2 -- two PCs already hold the whole plane -- and on the
    weak-migration side they fan apart, which is the signal spreading over more PCs.

    This replaces the S / r_eff twin-axis panel. Same measurement, but a reader can see the
    spreading without first being told what a participation ratio is; S and r_eff are still
    computed and written to RecentSpatialParticipation/ for the manuscript text (p_21 IS
    that S, at k_max = 20).

    Every curve is formed per replicate and only then reduced, so the bands are the genuine
    replicate spread rather than the spread of an average.

    The horizontal reference is at 2 = the whole plane. It is labelled as all the signal,
    not as a maximum or ceiling: at finite L no window attains it, and the distance to it is
    the part of the plane PCA never recovers.
    """
    if per_rank_rep is None or per_rank_rep.empty:
        return

    cum = _cumulative(per_rank_rep)
    ax.axhline(2.0, color="0.6", linewidth=0.6, linestyle=":", zorder=1)

    for r, color in zip(CUM_WINDOWS, CUM_COLORS):
        g = cum[cum.r == r].sort_values("inv_M")
        ax.fill_between(g.inv_M, g.lo, g.hi, color=color, **BAND_KW_D)
        ax.plot(g.inv_M, g.med, color=color, label=f"PCs 2-{r}", **SPEC_KW)

    g = cum[cum.r == CUM_WIDE].sort_values("inv_M")
    # The widest window gets its IQR band too. It is the narrowest of the four (0.0099 at
    # the tail against p_3's 0.3922) but not negligible -- at the strong-migration edge it
    # is 0.024, wider than p_10's -- and omitting it would imply the reference curve was
    # measured without error while the others were.
    ax.fill_between(g.inv_M, g.lo, g.hi, color=CUM_WIDE_COLOR, **BAND_KW_D)
    ax.plot(g.inv_M, g.med, color=CUM_WIDE_COLOR, label=f"PCs 2-{CUM_WIDE}", **CUM_WIDE_KW)

    if gaps is not None and "pair_p2" in gaps.columns:
        gp = gaps[(gaps.n_B == n_B) & (gaps.phi == phi)].sort_values("inv_M")
        if not gp.empty and gp.pair_p2.notna().any():
            ax.plot(gp.inv_M, gp.pair_p2, label=r"theoretical pair $p_2$",
                    **THEORY_P2_KW)

    if peak_x is not None and np.isfinite(peak_x):
        ax.axvline(peak_x, **PEAK_VLINE_KW)

    ax.set_xscale("log")
    ax.set_xlim(cum.inv_M.min(), cum.inv_M.max())
    ax.set_ylim(0.0, 2.05)
    ax.set_ylabel(r"cumulative spatial signal $S_r$")


def _cumulative(per_rank_rep, windows=None):
    """
    Median and IQR of p_r = sum_{k=2..r} p_k, formed PER REPLICATE then reduced.

    Returns tidy rows (inv_M, r, med, lo, hi). Reducing first and summing after would
    report the spread of an average rather than the spread across replicates, which is
    exactly the quantity the weak-migration end is evidence about.
    """
    if windows is None:
        windows = list(CUM_WINDOWS) + [CUM_WIDE]
    d = per_rank_rep[per_rank_rep.k >= 2]
    out = []
    for r in windows:
        sub = d[d.k <= r].groupby(["inv_M", "replicate"]).p_k.sum().reset_index()
        g = sub.groupby("inv_M").p_k
        out.append(pd.DataFrame({"inv_M": g.median().index, "r": r,
                                 "med": g.median().values,
                                 "lo": g.quantile(0.25).values,
                                 "hi": g.quantile(0.75).values}))
    return pd.concat(out, ignore_index=True)


def spectrum_handles(t_split=500.0):
    """
    Key for panels C and D.

    Panel C's four curves are NESTED windows, so they are keyed as a light-to-dark ramp in
    window order; the widest is faint and dashed because it is a reference rather than a
    fourth member of the ramp.

    Panel D's four curves are an OUTER PRODUCT -- two scenarios x two quantities -- so they
    are keyed as two statements about style rather than four separate lines: colour names
    the scenario, line style names the quantity and its axis. Listing all four individually
    would spend four entries to say something two say better.
    """
    # figure_encoding_legend fills COLUMN-MAJOR, one column under each panel, so the order
    # here decides which panel each entry sits under: C's 4 windows left, D's 4 right. The
    # count must stay even, or the split moves an entry into the wrong panel's column.
    return [
        # left column: panel C -- the theoretical ceiling first, then the nested measured
        # windows it bounds, light to dark
        Line2D([], [], **THEORY_P2_KW, label="theoretical eigenvectors"),
        *[Line2D([], [], color=c, **SPEC_KW, label=f"PCs 2-{r}")
          for r, c in zip(CUM_WINDOWS, CUM_COLORS)],
        Line2D([], [], color=CUM_WIDE_COLOR, **CUM_WIDE_KW,
               label=f"PCs 2-{CUM_WIDE}"),
        # right column: panel D -- quantity (style) above scenario (colour)
        Line2D([], [], color="0.35", **GAP_KW,
               label="relative gap to next mode (left axis)"),
        Line2D([], [], color="0.35", **NU_KW, label="spatial eigenvalue (right axis)"),
        Line2D([], [], color=TRANSIENT_COLOR, linewidth=2.2,
               label=rf"non-equilibrium stepping-stone (split $t={t_split:g}$)"),
        Line2D([], [], color=EQUILIBRIUM_COLOR, linewidth=2.2, label="equilibrium stepping-stone"),
    ]


def make_figure(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET,
                standardized=False, spectrum_dir=None, participation_dir=None):
    setup_plotting_style()
    # This demography publishes no theory, so the theory frame is deliberately
    # unused: these panels show the measurement alone.
    curves, crossings, points, _ = load_summary(summary_dir, with_theory=False)
    t_split = split_time(summary_dir)
    spec, gaps = load_spectrum(spectrum_dir)
    part, part_summary, per_rank_rep = load_participation(participation_dir)
    has_spectrum = spec is not None

    # 2 x 2 when the theoretical spectrum is available, the original single row otherwise,
    # so this script still runs before RecentSpatialSpectrum has been executed.
    if has_spectrum:
        fig, grid = plt.subplots(2, 2, figsize=figsize(6.35, 5.95))
        axes = grid[0]
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize(6.5, 4.35))
        grid = None

    res_n = peak_panel_from_summary(
        axes[0], curves, crossings, points, sweep="split_ss_vary_N",
        cmap=CMAP, cbar_label=r"$n_{B,\mathrm{per\,deme}}$", target=target, standardized=standardized)
    res_phi = peak_panel_from_summary(
        axes[1], curves, crossings, points, sweep="split_ss_vary_phi",
        cmap=CMAP_PHI, cbar_label=r"$\phi$", target=target, standardized=standardized)

    for ax in axes:
        style_swept_axis(ax, r"$1/M$", tick_row=False)
        # Name both vectors, as DemoRecentSS does. crossing_panel's shared default is the
        # generic <u_hat, u>^2; here both sides matter -- the OBSERVED axis of a grid
        # sampled t_split generations after the split, against the EQUILIBRIUM
        # stepping-stone pair that stands in for geographic position (the target axis is
        # equilibrium_B_axes; only nu stays on the transient solve). t_split is read from
        # the sweep rather than written as a literal.
        ax.set_ylabel(r"cosine similarity  "
                      rf"$\langle \hat u_{{t={t_split:g}}}, u_\infty\rangle^2$")
    axes[0].set_title(L_title("split_ss_vary_N", r"fixed $\phi=0.5$"))
    axes[1].set_title(L_title("split_ss_vary_phi",
                              r"fixed $n_{B,\mathrm{per\,deme}}=4$"))

    for ax in axes:
        variant_note(ax, standardized)

    # The vertical marker C and D carry: panel A's own measured maximum, for the n_B = 4
    # curve, which is the sampling C's spectrum is computed for.
    peak_x = next((r["peak_x"] for r in res_n if np.isclose(r["level"], 4.0)), np.nan)

    all_axes = list(axes)
    if has_spectrum:
        # C then D: the MEASUREMENT first, then the theory that explains it, so the row
        # reads observation -> mechanism.
        cumulative_panel(grid[1][0], per_rank_rep, part, peak_x=peak_x, gaps=gaps)
        mechanism_panel(grid[1][1], spec, gaps, phi=0.5, n_B=4, peak_x=peak_x)
        for ax in grid[1]:
            ax.set_xlabel(r"$1/M$")

        # C and D carry NO titles: their y labels already name what each measures, and
        # A/B's titles state the fixed cell that C and D share.
        all_axes += list(grid[1])

    for ax, letter in zip(all_axes, "ABCD"):
        # Offset in POINTS, not axes fraction: at 12 pt the fractional offset put the
        # letter on the "using L_eff" row and the panel title.
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + ScaledTranslation(
            -34 / 72, 26 / 72, fig.dpi_scale_trans),
            fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="top")

    # tight_layout knows nothing about the figure-level key, so the bottom margin is
    # opened for it after the fact; wspace holds panel B's y-axis label clear of A.
    # reserve_side_legend, not a bare wspace: the per-panel keys are anchored OUTSIDE
    # the axes, so tight_layout cannot see them and lays the axes across the full
    # width -- the key then lands on the neighbouring panel. This reserves the
    # legend's width in inches so each panel keeps its own square.
    if has_spectrum:
        # reserve_side_legend lays out a SINGLE row: it opens one bottom margin for one
        # legend row and re-anchors every curve key to it. Calling it here and then
        # patching subplots_adjust put row 1's keys across row 2's titles, so the two-row
        # case gets its own explicit layout instead -- generous hspace for the keys that
        # sit between the rows, and a bottom margin for the shared encoding key.
        # bottom leaves room for the shared key's four rows (8 handles in 2 columns).
        # right is pulled in from 0.98 to leave room for panel D's twin y axis, which
        # tight-bbox would otherwise push the saved figure past the 6.5 in page cap.
        # hspace only has to clear row 1's x label and its curve key now that C and D
        # carry no titles; it was 0.62 when it also had to clear two title lines.
        # Geometry chosen so the four AXES come out square (2.20 x 2.20 in): with two
        # rows each axes gets (top - bottom)/(2 + hspace) of the canvas, so these margins
        # and figsize(.., 6.13) solve to a 1:1 box. hspace still has to clear row 1's
        # curve key, which is what stops it going below ~0.38.
        fig.subplots_adjust(left=0.09, right=0.915, top=0.975, bottom=0.12,
                            wspace=0.38, hspace=0.38)
        _place_row_legends(fig, axes)
        # ncol=2, not 3: figure_encoding_legend lays out ONE COLUMN PER PANEL, aligned to
        # the panels' own x positions, so on a two-panel row a third column has no panel to
        # start under and is drawn on top of the second.
        # peak_handles' two entries ("smoothing spline fit", "maximum cosine similarity")
        # are deliberately NOT included: they describe panels A and B, which carry their
        # own curve keys, and in this figure they sat above C's and D's entries as though
        # they belonged to them.
        # y is passed explicitly rather than measured. figure_encoding_legend's default
        # places the key just under the lowest artist, which here put its bottom at
        # y = -0.116 -- a tenth of the canvas BELOW it -- and bbox_inches="tight" then grew
        # the saved figure by ~0.8 in to contain it. Anchoring it inside the reserved
        # bottom margin keeps the page short without touching the shared helper, which
        # every other sweep figure relies on. It must still sit BELOW row 2's x labels:
        # at y=0.105 the key was drawn straight through them.
        figure_encoding_legend(fig, spectrum_handles(t_split), ncol=2, y=0.055)
    else:
        reserve_side_legend(fig, n_panels=2, bottom=BOTTOM_MARGIN)
        figure_encoding_legend(fig, peak_handles(target=target), ncol=2)
    out = save_figure(fig, figure_dir, "PhaseTransitionRecentSS", standardized)
    print(f"wrote {out}")
    print(peak_table(res_n, "n_B", "1/M"))
    print(peak_table(res_phi, "phi", "1/M"))
    if has_spectrum:
        print(_spectrum_table(spec, gaps, peak_x))
    if per_rank_rep is not None:
        print(_cumulative_table(per_rank_rep, part, curves, peak_x))
    return fig


def _cumulative_table(per_rank_rep, part, curves, peak_x):
    """
    Panel C's numbers and the four checks that make it interpretable.

    Checks 1 and 2 are external consistency: p_3 must reproduce panel A's own measurement
    (which is the same two PCs, scaled by 2), and p_21 must reproduce the S already
    computed at k_max = 20. Check 3 separates real spreading from the free growth any wider
    window buys. Check 4 is the replicate spread, which is itself part of the evidence.
    """
    cum = _cumulative(per_rank_rep)
    xs = np.sort(cum.inv_M.unique())
    lo_x, hi_x = xs[0], xs[-1]
    mid_x = xs[int(np.argmin(np.abs(xs - peak_x)))]
    windows = list(CUM_WINDOWS) + [CUM_WIDE]

    def val(r, x):
        g = cum[(cum.r == r) & (cum.inv_M == x)]
        return float(g.med.iloc[0]) if len(g) else np.nan

    lines = ["\npanel C: cumulative spatial signal p_r (median over replicates)",
             f"{'1/M':>9}  " + "  ".join(f"{'p_'+str(r):>7}" for r in windows)]
    for label, x in (("left edge", lo_x), ("A maximum", mid_x), ("right edge", hi_x)):
        lines.append(f"{x:>9.4g}  "
                     + "  ".join(f"{val(r, x):>7.4f}" for r in windows)
                     + f"   ({label})")

    # --- check 1: p_3 against panel A's n_B = 4 curve, doubled ---------------
    ca = curves[(curves.sweep == "split_ss_vary_N") & (curves.level == 4.0)].sort_values("x")
    p3 = cum[cum.r == 3].sort_values("inv_M")
    a_interp = np.interp(np.log(p3.inv_M.values), np.log(ca.x.values), ca.y_fit.values) * 2.0
    d1 = np.abs(p3.med.values - a_interp)
    lines.append(f"\n  1. p_3 vs 2 x panel A (n_B=4): max |diff| = {d1.max():.4f} "
                 f"(at 1/M = {p3.inv_M.values[int(np.argmax(d1))]:.4g}); "
                 f"at the A maximum {np.abs(val(3, mid_x) - 2*np.interp(np.log(mid_x), np.log(ca.x.values), ca.y_fit.values)):.4f}")

    # --- check 2: the widest window against the stored S ---------------------
    # Compared at the SAME window. p_21 sums k = 2..21 while S_k20 sums k = 2..20, so they
    # differ by exactly the k=21 term (~5e-3 at its largest) and are not the same quantity;
    # matching them at k <= 20 is what actually tests that the two routes agree.
    S = part.groupby("inv_M").S_k20.median()
    cum20 = _cumulative(per_rank_rep, windows=[20])
    p20 = cum20.set_index("inv_M").med
    common = S.index.intersection(p20.index)
    d2 = np.abs(S.loc[common].values - p20.loc[common].values)
    p21 = cum[cum.r == CUM_WIDE].set_index("inv_M").med
    lines.append(f"  2. cumulative k<=20 vs stored S (k_max=20): max |diff| = {d2.max():.3e} "
                 f"over {len(common)} M values")
    lines.append(f"     (p_21 itself exceeds S by the k=21 term: max "
                 f"{np.abs(p21.loc[common].values - S.loc[common].values).max():.3e})")

    # --- check 3: free growth from widening ---------------------------------
    N = int(part.N.iloc[0])
    free = 2.0 / (N - 1)
    n_extra = CUM_WINDOWS[-1] - CUM_WINDOWS[0]
    lines.append(f"  3. free growth per extra PC ~ 2/(N-1) = {free:.5f} (N={N}); "
                 f"the {n_extra} extra PCs between the narrowest and widest solid window "
                 f"buy {n_extra*free:.5f}")
    narrow, wide = CUM_WINDOWS[0], CUM_WINDOWS[-1]
    lines.append(f"     measured p_{wide} - p_{narrow}: "
                 f"{val(wide, mid_x) - val(narrow, mid_x):+.4f} at the A maximum, "
                 f"{val(wide, hi_x) - val(narrow, hi_x):+.4f} at the tail")

    # --- check 4: replicate spread ------------------------------------------
    d = per_rank_rep[per_rank_rep.k >= 2]
    lines.append("  4. replicate std:")
    for label, x in (("A maximum", mid_x), ("tail", hi_x)):
        parts = []
        for r in (3, CUM_WIDE):
            sub = d[(d.inv_M == x) & (d.k <= r)].groupby("replicate").p_k.sum()
            parts.append(f"p_{r} {sub.std():.4f}")
        lines.append(f"     {label:>10}: " + ", ".join(parts))

    # --- the fanning claim, asserted ----------------------------------------
    # Nesting is checked over whatever windows are actually drawn, at EVERY x rather than
    # only at the tail: they are nested by construction, so a violation anywhere would mean
    # the cumulative sums had been built wrong.
    fan = val(wide, hi_x) - val(narrow, hi_x)
    seq = list(CUM_WINDOWS) + [CUM_WIDE]
    ordered = all(val(a, x) <= val(b, x) + 1e-12
                  for x in xs for a, b in zip(seq, seq[1:]))
    lines.append(f"  curves fan apart at the tail: p_{wide} - p_{narrow} = {fan:+.4f}, "
                 f"nesting holds at all {len(xs)} x: {ordered}")
    return "\n".join(lines)


def _place_row_legends(fig, row1):
    """
    Anchor each row-1 curve key just under its own panel, clear of row 2's titles.

    peak_panel_from_summary creates these keys for a one-row figure, where they end up at
    the page bottom; in a two-row figure that is on top of panels C and D. The offset is in
    AXES fraction and is what the hspace in make_figure is sized to clear.
    """
    for ax in row1:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((0.5, -0.235), transform=ax.transAxes)
            leg._loc = 9          # upper center of that anchor


def _spectrum_table(spec, gaps, peak_x, phi=0.5, n_B=4):
    """
    Panel D's numbers, plus the assertions the panel depends on.

    Three x positions (left edge, panel-A maximum, right edge) x two scenarios, then the
    checks that make the panel legitimate: the pair really is degenerate, and it really is
    at ranks 1-2 rather than wherever the overlap search happened to land.
    """
    g = gaps[(gaps.n_B == n_B) & (gaps.phi == phi)].sort_values("inv_M")
    sp = spec[(spec.n_B == n_B) & (spec.phi == phi)]
    eq_nu = sp[sp.is_spatial].groupby("inv_M").nu_eq.mean().sort_index()

    def at(x):
        i = int(np.argmin(np.abs(g.inv_M.values - x)))
        row = g.iloc[i]
        return row, float(eq_nu.iloc[int(np.argmin(np.abs(eq_nu.index.values - x)))])

    lines = [f"\npanel D (phi={phi:g}, n_B={n_B}): gap and nu_spatial",
             f"{'1/M':>9}  {'gap tr':>8} {'gap eq':>8}  "
             f"{'nu_sp tr':>9} {'nu_sp eq':>9}"]
    for label, x in (("left edge", g.inv_M.min()),
                     ("A maximum", peak_x),
                     ("right edge", g.inv_M.max())):
        row, nu_eq = at(x)
        lines.append(f"{row.inv_M:>9.4g}  {row.gap:>8.4f} {row.gap_eq:>8.4f}  "
                     f"{row.nu_spatial:>9.4f} {nu_eq:>9.4f}   ({label})")

    # Degeneracy of the transient pair: nu_2 vs nu_3 on the is_spatial rows.
    pair = sp[sp.is_spatial].pivot(index="inv_M", columns="mode", values="nu")
    cols = list(pair.columns)
    rel = ((pair[cols[0]] - pair[cols[1]]).abs() / pair[cols[0]]).max()
    lines.append(f"  spatial pair degeneracy: max rel |nu_a - nu_b| = {rel:.3e} "
                 f"(drawn as one curve)")

    at_12 = ((g.pair_i == 1) & (g.pair_j == 2)).mean()
    lines.append(f"  spatial pair at ranks (1,2): {100*at_12:.1f}% of M values"
                 + ("" if at_12 == 1.0
                    else f"  [OTHER RANKS: {sorted(set(zip(g.pair_i, g.pair_j)))}]"))
    lines.append(f"  nu_next is the largest structural mode below the pair, "
                 f"A-vs-B mode excluded")
    return "\n".join(lines)


def make_all(summary_dir=SUMMARY_DIR, figure_dir=FIGURE_DIR, target=TARGET,
             spectrum_dir=None, participation_dir=None):
    """Both versions: the ordinary GRM and the standardized one, same theory marks."""
    return [make_figure(summary_dir, figure_dir, target, standardized=s,
                        spectrum_dir=spectrum_dir,
                        participation_dir=participation_dir)
            for s in (False, True)]


if __name__ == "__main__":
    make_all()

# %%
