#%%
"""
Simulate binary genotype matrices for a two-population split, specified by target
population-genetic statistics rather than raw demographic parameters.

Demography (K = 2):
    - ancestral population of size Ne_ancestral
    - at t_split generations ago it splits into two populations
    - P1 stays at constant size Ne_ancestral (alpha_1 = 0)
    - P2 starts at Ne_ancestral and grows exponentially to a present-day size
      Ne_recent, so its growth rate is

          alpha_2 = log(Ne_recent / Ne_ancestral) / t_split

      (msprime's exponential growth convention: N(t) = N(0) * exp(-alpha * t)
       going *backwards* in time, so N(t_split) = Ne_recent * exp(-alpha_2 * t_split)
       = Ne_ancestral, which is continuity at the split.)

With Ne_ancestral fixed the demography has exactly two free parameters, and
(Fst, T2/T1) is an equivalent, more interpretable coordinate system for them:
Fst measures differentiation, T2/T1 measures how much the growing population's
within-population coalescent time differs from the constant one's. This script
takes targets in those coordinates and inverts to (t_split, Ne_recent).

The sweep is one-dimensional in Fst: --num-Fst log-spaced values from the feasible
floor (~5e-4 at the default bounds, where t_split >= 10 binds) up to --max-Fst.
T2/T1 is a *free* parameter, not a swept one -- each Fst is realizable by a whole
family of ratios, so the script simply takes any one whose (t_split, Ne_recent) fall
inside the plausibility windows, preferring T2/T1 = 1 (P2 held at Ne_ancestral, so
Fst is carried entirely by t_split). The ratio actually used is recorded per row.

Haplotypes come from DemographySpectrum.simulate_haplotypes, which simulates L
independent trees and places exactly one mutation per tree on a branch chosen with
probability proportional to its length -- i.e. L independent biallelic SNPs.

Example:
    python simulate_generic_split.py --num-Fst 500 --max-Fst 0.1 --num-threads 4

Outputs (results/simulations/K2_simulations by default):
    simulation_parameters.csv   one row per simulation
    <simID>.npy                 the 2n x L genotype matrix, saved as np.uint8
"""
import argparse
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import msprime
from scipy.special import exp1, expi
from scipy.optimize import brentq
from tqdm import tqdm

import analysis.demography_spectrum as demography_spectrum_module
from analysis.demography_spectrum import DemographySpectrum

DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "simulations", "K2_simulations",
)

DEFAULT_NUM_FST = 500
DEFAULT_MAX_FST = 0.1
DEFAULT_T_SPLIT_BOUNDS = (1e1, 1e4)
DEFAULT_NE_RECENT_BOUNDS = (1e2, 1e5)

# T2/T1 is a free parameter: for each target Fst any ratio reproducing it with
# in-bounds (t_split, Ne_recent) will do, and the search below starts here.
# Ratio 1 is the natural preference -- it means P2 is held at Ne_ancestral, so
# Ne_recent == Ne_ancestral and only t_split moves with Fst -- and it turns out to
# be feasible across the whole default Fst range.
PREFERRED_RATIO = 1.0
RATIO_SEARCH_BOUNDS = (0.2, 2.5)


# ---------------------------------------------------------------------------
# Demography
# ---------------------------------------------------------------------------

def growth_rate(Ne_present, Ne_ancestral, t_split):
    """
    Exponential growth rate taking Ne_ancestral at the split to Ne_present today.

    N(t) = Ne_present * exp(-alpha * t) backwards in time, so requiring
    N(t_split) == Ne_ancestral gives alpha = log(Ne_present/Ne_ancestral)/t_split.
    """
    return np.log(Ne_present / Ne_ancestral) / t_split


def build_demography(t_split, Ne_recent, Ne_ancestral):
    """
    Two populations splitting from ANC at t_split. P1 holds constant at
    Ne_ancestral; P2 grows exponentially from Ne_ancestral to Ne_recent.
    """
    alpha_1 = 0.0
    alpha_2 = growth_rate(Ne_recent, Ne_ancestral, t_split)

    # default_sampling_time is set explicitly on the derived populations: msprime
    # leaves it as None until simulation time, and DemographySpectrum selects the
    # populations to sample with `default_sampling_time == 0`, which would otherwise
    # match nothing and make sim_ancestry raise "Zero samples specified".
    demography = msprime.Demography()
    demography.add_population(name="ANC", initial_size=Ne_ancestral)
    demography.add_population(name="P1", initial_size=Ne_ancestral, growth_rate=alpha_1,
                              default_sampling_time=0)
    demography.add_population(name="P2", initial_size=Ne_recent, growth_rate=alpha_2,
                              default_sampling_time=0)
    demography.add_population_split(time=t_split, derived=["P1", "P2"], ancestral="ANC")
    return demography, alpha_1, alpha_2


# ---------------------------------------------------------------------------
# Exact coalescent times
# ---------------------------------------------------------------------------

def _E1_general(z):
    """
    Exponential integral E1(z) for real z != 0, extended to negative z
    via E1(z) = -Ei(-z). scipy.special.exp1 only covers z > 0 cleanly.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 0:
        return exp1(z) if z > 0 else -expi(-z)
    out = np.empty_like(z)
    pos = z > 0
    out[pos] = exp1(z[pos])
    out[~pos] = -expi(-z[~pos])
    return out


def mean_coal_time_exp_growth(N_recent, N_ancestral, t_split):
    """
    Mean pairwise coalescence time for a single population whose size,
    going backward in time from the present (t=0), follows:

        N(t) = N_recent * exp(-alpha * t)   for 0 <= t <= t_split
        N(t) = N_ancestral                   for t > t_split

    with alpha = (1/t_split) * ln(N_recent / N_ancestral), chosen so that
    N(0) = N_recent and N(t_split) = N_ancestral exactly. Handles growth
    (N_recent > N_ancestral), decline (N_recent < N_ancestral), and the
    degenerate no-change case.

    N_recent, N_ancestral should already be in coalescent-rate units,
    i.e. pass 2*Ne_recent and 2*Ne_ancestral for a diploid population.

    Returns
    -------
    float : E[T], the mean pairwise coalescence time in generations.
    """
    if N_recent <= 0 or N_ancestral <= 0 or t_split <= 0:
        raise ValueError("N_recent, N_ancestral, and t_split must be positive.")

    if np.isclose(N_recent, N_ancestral):
        return float(N_ancestral)  # constant-size result: E[T] = N

    alpha = (1.0 / t_split) * np.log(N_recent / N_ancestral)
    beta = 1.0 / (alpha * N_recent)
    C = (1.0 / alpha) * (1.0 / N_ancestral - 1.0 / N_recent)

    growth_term = (np.exp(beta) / alpha) * (
        _E1_general(beta) - _E1_general(beta * N_recent / N_ancestral)
    )
    ancestral_term = N_ancestral * np.exp(-C)

    return float(growth_term + ancestral_term)


def coalescent_times(t_split, Ne_recent, Ne_ancestral):
    """
    Exact (T1, T2, T12) mean pairwise coalescent times for this demography.

        T1  = 2 * Ne_ancestral
              two lineages in a constant-size diploid population of size Ne
              coalesce at rate 1/(2 Ne), so E[T] = 2 Ne.

        T12 = t_split + 2 * Ne_ancestral
              lineages in different populations cannot coalesce before they meet in
              the ancestral population at t_split, after which they wait 2 Ne_anc.

        T2  = mean_coal_time_exp_growth(2 Ne_recent, 2 Ne_ancestral, t_split)
              exact expression in terms of the exponential integral E1.

    This replaces msprime's DemographyDebugger.mean_coalescence_time, which is an
    iterative numerical integration: it costs ~0.4 s per call, converges to a value
    ~17-31 generations below the exact answer, and fails outright ("Did not converge
    on an adequate discretisation") for strong contraction.
    """
    T1 = 2.0 * Ne_ancestral
    T12 = t_split + 2.0 * Ne_ancestral
    # sizes in coalescent-rate units: 2*Ne for a diploid population
    T2 = mean_coal_time_exp_growth(2.0 * Ne_recent, 2.0 * Ne_ancestral, t_split)
    return T1, T2, T12


def summary_statistics(T1, T2, T12):
    """Hudson's Fst and the F_2 statistic from mean coalescent times."""
    Fst = 1 - (T1 + T2) / (2 * T12)
    F2 = T12 - (T1 + T2) / 2
    return Fst, F2


# ---------------------------------------------------------------------------
# Inverse design: (Fst, T2/T1) -> (t_split, Ne_recent)
# ---------------------------------------------------------------------------

def invert_mean_coal_time(T_target, Ne_ancestral, t_split,
                          Ne_bracket=None, xtol=1e-10):
    """
    Invert mean_coal_time_exp_growth for Ne_recent: find the present-day size that
    yields a prescribed mean pairwise coalescence time T_target, given t_split and
    Ne_ancestral.

    There is no closed form -- Ne_recent appears inside the E1 arguments, inside
    alpha, and in the exponents, and E1 has no elementary inverse -- but E[T2] is
    strictly increasing in Ne_recent, so a 1-D bisection is robust and exact to
    machine precision.

    The attainable range is 0 < T_target < t_split + 2*Ne_ancestral:
      - Ne_recent -> 0    gives E[T2] -> 0 (instant coalescence in a vanishing pop)
      - Ne_recent -> inf  gives E[T2] -> t_split + 2*Ne_ancestral, since lineages in
        an arbitrarily large recent population never coalesce before the split and
        then wait 2*Ne_ancestral, which is exactly T12.
    A T_target outside that interval is unattainable and raises ValueError.

    Returns:
        float: Ne_recent (diploid) achieving E[T2] == T_target.
    """
    T_max = t_split + 2.0 * Ne_ancestral
    if not (0.0 < T_target < T_max):
        raise ValueError(
            f"T_target={T_target:g} is unattainable: E[T2] is confined to "
            f"(0, t_split + 2*Ne_ancestral) = (0, {T_max:g})."
        )

    def residual(Ne_recent):
        return mean_coal_time_exp_growth(
            2.0 * Ne_recent, 2.0 * Ne_ancestral, t_split) - T_target

    if Ne_bracket is not None:
        lo, hi = Ne_bracket
    else:
        # Expand the high end until residual(hi) > 0. As T_target approaches its
        # supremum the required Ne_recent diverges, so a fixed upper bound would
        # fail for targets close to that limit.
        hi = max(Ne_ancestral, 1.0)
        for _ in range(400):
            r_hi = residual(hi)
            if np.isfinite(r_hi) and r_hi > 0:
                break
            hi *= 4.0
        else:
            raise ValueError(
                f"could not bracket the root from above for T_target={T_target:g}; "
                f"it may be too close to the supremum {T_max:g}."
            )

        lo = Ne_ancestral
        r_lo = residual(lo)
        if np.isfinite(r_lo) and r_lo == 0.0:
            return float(lo)  # the starting point is already the exact root
        # Halve until residual(lo) < 0. The forward model overflows to nan for very
        # small Ne_recent, so keep the last finite negative point rather than
        # stepping past it; E[T2] -> 0 monotonically, so such a point always exists.
        last_good = None
        for _ in range(200):
            if np.isfinite(r_lo) and r_lo < 0:
                last_good = lo
                break
            lo /= 2.0
            r_lo = residual(lo)
        if last_good is None:
            raise ValueError(
                f"could not bracket the root for T_target={T_target:g} "
                f"(t_split={t_split:g}, Ne_ancestral={Ne_ancestral:g})."
            )
        lo = last_good

    f_lo, f_hi = residual(lo), residual(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        raise ValueError(
            f"forward model is not finite at the bracket ends "
            f"({lo:g}, {hi:g}): residuals {f_lo:g}, {f_hi:g}."
        )
    if f_lo > 0 or f_hi < 0:
        raise ValueError(
            f"bracket ({lo:g}, {hi:g}) does not contain the root for "
            f"T_target={T_target:g} (residuals {f_lo:g}, {f_hi:g}); widen it."
        )

    # rtol keeps the result accurate for large Ne_recent, where an absolute xtol
    # alone would stop the solver early (the residual is very flat there).
    return float(brentq(residual, lo, hi, xtol=xtol, rtol=8.9e-16))


def params_from_Fst_and_ratio(Fst, T2_over_T1, Ne_ancestral):
    """
    Find the (t_split, Ne_recent) reproducing a prescribed Fst and coalescent time
    ratio T2/T1, at fixed Ne_ancestral.

    The inversion is mostly algebraic -- T2/T1 pins T2, which makes Fst linear in
    T12, which gives t_split -- so only the final step needs a numerical root-find:

        T1        = 2 * Ne_ancestral                     (P1 is held constant)
        T2        = (T2/T1) * T1
        T12       = (T1 + T2) / (2 * (1 - Fst))          from Fst = 1-(T1+T2)/(2 T12)
        t_split   = T12 - 2 * Ne_ancestral               from T12 = t_split + 2 Ne_anc
        Ne_recent = invert_mean_coal_time(T2, Ne_ancestral, t_split)

    Not every (Fst, T2/T1) pair is realizable; two constraints bound it from below
    and together give

        Fst > |1 - T2/T1| / 2.

      - Contraction (T2/T1 < 1): t_split > 0 requires Fst > (1 - T2/T1)/2.
      - Expansion (T2/T1 > 1): E[T2] cannot exceed t_split + 2*Ne_ancestral, so
        reaching T2 > T1 needs a split old enough to accommodate it, which in turn
        forces Fst > (T2/T1 - 1)/2.

    Requests below the floor raise ValueError rather than returning a spurious answer.

    Returns:
        (t_split, Ne_recent) reproducing the targets.
    """
    if not (0.0 < Fst < 1.0):
        raise ValueError(f"Fst must lie in (0, 1), got {Fst:g}")
    if T2_over_T1 <= 0.0:
        raise ValueError(f"T2/T1 must be positive, got {T2_over_T1:g}")

    T1 = 2.0 * Ne_ancestral
    T2 = T2_over_T1 * T1
    T12 = (T1 + T2) / (2.0 * (1.0 - Fst))
    t_split = T12 - 2.0 * Ne_ancestral

    floor = abs(1.0 - T2_over_T1) / 2.0
    if t_split <= 0.0 or T2 >= t_split + 2.0 * Ne_ancestral:
        raise ValueError(
            f"Fst={Fst:g} is unattainable at T2/T1={T2_over_T1:g}: "
            f"need Fst > |1 - T2/T1|/2 = {floor:g} for this ratio."
        )

    Ne_recent = invert_mean_coal_time(T2, Ne_ancestral, t_split)
    return t_split, Ne_recent


def why_infeasible(Fst, T2_over_T1, Ne_ancestral,
                   t_split_bounds=DEFAULT_T_SPLIT_BOUNDS,
                   Ne_recent_bounds=DEFAULT_NE_RECENT_BOUNDS):
    """
    None if this (Fst, T2/T1) maps to in-bounds parameters, else a label naming the
    violated constraint. Bundles the existence floor with the two plausibility
    windows so callers can test a candidate with a single call.
    """
    t_lo, t_hi = t_split_bounds
    Ne_lo, Ne_hi = Ne_recent_bounds
    try:
        t_split, Ne_recent = params_from_Fst_and_ratio(Fst, T2_over_T1, Ne_ancestral)
    except ValueError:
        return "existence floor"
    if t_split < t_lo:
        return "t_split too small"
    if t_split > t_hi:
        return "t_split too large"
    if Ne_recent < Ne_lo:
        return "Ne_recent too small"
    if Ne_recent > Ne_hi:
        return "Ne_recent too large"
    return None


def find_ratio_for_Fst(Fst, Ne_ancestral,
                       t_split_bounds=DEFAULT_T_SPLIT_BOUNDS,
                       Ne_recent_bounds=DEFAULT_NE_RECENT_BOUNDS,
                       preferred=PREFERRED_RATIO,
                       ratio_bounds=RATIO_SEARCH_BOUNDS, n_scan=401):
    """
    Find any T2/T1 that reproduces this Fst with in-bounds (t_split, Ne_recent).

    T2/T1 is a free parameter here, so a target Fst does not pick out one demography
    but a one-parameter family of them. Any member will do, and this returns the
    feasible ratio closest to `preferred`: ratio 1 means P2 never changes size, so
    Ne_recent == Ne_ancestral and Fst is carried entirely by t_split, which keeps the
    sweep a clean function of Fst instead of mixing in an arbitrary ratio drift.

    The preferred ratio is tried first and is feasible across the default Fst range.
    The scan is the fallback for tightened bounds or an unusual `preferred`, where the
    feasible ratio set can exclude it; the scan then takes the nearest feasible grid
    point and bisects toward the preferred side to sit just inside the boundary rather
    than at scan resolution.

    Returns:
        (T2_over_T1, t_split, Ne_recent), or None if no ratio in ratio_bounds works.
    """
    def check(ratio):
        return why_infeasible(Fst, ratio, Ne_ancestral,
                              t_split_bounds, Ne_recent_bounds)

    if check(preferred) is None:
        t_split, Ne_recent = params_from_Fst_and_ratio(Fst, preferred, Ne_ancestral)
        return preferred, t_split, Ne_recent

    grid = np.linspace(ratio_bounds[0], ratio_bounds[1], n_scan)
    feasible = [r for r in grid if check(r) is None]
    if not feasible:
        return None

    best = min(feasible, key=lambda r: abs(r - preferred))
    # Walk from the nearest feasible grid point toward `preferred`; the boundary lies
    # between them, so bisecting recovers it independently of the grid spacing.
    bad = preferred
    for _ in range(60):
        mid = 0.5 * (bad + best)
        if check(mid) is None:
            best = mid
        else:
            bad = mid

    t_split, Ne_recent = params_from_Fst_and_ratio(Fst, best, Ne_ancestral)
    return float(best), t_split, Ne_recent


def feasible_Fst_range(Ne_ancestral,
                       t_split_bounds=DEFAULT_T_SPLIT_BOUNDS,
                       Ne_recent_bounds=DEFAULT_NE_RECENT_BOUNDS,
                       Fst_search=(1e-6, 0.99), n_scan=2001):
    """
    Range of Fst realizable once T2/T1 is free to be chosen per Fst.

    With the ratio free the feasible Fst set is a union over ratios and is much wider
    than at any single fixed ratio. This scans Fst, asks find_ratio_for_Fst whether
    each value admits some ratio, and returns the span of the feasible set, refining
    both endpoints by bisection so they do not depend on scan resolution.

    At the default bounds the binding constraint at the low end is t_split >= 10,
    giving Fst_min ~ 5.0e-4.

    Returns:
        dict with Fst_min, Fst_max, feasible, binding_low, binding_high.
    """
    def ok(Fst):
        return find_ratio_for_Fst(Fst, Ne_ancestral,
                                  t_split_bounds, Ne_recent_bounds) is not None

    lo, hi = Fst_search
    grid = np.geomspace(lo, hi, n_scan)
    mask = np.array([ok(f) for f in grid])

    if not mask.any():
        return {"Fst_min": np.nan, "Fst_max": np.nan, "feasible": False,
                "binding_low": None, "binding_high": None}

    idx = np.flatnonzero(mask)
    i_first, i_last = idx[0], idx[-1]

    def refine(f_bad, f_good):
        """Bisect the feasible/infeasible boundary between two Fst values."""
        for _ in range(80):
            mid = 0.5 * (f_bad + f_good)
            if ok(mid):
                f_good = mid
            else:
                f_bad = mid
        return f_good

    def binding_at(Fst):
        """Why Fst fails, reported at the preferred ratio as the representative one."""
        return why_infeasible(Fst, PREFERRED_RATIO, Ne_ancestral,
                              t_split_bounds, Ne_recent_bounds)

    if i_first == 0:
        Fst_min, binding_low = float(grid[0]), "search bound"
    else:
        Fst_min = refine(grid[i_first - 1], grid[i_first])
        binding_low = binding_at(grid[i_first - 1])

    if i_last == len(grid) - 1:
        Fst_max, binding_high = float(grid[-1]), "search bound"
    else:
        Fst_max = refine(grid[i_last + 1], grid[i_last])
        binding_high = binding_at(grid[i_last + 1])

    return {"Fst_min": float(Fst_min), "Fst_max": float(Fst_max),
            "feasible": True, "binding_low": binding_low,
            "binding_high": binding_high}


# ---------------------------------------------------------------------------
# Building the parameter table
# ---------------------------------------------------------------------------

def build_parameter_table(num_Fst=DEFAULT_NUM_FST, max_Fst=DEFAULT_MAX_FST,
                          min_Fst=None, Ne_ancestral=1e4, n=100, L=int(1e5),
                          t_split_bounds=DEFAULT_T_SPLIT_BOUNDS,
                          Ne_recent_bounds=DEFAULT_NE_RECENT_BOUNDS,
                          random_seed=42, prefix="K2", verbose=True):
    """
    Sweep Fst alone: sample num_Fst log-spaced values from min_Fst to max_Fst and, for
    each, pick any T2/T1 whose (t_split, Ne_recent) land inside the plausibility
    windows.

    T2/T1 is a free parameter rather than a swept one -- a target Fst is realizable by
    a whole family of ratios, and find_ratio_for_Fst returns the one nearest
    PREFERRED_RATIO. So the table is a one-dimensional sweep in Fst, and the achieved
    T2/T1 is recorded per row as an outcome rather than fixed in advance.

    min_Fst defaults to the low end of the feasible range, which at the default bounds
    is set by t_split >= 10 (about 5.0e-4). Fst values are log-spaced because the
    range spans over two orders of magnitude and the interesting behaviour (the
    detectability transition) sits at the low end.

    Returns:
        DataFrame with one row per simulation, including the achieved Fst and T2/T1
        recomputed from the solved parameters as a self-check.
    """
    rng = feasible_Fst_range(Ne_ancestral, t_split_bounds=t_split_bounds,
                             Ne_recent_bounds=Ne_recent_bounds)
    if not rng["feasible"]:
        raise ValueError(
            "no feasible Fst at any T2/T1 within the given t_split and Ne_recent "
            "bounds; widen them."
        )

    lo = rng["Fst_min"] if min_Fst is None else min_Fst
    hi = min(rng["Fst_max"], max_Fst)
    if lo < rng["Fst_min"]:
        raise ValueError(
            f"min_Fst={lo:g} is below the feasible floor {rng['Fst_min']:.6g} "
            f"(bound by: {rng['binding_low']}); raise min_Fst or widen the bounds."
        )
    if lo >= hi:
        raise ValueError(
            f"empty Fst range: min_Fst={lo:g} is not below max_Fst={hi:g}."
        )

    Fst_values = np.geomspace(lo, hi, num_Fst) if num_Fst > 1 else np.array([lo])
    if verbose:
        print(f"Fst in [{lo:.6g}, {hi:.6g}] ({num_Fst} log-spaced values), "
              f"T2/T1 free (preferring {PREFERRED_RATIO:g})")

    rows = []
    for Fst in Fst_values:
        found = find_ratio_for_Fst(Fst, Ne_ancestral,
                                   t_split_bounds=t_split_bounds,
                                   Ne_recent_bounds=Ne_recent_bounds)
        if found is None:
            if verbose:
                print(f"Fst={Fst:g}: no feasible T2/T1 -- skipped")
            continue
        ratio, t_split, Ne_recent = found

        T1, T2, T12 = coalescent_times(t_split, Ne_recent, Ne_ancestral)
        Fst_achieved, F2 = summary_statistics(T1, T2, T12)
        rows.append({
            "Fst_target": Fst,
            "T2_over_T1_target": ratio,
            "t_split": t_split,
            "Ne_recent": Ne_recent,
            "Ne_ancestral": Ne_ancestral,
            "alpha_1": 0.0,
            "alpha_2": growth_rate(Ne_recent, Ne_ancestral, t_split),
            "T1": T1, "T2": T2, "T12": T12,
            "Fst": Fst_achieved,
            "T2_over_T1": T2 / T1,
            "F2": F2,
            # Ne_1/Ne_2 keep a uniform K=2 schema for downstream code
            "Ne_1": Ne_ancestral,
            "Ne_2": Ne_recent,
            "n": n,
            "L": L,
        })

    if not rows:
        raise ValueError(
            "no feasible Fst targets: no sampled value admitted an in-bounds T2/T1."
        )

    params = pd.DataFrame(rows)
    params.insert(0, "simID",
                  [f"{prefix}_{i:04d}" for i in range(len(params))])
    # vary the seed per simulation so replicates are independent
    params["random_seed"] = random_seed + np.arange(len(params))
    return params


# ---------------------------------------------------------------------------
# Running the simulations
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet_inner_tqdm():
    """
    Suppress the per-tree progress bar drawn inside simulate_haplotypes.

    demography_spectrum does `from tqdm import tqdm`, so it holds its own reference
    to the class. Rebinding that module-level name to a pass-through wrapper mutes
    the inner bar without touching tqdm itself or the outer bar.
    """
    original = demography_spectrum_module.tqdm

    def silent_tqdm(iterable=None, *args, **kwargs):
        kwargs["disable"] = True
        return original(iterable, *args, **kwargs)

    demography_spectrum_module.tqdm = silent_tqdm
    try:
        yield
    finally:
        demography_spectrum_module.tqdm = original


def simulate_one(simID, t_split, Ne_recent, Ne_ancestral, n, L, outdir, random_seed):
    """
    Simulate one demography and save its genotype matrix as <simID>.npy.

    Returns (simID, output path). Kept picklable (module-level, plain arguments) so
    it can be dispatched to a ProcessPoolExecutor.

    DemographySpectrum.simulate_haplotypes draws its own per-tree tqdm bar. That
    would interleave with (and scroll away) the outer per-simulation bar, and under
    a process pool several would fight over the same terminal lines, so it is
    silenced here via tqdm's global disable flag.
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even (haploid samples per population), got {n}")

    demography, _, _ = build_demography(t_split, Ne_recent, Ne_ancestral)
    # simulate_haplotypes takes DIPLOID individuals per population and returns 2n
    # haploid rows per population; n here is the haploid count per population.
    spectrum = DemographySpectrum(demography)
    with _quiet_inner_tqdm():
        haplotypes = spectrum.simulate_haplotypes(
            n=n // 2, L=L, random_seed=random_seed)

    expected_shape = (2 * n, L)
    assert haplotypes.shape == expected_shape, (
        f"{simID}: expected {expected_shape}, got {haplotypes.shape}"
    )

    path = os.path.join(outdir, f"{simID}.npy")
    np.save(path, haplotypes.astype(np.uint8))
    return simID, path


def run_simulations(params, outdir=DEFAULT_OUTDIR, num_threads=1, verbose=True):
    """
    Run every simulation in the parameter table, optionally in parallel.

    num_threads > 1 uses a process pool: msprime releases work to C but the
    per-tree mutation placement in simulate_haplotypes is pure Python, so threads
    would contend on the GIL and processes are the useful unit of parallelism.
    """
    os.makedirs(outdir, exist_ok=True)
    total = len(params)

    jobs = [
        (r.simID, r.t_split, r.Ne_recent, r.Ne_ancestral,
         int(r.n), int(r.L), outdir, int(r.random_seed))
        for r in params.itertuples()
    ]

    # One bar over simulations, ticked as each finishes. In the parallel case
    # as_completed yields in completion order, so the bar advances on genuine
    # completions rather than on dispatch.
    bar = tqdm(total=total, desc="simulations", unit="sim", disable=not verbose)

    if num_threads <= 1:
        with bar:
            for job in jobs:
                simulate_one(*job)
                bar.set_postfix_str(job[0], refresh=False)
                bar.update(1)
        return

    with bar, ProcessPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(simulate_one, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            simID = futures[future]
            future.result()  # re-raise any worker exception
            bar.set_postfix_str(simID, refresh=False)
            bar.update(1)


def run(num_Fst=DEFAULT_NUM_FST, max_Fst=DEFAULT_MAX_FST, min_Fst=None,
        Ne_ancestral=1e4, n=100, L=int(1e5), outdir=DEFAULT_OUTDIR,
        num_threads=1, random_seed=42, prefix="K2", dry_run=False, verbose=True):
    """Build the parameter table, write it, and run the simulations."""
    params = build_parameter_table(
        num_Fst=num_Fst, max_Fst=max_Fst, min_Fst=min_Fst,
        Ne_ancestral=Ne_ancestral, n=n, L=L,
        random_seed=random_seed, prefix=prefix, verbose=verbose)

    if verbose:
        err = max((params.Fst - params.Fst_target).abs().max(),
                  (params.T2_over_T1 - params.T2_over_T1_target).abs().max())
        ratios_used = params.T2_over_T1.unique()
        print(f"\n{len(params)} simulations over Fst in "
              f"[{params.Fst.min():.6g}, {params.Fst.max():.6g}]")
        print(f"T2/T1 used: {params.T2_over_T1.min():.6g} to "
              f"{params.T2_over_T1.max():.6g} ({len(ratios_used)} distinct)")
        print(f"max |achieved - target| over Fst and T2/T1: {err:.2e}")

    os.makedirs(outdir, exist_ok=True)
    params_path = os.path.join(outdir, "simulation_parameters.csv")
    params.to_csv(params_path, index=False)
    if verbose:
        print(f"parameters -> {params_path}")

    if dry_run:
        return params

    run_simulations(params, outdir=outdir, num_threads=num_threads, verbose=verbose)
    if verbose:
        print(f"\nwrote {len(params)} simulations to {outdir}")
    return params


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-Fst", type=int, default=DEFAULT_NUM_FST, help=f"number of log-spaced Fst values (default: {DEFAULT_NUM_FST})")
    p.add_argument("--max-Fst", type=float, default=DEFAULT_MAX_FST, help=f"upper end of the Fst sweep (default: {DEFAULT_MAX_FST})")
    p.add_argument("--min-Fst", type=float, default=None, help="lower end of the Fst sweep (default: the feasible floor, ~5e-4)")
    p.add_argument("--L", type=int, default=int(1e5), help="number of independent SNPs")
    p.add_argument("--n", type=int, default=100, help="haploid samples per population (must be even)")
    p.add_argument("--Ne-ancestral", type=float, default=1e4, help="ancestral population size")
    p.add_argument("--num-threads", type=int, default=1, help="run simulations in parallel across this many processes")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--prefix", default="K2")
    p.add_argument("--dry-run", action="store_true", help="write the parameter table and exit without simulating")
    args = p.parse_args()

    run(num_Fst=args.num_Fst, max_Fst=args.max_Fst, min_Fst=args.min_Fst,
        Ne_ancestral=args.Ne_ancestral, n=args.n, L=args.L,
        outdir=args.outdir, num_threads=args.num_threads,
        random_seed=args.random_seed, prefix=args.prefix, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
