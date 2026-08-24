#%%
"""
PC alignment for a recently split sister pair, across sample size and sampling balance.

Measures how well PCA recovers the sister-separating axis

    u_1 = (sqrt(phi_2), -sqrt(phi_1), 0, ..., 0) / sqrt(phi_1 + phi_2)

on results/simulations/african_split (K = 4: CEU, CHB, YRI1, YRI2, in that
alphabetical block order; YRI1/YRI2 are the sisters). Detectability is governed by

    nu_1 = N * H * Fst,   H = 2*phi_1*phi_2/(phi_1+phi_2) = harmonic mean(phi_1, phi_2)

so the axis separates when nu_1 > nu_c, i.e. when

    H > H_c = nu_c / (N * Fst)

nu_c is the exact heteroskedastic threshold from src/threshold_and_overlap, which accounts
for the unequal within-population coalescent times (eps_a) and the spread of the per-locus
weights. sqrt(N/L_eff) is the eps = 0, flat-weight special case of it and is kept in the
output as nu_c_flat for comparison, not used as the criterion.

Two sweeps:

    vary_N.csv    phi_1 = phi_2 = 0.05 fixed, N from 200 to 1000
    vary_phi.csv  N = 1000 fixed, the sisters supplying 200 samples between them, split
                  from (5, 195) through (100, 100)

Why these values. The point is to place the phase transition *inside* the grid rather than
sweeping a regime where the answer is always the same. With L_input = 5e4, the OOA3
matrices give L_eff ~ 0.161 * L_input = 8050 (much lower than a panmictic sample's ~40%,
because the four-population frequency spectrum is far richer in rare variants), so at
N = 1000 and Fst = 1e-2 the critical harmonic mean is H_c = 0.035. Both grids straddle it:

  * the N sweep at H = 0.05 predicts |cos| = 0 at N = 400 rising to 0.64 at N = 1000
  * the phi sweep crosses H_c at n_1 ~ 20 of the 200 sister samples

The phi sweep is the substantive one. Every cell spends exactly the same 200 samples on the
sister pair, but H depends on their *balance*: (5, 195) gives H = 0.0097 and (100, 100)
gives H = 0.100, a factor of ten. So Fst = 1e-2 -- not a small value -- is predicted to be
completely undetectable under the lopsided design and cleanly recovered under the balanced
one, with the same sequencing effort. A harmonic mean is dominated by its smaller argument,
so the 195 extra samples in the first design buy essentially nothing.

Sample-size cap. Each population was simulated with 5000 haploid samples, so N <= 20000 and
any design needs every n_a <= 5000. The grids below respect that; run() asserts it.

Fst is always the theoretical value from the filename, never re-estimated.

Outputs (results/analysis/SisterPCAlignment):
    vary_N.csv    one row per (simulation, N cell)
    vary_phi.csv  one row per (simulation, sampling-balance cell)

Job array. pop_coal_times (exact_sister_nu) is the cost: an msprime numerical
integration per population pair, independent per simulation and ~15-20 min each. Serially
over all 100 african_split simulations that is too slow for a single job, but every
simulation is otherwise independent, so --index runs one simulation (both sweeps, all
designs) and writes its own partial/sim<index>.csv per sweep; --aggregate concatenates them
into the final vary_N.csv / vary_phi.csv once the array finishes. run() (no --index) still
does everything serially in-process, for dry runs on a handful of simulations.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_recovery import (load_simulations, pc_recovery, subsample, support_overlap_floor,
                         SIMULATION_ROOT)
from crossing_fit import crossings_table, replicate_rows
from exact_sister_nu import cached_coal_times, exact_sister_nu, pop_coal_times
from sweeps import ANALYSIS_ROOT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from threshold_and_overlap import approx_nu_c, approx_overlap, exact_nu_c, exact_overlap

# Loaded by path, not by bare module name: results/plotting_scripts holds a *different*
# SisterEigenvalue.py (the figure script for the same analysis), so a plain
# `from SisterEigenvalue import ...` resolves to whichever directory happens to come first
# on sys.path. Every analysis module here has a same-named plotting counterpart, so this is
# a general hazard, not a one-off.
_spec = importlib.util.spec_from_file_location(
    "_sister_eigenvalue",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "SisterEigenvalue.py"))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)
SISTER, POPS = _se.SISTER, _se.POPS
harmonic_phi, sister_axis = _se.harmonic_phi, _se.sister_axis
N_PER_POP_FULL = _se.N_PER_POP_FULL

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "african_split")
DEFAULT_NAME = "SisterPCAlignment"

DEFAULT_L = 50000
DEFAULT_NUM_REPLICATES = 10
DEFAULT_SEED = 0

# --- N sweep: sisters held at phi_1 = phi_2 = 0.05, so H = 0.05 throughout.
PHI_SISTER = 0.05
N_VALUES = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])

# --- phi sweep: N = 1000, the sisters supplying SISTER_TOTAL samples between them.
PHI_SWEEP_N = 1000
SISTER_TOTAL = 200
SISTER_SPLITS = [(5, 195), (10, 190), (15, 185), (20, 180), (30, 170),
                 (40, 160), (50, 150), (70, 130), (85, 115), (100, 100)]


def design_N(N, phi=PHI_SISTER):
    """n = [n_other, n_other, n_sister, n_sister] with each sister at fraction phi of N."""
    n_s = int(round(phi * N))
    n_o = (N - 2 * n_s) // 2
    n = np.array([n_o, n_o, n_s, n_s], dtype=int)
    n[SISTER[1]] += N - n.sum()          # absorb the rounding residual
    return n


def design_phi(n1, n2, N=PHI_SWEEP_N):
    """n = [n_other, n_other, n1, n2], the other two populations splitting the remainder."""
    n_o = (N - n1 - n2) // 2
    n = np.array([n_o, n_o, 0, 0], dtype=int)
    n[SISTER[0]], n[SISTER[1]] = n1, n2
    n[0] += N - n.sum()
    return n


def check_feasible(designs):
    """Every population was simulated with N_PER_POP_FULL samples; nothing may exceed it."""
    for label, n in designs:
        if n.max() > N_PER_POP_FULL:
            raise ValueError(f"design {label} asks for {n.max()} samples from one "
                             f"population but only {N_PER_POP_FULL} were simulated: {list(n)}")
        if n.min() < 2:
            raise ValueError(f"design {label} has a population with < 2 samples: {list(n)}")


SISTER_RANK = 2   # PC3 (0-indexed): see SisterEigenvalue.SISTER_RANK


def measure_cell(Z, n, L, num_replicates, seed, nu=None, eps=None):
    """
    PC alignment with the predicted sister axis, median over replicates.

    Fixed-rank selection at SISTER_RANK (PC3), not an overlap search: the sister axis is
    *not* PC1, so indices=[0] would return a near-orthogonal vector (overlap 2e-4 where the
    true axis gives 0.97) -- but the deep OOA splits carry far larger eigenvalues than the
    recent YRI1/YRI2 split, so the sister axis sits deterministically at rank 2, measured on
    every one of 175 cells in SisterEigenvalue.py with zero exceptions. A search (over the
    full spectrum, or even a couple of neighbouring ranks) doesn't change which eigenvector
    gets picked, but it does inflate the no-signal floor -- a max over several candidates
    finds a larger baseline overlap by chance than a single fixed-rank read -- so this reads
    SISTER_RANK directly via indices, not select_by_overlap.

    nu is passed so analytical_cos is populated and floored at the finite-N random overlap
    rather than left NaN below threshold. eps carries the within-population deviations so
    the prediction uses the exact heteroskedastic threshold rather than sqrt(gamma).
    """
    V = sister_axis(n)[:, None]
    return pc_recovery(Z, n, V, indices=[SISTER_RANK], L=L, num_replicates=num_replicates,
                       seed=seed, nu=[nu] if nu is not None else None, eps=eps)


DESIGNS_N = [(f"N{N}", design_N(int(N))) for N in N_VALUES]
DESIGNS_PHI = [(f"n1_{a}", design_phi(a, b)) for a, b in SISTER_SPLITS]
check_feasible(DESIGNS_N)
check_feasible(DESIGNS_PHI)


def _cell_row(prow, label, n, T, Z, L, num_replicates, seed):
    """One (simulation, design) row: exact nu, then the exact threshold/overlap and the
    measurement.

    nu_c and the predicted overlap use threshold_and_overlap's exact route (per-locus h,
    not the two-moment gamma_eff expansion pc_recovery.analytical_cos relies on): a
    dedicated subsample the same shape as pc_recovery's own replicates supplies h directly,
    since pc_recovery discards it after reducing to (mu2, delta). This subsample uses
    rep index 0's seed so it matches what the measurement's first replicate draws, but it
    is otherwise independent of the num_replicates used for cos_median.
    """
    Fst = float(prow["Fst"])
    N = int(n.sum())
    H = harmonic_phi(n)
    nu_approx = N * H * Fst
    nu = exact_sister_nu(T, n, sister_axis(n))["nu_exact"]
    # eps_a = tau_a/tau_bar - 1, weighted by the sampling proportions: the within-population
    # coalescent times are heterogeneous on this demography (tau/tau_bar spans 0.81-1.17),
    # so eps = 0 is not available here.
    tau = np.diag(T)
    phi_all = n / N
    eps = tau / float(np.sum(phi_all * tau)) - 1.0

    rng_h = np.random.default_rng([seed, 0])
    sub_h = subsample(Z, n, L, rng_h)
    p_h = sub_h.mean(axis=0)
    h = p_h * (1.0 - p_h)
    h = h[h > 0]
    gamma_raw = N / sub_h.shape[1]
    nu_c = exact_nu_c(phi_all, eps, h, gamma_raw)
    overlap_pred = exact_overlap(nu, phi_all, eps, h, gamma_raw, nu_c=nu_c)

    res = measure_cell(Z, n, L, num_replicates, seed, nu=nu, eps=eps)
    gamma = N / res["L_eff"]
    # No-signal floor for the fixed-rank read (see measure_cell): depends only on the
    # axis's support (n_sister1 + n_sister2), not on Fst or N, so it is comparable to
    # overlap_pred/analytical_cos as a lower bound. num_candidates=1 since measure_cell
    # reads a single fixed rank rather than searching several.
    overlap_floor = support_overlap_floor(sister_axis(n)[:, None], n, num_candidates=1)
    row = dict(
        simID=prow["simID"], t_split=float(prow["t_split"]), Fst=Fst,
        design=label, N=N, H=H,
        n_sister1=int(n[SISTER[0]]), n_sister2=int(n[SISTER[1]]),
        n_other=int(n[0]),
        phi1=n[SISTER[0]] / N, phi2=n[SISTER[1]] / N,
        nu=nu, nu_approx=nu_approx, gamma=gamma,
        gamma_raw=gamma_raw, mu2=res["mu2"], delta=res["delta"],
        nu_c=nu_c, nu_c_flat=np.sqrt(gamma), detectable=bool(nu > nu_c),
        H_c=nu_c / (N * Fst),
        cos_median=res["median_cos"], se_cos=res["se_cos"],
        analytical_cos=res["analytical_cos"], overlap_pred=overlap_pred,
        overlap_floor=overlap_floor,
        L_raw=res["L_raw"], L_eff=res["L_eff"],
        num_replicates=num_replicates, seed=seed)
    # Fst is the swept axis for this figure, so it is the column the fit regresses against.
    base = {k: v for k, v in row.items() if k != "Fst"}
    return row, replicate_rows(base, Fst, "Fst", res["cos_replicates"], seed=seed)


def run_one_simulation(index, simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, L=DEFAULT_L,
                       num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED,
                       verbose=True):
    """
    Both sweeps for a single simulation (one array task).

    pop_coal_times for this simulation's t_split is computed fresh, not through the shared
    cache: 100 array tasks writing to one npz concurrently would race. Each task's coal
    time is cheap to compute once and never reused across tasks, so there is nothing to
    cache here -- the shared cache in results/analysis/SisterEigenvalue is for the serial
    dry-run path (run()) and for SisterEigenvalue.py itself.

    Outputs (results/analysis/<name>/partial): sim<index>_vary_N.csv, sim<index>_vary_phi.csv
    """
    params, arrays = load_simulations(simdir)
    if not 0 <= index < len(params):
        raise IndexError(f"index {index} out of range for {len(params)} simulations")
    prow = params.iloc[index]
    Z = arrays[index]
    T = pop_coal_times(float(prow["t_split"]))

    outdir = os.path.join(ANALYSIS_ROOT, name, "partial")
    os.makedirs(outdir, exist_ok=True)
    for sweep_name, designs in [("vary_N", DESIGNS_N), ("vary_phi", DESIGNS_PHI)]:
        results = [_cell_row(prow, label, n, T, Z, L, num_replicates, seed)
                   for label, n in designs]
        df = pd.DataFrame([r[0] for r in results])
        out = os.path.join(outdir, f"sim{index}_{sweep_name}.csv")
        df.to_csv(out, index=False)
        df_rep = pd.DataFrame([r for res in results for r in res[1]])
        rep_out = os.path.join(outdir, f"sim{index}_{sweep_name}_replicates.csv")
        df_rep.to_csv(rep_out, index=False)
        if verbose:
            print(f"  wrote {len(df)} rows -> {out}", flush=True)
            print(f"  wrote {len(df_rep)} replicate rows -> {rep_out}", flush=True)


def _theory_overlap_for(df, level_keys):
    """
    curve -> (Fst -> predicted squared overlap), using the exact heteroskedastic route.

    nu is very nearly linear in Fst at fixed design (the analysis records the exact nu per
    cell, from the full coalescent-time matrix), so the closure fits a line through this
    curve's own (Fst, nu) pairs and feeds it to exact_overlap. That keeps the additive
    pedestal -- nu does not pass through the origin, and dropping the intercept is what
    makes the nu = N*H*Fst approximation understate the eigenvalue.

    eps_a is nonzero on this demography (tau/tau_bar spans 0.81-1.17), so the flat
    sqrt(gamma) threshold is not usable here: at low Fst the exact nu_c sits above nu, and
    the flat formula would report a large false-positive plateau.
    """
    def make(levels):
        d = df
        for k, v in levels.items():
            d = d[d[k] == v]
        if d.empty:
            return None
        cells = d.groupby("Fst").first().reset_index()
        if len(cells) < 2:
            return None
        slope, intercept = np.polyfit(cells.Fst.to_numpy(float), cells.nu.to_numpy(float), 1)

        # Rebuild phi/eps for the K populations from what the analysis stored. Only the
        # sister pair's proportions vary across this sweep; the rest is fixed by the
        # demography, so the recorded gamma_raw/mu2/delta and nu_c carry the context.
        gamma_raw = float(cells.gamma_raw.median())
        mu2 = float(cells.mu2.median())
        delta = float(cells.delta.median())
        phi1, phi2 = float(cells.phi1.iloc[0]), float(cells.phi2.iloc[0])
        phi = np.array([phi1, phi2, max(1.0 - phi1 - phi2, 1e-12)])
        eps = np.zeros(3)
        nu_c = approx_nu_c(phi, eps, gamma_raw, mu2, delta)
        return lambda fst: approx_overlap(slope * fst + intercept, phi, eps,
                                          gamma_raw, mu2, delta, nu_c=nu_c)
    return make


def _write_crossings(df_rep, sweep_name, outdir, verbose=True):
    """Where measured and predicted overlap cross the target, one row per design."""
    level_keys = ["design"]
    df_cross = crossings_table(
        df_rep, level_keys, "Fst",
        theory_overlap=_theory_overlap_for(df_rep, level_keys),
        extra_keys=("N", "H", "n_sister1", "n_sister2", "n_other", "phi1", "phi2",
                    "gamma", "gamma_raw", "mu2", "delta", "L_eff", "num_replicates"))
    out = os.path.join(outdir, f"{sweep_name}_crossings.csv")
    df_cross.to_csv(out, index=False)
    if verbose:
        print(f"wrote {len(df_cross)} crossing rows -> {out}")
    return df_cross


def aggregate(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, verbose=True):
    """Concatenate every partial/sim<i>_<sweep>.csv into the final per-sweep CSVs."""
    params, _ = load_simulations(simdir)
    outdir = os.path.join(ANALYSIS_ROOT, name)
    partial_dir = os.path.join(outdir, "partial")

    for sweep_name in ("vary_N", "vary_phi"):
        for suffix in ("", "_replicates"):
            paths = [os.path.join(partial_dir, f"sim{i}_{sweep_name}{suffix}.csv")
                     for i in range(len(params))]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"{len(missing)} of {len(paths)} partial files "
                                        f"missing for {sweep_name}{suffix}, e.g. {missing[0]}")
            df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            out = os.path.join(outdir, f"{sweep_name}{suffix}.csv")
            df.to_csv(out, index=False)
            if verbose:
                print(f"wrote {len(df)} rows -> {out}")
            if suffix == "_replicates":
                _write_crossings(df, sweep_name, outdir, verbose=verbose)


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, L=DEFAULT_L,
        num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, verbose=True,
        n_simulations=None):
    """
    Serial, in-process run over the first n_simulations (default: all) -- for dry runs.

    Uses the shared coal-time cache, since this path is single-process and there is no
    race. The cluster path is run_one_simulation + aggregate, driven by a job array.
    """
    params, _ = load_simulations(simdir)
    n_simulations = len(params) if n_simulations is None else n_simulations
    Tmap = cached_coal_times(sorted(params.t_split.unique()[:n_simulations]),
                             os.path.join(ANALYSIS_ROOT, "SisterEigenvalue",
                                          "coal_times.npz"), verbose=verbose)
    # run_one_simulation calls pop_coal_times (bound into this module's own namespace by
    # the `from exact_sister_nu import ...` above) per call; patch that name here to reuse
    # Tmap instead, since this path already paid for the shared cache. Patching the
    # definition in exact_sister_nu would not affect this module's already-bound name.
    global pop_coal_times
    orig = pop_coal_times
    pop_coal_times = lambda t: Tmap[float(t)]
    try:
        for i in range(n_simulations):
            run_one_simulation(i, simdir=simdir, name=name, L=L,
                               num_replicates=num_replicates, seed=seed, verbose=verbose)
    finally:
        pop_coal_times = orig
    aggregate(simdir=simdir, name=name, verbose=verbose)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--L", type=int, default=DEFAULT_L)
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--index", type=int, default=None,
                   help="run a single simulation (one array task) instead of the full set")
    p.add_argument("--aggregate", action="store_true",
                   help="concatenate partial/ outputs into vary_N.csv / vary_phi.csv")
    p.add_argument("--n-simulations", type=int, default=None,
                   help="limit to the first N simulations, for a serial dry run")
    args = p.parse_args()
    if args.aggregate:
        aggregate(simdir=args.simdir, name=args.name)
    elif args.index is not None:
        run_one_simulation(args.index, simdir=args.simdir, name=args.name, L=args.L,
                           num_replicates=args.num_replicates, seed=args.seed)
    else:
        run(simdir=args.simdir, name=args.name, L=args.L,
            num_replicates=args.num_replicates, seed=args.seed,
            n_simulations=args.n_simulations)


if __name__ == "__main__":
    main()

# %%
