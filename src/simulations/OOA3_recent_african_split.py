#%%
"""
Simulate haplotypes under stdpopsim's HomSap/OutOfAfrica_3G09 (Gutenkunst et al. 2009)
model, extended with a recent split within the African (YRI) population, over a sweep
of split times t_split.

The base model has three populations (YRI, CEU, CHB) with the standard OOA / CEU-CHB
split history. A fourth population, YRI2, is added: going backward in time, YRI and
YRI2 are both sampled at present and merge into a single YRI lineage at t_split
generations ago (msprime's add_population_split), rejoining the unmodified OOA history
above that. YRI2 inherits YRI's migration rates with CEU and CHB.

t_split is swept from t_split_min to t_split_max, log-spaced, num_t_split values (dense
towards the min side -- the most recent splits are the ones of interest). Note this
range can exceed the OOA split itself (848 generations), in which case the within-Africa
split simply predates the OOA split.

Example:
    python OOA3_recent_african_split.py                    # the default 100-point sweep
    python OOA3_recent_african_split.py --num-t-split 20    # a quicker sweep

The theoretical Fst between YRI and YRI2 is computed exactly from the demography via
DemographyDebugger.mean_coalescence_time, as 1 - t_within / t_between, where t_within is
the (shared, by symmetry) mean coalescence time within either African population and
t_between is the mean coalescence time between them.

Outputs (results/simulations/OOA3_recent_african_split by default):
    simulation_parameters.csv   one row per simulation
    t<t_split>_Fst<Fst>.npy     the (4 * n_per_pop) x L genotype matrix, saved as
                                 np.bool_, with t_split and Fst in scientific notation to
                                 4 decimal places (e.g. t1.0000e+02_Fst4.0255e-03.npy).
                                 Sample order is alphabetical by population: CEU, CHB,
                                 YRI, YRI2.
"""
import argparse
import copy
import os
import sys
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import stdpopsim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from utils import simulate_haplotypes

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTDIR = os.path.join(
    _PROJECT_ROOT, "results", "simulations", "OOA3_recent_african_split",
)

DEFAULT_N_PER_POP = 500
# Per-population haploid sample sizes, keyed by population name (alphabetical: CEU, CHB,
# YRI, YRI2). Overrides DEFAULT_N_PER_POP when a population isn't given a uniform size --
# CEU/CHB simulated deeper than YRI/YRI2 to cover both the phi sweep (sisters up to 2000
# each, i.e. phi=0.2 at N=10000) and the N sweep (others up to 4500 each at N=10000).
DEFAULT_N_PER_POP_BY_POP = {"CEU": 5000, "CHB": 5000, "YRI": 2000, "YRI2": 2000}
DEFAULT_L = int(2e5)
DEFAULT_NUM_T_SPLIT = 50
DEFAULT_T_SPLIT_MIN = 10
DEFAULT_T_SPLIT_MAX = 1000

DEFAULT_RANDOM_SEED = 42
DEFAULT_N_WORKERS = max(cpu_count() - 3, 1)


def format_simID(t_split, Fst):
    """Filename stem for a simulation: t_split and Fst in scientific notation to 4 decimals."""
    return f"t{t_split:.4e}_Fst{Fst:.4e}"


def t_split_values(num_t_split, t_split_min, t_split_max):
    """num_t_split log-spaced split times, denser towards t_split_min (the recent end)."""
    if num_t_split < 1:
        raise ValueError(f"num_t_split must be at least 1, got {num_t_split}")
    if not (0.0 < t_split_min <= t_split_max):
        raise ValueError(
            f"need 0 < t_split_min <= t_split_max, got t_split_min={t_split_min:g}, "
            f"t_split_max={t_split_max:g}"
        )
    if num_t_split == 1:
        return np.array([t_split_min])
    return np.geomspace(t_split_min, t_split_max, num_t_split)


def ooa3_recent_split_demography(t_split):
    """
    stdpopsim's OutOfAfrica_3G09 with a fourth population YRI2 split off YRI at
    t_split generations ago. YRI2 mirrors YRI's migration rates with CEU/CHB.
    """
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("OutOfAfrica_3G09")
    # model.model is a cached object shared across calls -- copy before mutating
    demography = copy.deepcopy(model.model)

    yri = demography["YRI"].id
    demography.add_population(
        name="YRI2", initial_size=demography["YRI"].initial_size, initially_active=True)
    yri2 = demography["YRI2"].id

    demography.migration_matrix[yri2, :] = demography.migration_matrix[yri, :]
    demography.migration_matrix[:, yri2] = demography.migration_matrix[:, yri]
    demography.migration_matrix[yri2, yri] = 0
    demography.migration_matrix[yri, yri2] = 0

    demography.add_population_split(time=t_split, derived=["YRI2"], ancestral="YRI")
    demography["YRI"].initially_active = True  # still sampled at present, not just post-split
    demography.sort_events()
    return demography


def theoretical_Fst(demography):
    """
    Exact theoretical Fst between YRI and YRI2: 1 - t_within / t_between, with both mean
    coalescence times computed by the DemographyDebugger. t_within is the same for YRI
    and YRI2 by symmetry (they are exchangeable derived populations of a single split).
    """
    dbg = demography.debug()
    t_within = dbg.mean_coalescence_time(lineages={"YRI": 2})
    t_between = dbg.mean_coalescence_time(lineages={"YRI": 1, "YRI2": 1})
    return 1.0 - t_within / t_between


POPS = ["CEU", "CHB", "YRI", "YRI2"]


def _n_per_pop_dict(n_per_pop):
    """Normalize a scalar or {pop: n} mapping to a full {pop: n} dict over POPS."""
    if isinstance(n_per_pop, dict):
        missing = [p for p in POPS if p not in n_per_pop]
        if missing:
            raise ValueError(f"n_per_pop is missing populations: {missing}")
        return {p: int(n_per_pop[p]) for p in POPS}
    return {p: int(n_per_pop) for p in POPS}


def build_parameter_table(num_t_split=DEFAULT_NUM_T_SPLIT, t_split_min=DEFAULT_T_SPLIT_MIN,
                          t_split_max=DEFAULT_T_SPLIT_MAX, n_per_pop=DEFAULT_N_PER_POP,
                          L=DEFAULT_L, random_seed=DEFAULT_RANDOM_SEED):
    """
    One row per simulation over num_t_split log-spaced split times, each with its exact
    theoretical Fst between YRI and YRI2 (built from the demography, not simulated).

    n_per_pop may be a scalar (same size for all four populations) or a
    {pop_name: n} dict for asymmetric sample sizes, e.g. DEFAULT_N_PER_POP_BY_POP.
    """
    n_dict = _n_per_pop_dict(n_per_pop)
    N = sum(n_dict.values())

    t_values = t_split_values(num_t_split, t_split_min, t_split_max)
    Fst_values = [theoretical_Fst(ooa3_recent_split_demography(t)) for t in t_values]

    params = pd.DataFrame({
        "simID": [format_simID(t, f) for t, f in zip(t_values, Fst_values)],
        "t_split": t_values,
        "Fst": Fst_values,
        "n_CEU": n_dict["CEU"], "n_CHB": n_dict["CHB"],
        "n_YRI": n_dict["YRI"], "n_YRI2": n_dict["YRI2"],
        "N": N,
        "L": L,
        "random_seed": random_seed + np.arange(num_t_split),
    })

    duplicates = params.simID.duplicated()
    if duplicates.any():
        raise ValueError(
            f"{int(duplicates.sum())} t_split values collide after rounding to 4 "
            f"decimal places (e.g. {params.simID[duplicates].iloc[0]}); use fewer "
            f"values or a wider range."
        )
    return params


def simulate_one(simID, t_split, n_per_pop, L, outdir, random_seed, n_workers=1):
    """Simulate one split time and save its genotype matrix as <simID>.npy."""
    n_dict = _n_per_pop_dict(n_per_pop)
    demography = ooa3_recent_split_demography(t_split)
    X = simulate_haplotypes(
        demography, n={p: n / 2 for p, n in n_dict.items()}, L=L,
        random_seed=random_seed, n_workers=n_workers)
    np.save(os.path.join(outdir, f"{simID}.npy"), X)
    return X.shape[1]


def run(num_t_split=DEFAULT_NUM_T_SPLIT, t_split_min=DEFAULT_T_SPLIT_MIN,
        t_split_max=DEFAULT_T_SPLIT_MAX, n_per_pop=DEFAULT_N_PER_POP, L=DEFAULT_L,
        outdir=DEFAULT_OUTDIR, random_seed=DEFAULT_RANDOM_SEED,
        n_workers=DEFAULT_N_WORKERS, dry_run=False, verbose=True):
    """
    Run the full t_split sweep.

    Each simulation's L replicate coalescent trees are independent, so simulate_haplotypes
    splits them across n_workers processes; one simulation runs at a time, using all
    n_workers for it, rather than running multiple simulations concurrently. For cluster
    use, run_one() runs a single row instead -- see build_parameter_table for how a task
    index maps to a row without needing this loop.
    """
    params = build_parameter_table(
        num_t_split=num_t_split, t_split_min=t_split_min, t_split_max=t_split_max,
        n_per_pop=n_per_pop, L=L, random_seed=random_seed)

    if verbose:
        print(f"{len(params)} simulations, t_split in "
              f"[{params.t_split.min():.4e}, {params.t_split.max():.4e}] generations, "
              f"n_per_pop={_n_per_pop_dict(n_per_pop)}, N={params.N.iloc[0]}, L={L}, "
              f"n_workers={n_workers}")
        print(f"output -> {outdir}")

    if dry_run:
        return params

    os.makedirs(outdir, exist_ok=True)
    params.to_csv(os.path.join(outdir, "simulation_parameters.csv"), index=False)

    bar = tqdm(params.itertuples(), total=len(params), disable=not verbose)
    for i, r in enumerate(bar, start=1):
        if os.path.exists(os.path.join(outdir, f"{r.simID}.npy")):
            bar.set_description(f"skipping simulation {i}/{num_t_split} (exists), "
                                 f"t_split={r.t_split:.4g}")
            continue
        bar.set_description(f"running simulation {i}/{num_t_split}, t_split={r.t_split:.4g}")
        simulate_one(r.simID, r.t_split, n_per_pop, L, outdir, int(r.random_seed),
                     n_workers=n_workers)

    if verbose:
        print(f"wrote {len(params)} simulations to {outdir}")
    return params


def run_one(task_index, num_t_split=DEFAULT_NUM_T_SPLIT, t_split_min=DEFAULT_T_SPLIT_MIN,
            t_split_max=DEFAULT_T_SPLIT_MAX, n_per_pop=DEFAULT_N_PER_POP, L=DEFAULT_L,
            outdir=DEFAULT_OUTDIR, random_seed=DEFAULT_RANDOM_SEED,
            n_workers=1, verbose=True):
    """
    Run a single row of the sweep, selected by task_index (0-based) into the same
    parameter table run() would build. For a SLURM array job: build_parameter_table is
    cheap (no simulation, just DemographyDebugger calls) and deterministic given the same
    arguments, so every array task can independently rebuild the full table and index into
    it -- no shared setup step or param file needs to exist first.

    Skips (rather than overwrites) a simID whose .npy already exists, so a re-submitted or
    retried task is a no-op.
    """
    params = build_parameter_table(
        num_t_split=num_t_split, t_split_min=t_split_min, t_split_max=t_split_max,
        n_per_pop=n_per_pop, L=L, random_seed=random_seed)
    if not (0 <= task_index < len(params)):
        raise IndexError(f"task_index {task_index} out of range for {len(params)} simulations")
    r = params.iloc[task_index]

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{r.simID}.npy")
    if os.path.exists(out_path):
        if verbose:
            print(f"task {task_index}: {r.simID} already exists, skipping")
        return

    if verbose:
        print(f"task {task_index}/{len(params)}: {r.simID}  "
              f"(t_split={r.t_split:.4g}, Fst={r.Fst:.4e}, N={r.N}, L={L})")
    simulate_one(r.simID, r.t_split, n_per_pop, L, outdir, int(r.random_seed),
                 n_workers=n_workers)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per-pop", type=int, default=None, help=f"haploid samples per population, same for all four (default: {DEFAULT_N_PER_POP} unless --n-per-pop-by-pop is given)")
    p.add_argument("--n-per-pop-by-pop", default=None, help="asymmetric haploid sample sizes as CEU=<n>,CHB=<n>,YRI=<n>,YRI2=<n>; overrides --n-per-pop and the default")
    p.add_argument("--L", type=int, default=DEFAULT_L, help=f"number of independent SNPs (default: {DEFAULT_L:g})")
    p.add_argument("--num-t-split", type=int, default=DEFAULT_NUM_T_SPLIT, help=f"number of log-spaced split times (default: {DEFAULT_NUM_T_SPLIT})")
    p.add_argument("--t-split-min", type=float, default=DEFAULT_T_SPLIT_MIN, help=f"lower end of the t_split sweep, in generations (default: {DEFAULT_T_SPLIT_MIN:g})")
    p.add_argument("--t-split-max", type=float, default=DEFAULT_T_SPLIT_MAX, help=f"upper end of the t_split sweep, in generations (default: {DEFAULT_T_SPLIT_MAX:g})")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS, help=f"processes to split each simulation's L replicates across (default: {DEFAULT_N_WORKERS}, i.e. cpu_count - 1)")
    p.add_argument("--dry-run", action="store_true", help="report the plan and exit without simulating")
    p.add_argument("--task-index", type=int, default=None, help="run only this 0-based row of the sweep and exit (for a cluster array job); reads $SLURM_ARRAY_TASK_ID if set and this is omitted but the env var is present")
    args = p.parse_args()

    if args.n_per_pop_by_pop is not None:
        n_per_pop = {}
        for tok in args.n_per_pop_by_pop.split(","):
            pop, n = tok.split("=")
            n_per_pop[pop.strip()] = int(n)
    elif args.n_per_pop is not None:
        n_per_pop = args.n_per_pop
    else:
        n_per_pop = DEFAULT_N_PER_POP

    task_index = args.task_index
    if task_index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if task_index is not None:
        run_one(task_index, num_t_split=args.num_t_split, t_split_min=args.t_split_min,
                t_split_max=args.t_split_max, n_per_pop=n_per_pop, L=args.L,
                outdir=args.outdir, random_seed=args.random_seed, n_workers=args.n_workers)
        return

    run(num_t_split=args.num_t_split, t_split_min=args.t_split_min,
        t_split_max=args.t_split_max, n_per_pop=n_per_pop, L=args.L,
        outdir=args.outdir, random_seed=args.random_seed, n_workers=args.n_workers,
        dry_run=args.dry_run)


if __name__ == "__main__":
    main()

# %%
