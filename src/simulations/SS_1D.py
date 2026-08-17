#%%
"""
Simulate 1D stepping-stone haplotypes over a sweep of migration rates, for each of
several deme counts K, using msprime.Demography.stepping_stone_model with reflecting
boundaries.

For each K, num_m migration rates are drawn log-spaced between max_m and min_m, denser
towards the max_m side (the interesting, well-mixed end of the range), and one
simulation is run per rate.

Example:
    python SS_1D.py                              # the default sweep over K = 10, 20, 50
    python SS_1D.py --K 10 20 --num-m 20          # a quicker two-K sweep

Outputs (results/simulations/SS_1D/K<K> by default, one subdirectory per K):
    simulation_parameters.csv   one row per simulation
    m<value>.npy                the (K * n_per_deme) x L genotype matrix, saved as
                                 np.bool_, with m in scientific notation to 4 decimal
                                 places (e.g. m2.5000e-01.npy)
"""
import argparse
import os
import sys
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import msprime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from utils import simulate_haplotypes

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTDIR_TEMPLATE = os.path.join(
    _PROJECT_ROOT, "results", "simulations", "SS_1D", "K{K}",
)

DEFAULT_K_VALUES = [10, 20, 50]
DEFAULT_NE = 500
DEFAULT_N_PER_DEME = 10
# K=50 has N = K * n_per_deme lineages migrating through 50 demes: coalescence -- and
# so simulation cost -- grows steeply with N here, so it gets a lower per-deme override.
DEFAULT_N_PER_DEME_BY_K = {50: 2}
DEFAULT_L = int(1e4)
DEFAULT_NUM_M = 100
DEFAULT_MAX_M = 0.25
DEFAULT_MIN_M = 0.01

DEFAULT_RANDOM_SEED = 42
DEFAULT_N_WORKERS = max(cpu_count() - 1, 1)


def format_m(m):
    """Filename stem for a simulation: m in scientific notation to 4 decimals."""
    return f"m{m:.4e}"


def migration_rates(num_m, min_m, max_m):
    """
    num_m migration rates from max_m down to min_m, log-spaced and denser towards
    max_m: mirror image of geomspace(min_m, max_m), which is denser towards its
    small end.
    """
    if num_m < 1:
        raise ValueError(f"num_m must be at least 1, got {num_m}")
    if not (0.0 < min_m <= max_m):
        raise ValueError(f"need 0 < min_m <= max_m, got min_m={min_m:g}, max_m={max_m:g}")
    if num_m == 1:
        return np.array([max_m])
    mirrored = min_m + max_m - np.geomspace(min_m, max_m, num_m)
    return np.sort(mirrored)[::-1]


def build_parameter_table(K, num_m=DEFAULT_NUM_M, min_m=DEFAULT_MIN_M,
                          max_m=DEFAULT_MAX_M, Ne=DEFAULT_NE,
                          n_per_deme=DEFAULT_N_PER_DEME, L=DEFAULT_L,
                          random_seed=DEFAULT_RANDOM_SEED):
    """One row per simulation over num_m log-spaced migration rates, for a given K."""
    m_values = migration_rates(num_m, min_m, max_m)

    params = pd.DataFrame({
        "simID": [format_m(m) for m in m_values],
        "K": K,
        "m": m_values,
        "Ne": Ne,
        "n_per_deme": n_per_deme,
        "N": K * n_per_deme,
        "L": L,
        "random_seed": random_seed + np.arange(num_m),
    })

    duplicates = params.simID.duplicated()
    if duplicates.any():
        raise ValueError(
            f"{int(duplicates.sum())} m values collide after rounding to 4 decimal "
            f"places (e.g. {params.simID[duplicates].iloc[0]}); use fewer values or "
            f"a wider range."
        )
    return params


def simulate_one(simID, K, m, Ne, n_per_deme, L, outdir, random_seed, n_workers=1):
    """Simulate one migration rate and save its genotype matrix as <simID>.npy."""
    demography = msprime.Demography.stepping_stone_model(
        initial_size=[Ne] * K, migration_rate=m, boundaries=True)
    X = simulate_haplotypes(
        demography, n=n_per_deme / 2, L=L, random_seed=random_seed,
        n_workers=n_workers)
    X = X.astype(np.bool_)
    np.save(os.path.join(outdir, f"{simID}.npy"), X)
    return X.shape[1]


def run(K_values=DEFAULT_K_VALUES, num_m=DEFAULT_NUM_M, min_m=DEFAULT_MIN_M,
        max_m=DEFAULT_MAX_M, Ne=DEFAULT_NE, n_per_deme=DEFAULT_N_PER_DEME,
        n_per_deme_by_K=DEFAULT_N_PER_DEME_BY_K,
        L=DEFAULT_L, outdir_template=DEFAULT_OUTDIR_TEMPLATE,
        random_seed=DEFAULT_RANDOM_SEED, n_workers=DEFAULT_N_WORKERS,
        dry_run=False, verbose=True):
    """
    Run the full sweep: one migration-rate parameter table and simulation loop per K.

    n_per_deme is the default haploid samples per deme; n_per_deme_by_K overrides it
    for specific K values (e.g. large K needs fewer samples per deme to stay tractable).

    Each simulation's L replicate coalescent trees are independent, so simulate_haplotypes
    splits them across n_workers processes; one simulation runs at a time, using all
    n_workers for it, rather than running multiple simulations concurrently.
    """
    for K in K_values:
        outdir = outdir_template.format(K=K)
        K_n_per_deme = n_per_deme_by_K.get(K, n_per_deme)
        params = build_parameter_table(
            K, num_m=num_m, min_m=min_m, max_m=max_m, Ne=Ne,
            n_per_deme=K_n_per_deme, L=L, random_seed=random_seed)

        if verbose:
            print(f"K={K}: {len(params)} simulations, m in "
                  f"[{params.m.min():.4e}, {params.m.max():.4e}], "
                  f"n_per_deme={K_n_per_deme}, N={K * K_n_per_deme}, L={L}, "
                  f"n_workers={n_workers}")
            print(f"output -> {outdir}")

        if dry_run:
            continue

        os.makedirs(outdir, exist_ok=True)

        # smallest m first: those simulations are far cheaper (less migration to
        # simulate), so this front-loads the fast runs rather than the slow ones
        run_order = params.iloc[::-1]
        bar = tqdm(run_order.itertuples(), total=len(run_order), disable=not verbose)
        for i, r in enumerate(bar, start=1):
            if os.path.exists(os.path.join(outdir, f"{r.simID}.npy")):
                bar.set_description(f"skipping simulation {i}/{num_m} (exists), m={r.m:.4g}")
                continue
            bar.set_description(f"running simulation {i}/{num_m}, m={r.m:.4g}")
            simulate_one(r.simID, K, r.m, Ne, K_n_per_deme, L, outdir, int(r.random_seed),
                         n_workers=n_workers)

        params_path = os.path.join(outdir, "simulation_parameters.csv")
        params.to_csv(params_path, index=False)
        if verbose:
            print(f"wrote {len(params)} simulations to {outdir}")
            print(f"parameters -> {params_path}\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, nargs="+", default=DEFAULT_K_VALUES, help=f"deme counts to sweep (default: {DEFAULT_K_VALUES})")
    p.add_argument("--Ne", type=float, default=DEFAULT_NE, help=f"effective size per deme (default: {DEFAULT_NE:g})")
    p.add_argument("--n-per-deme", type=int, default=DEFAULT_N_PER_DEME, help=f"haploid samples per deme (default: {DEFAULT_N_PER_DEME})")
    p.add_argument("--n-per-deme-by-K", type=str, nargs="+", default=[f"{K}:{n}" for K, n in DEFAULT_N_PER_DEME_BY_K.items()], metavar="K:n", help=f"per-K overrides of --n-per-deme, as K:n pairs (default: {DEFAULT_N_PER_DEME_BY_K})")
    p.add_argument("--L", type=int, default=DEFAULT_L, help=f"number of independent SNPs (default: {DEFAULT_L:g})")
    p.add_argument("--num-m", type=int, default=DEFAULT_NUM_M, help=f"number of log-spaced migration rates (default: {DEFAULT_NUM_M})")
    p.add_argument("--min-m", type=float, default=DEFAULT_MIN_M, help=f"lower end of the migration-rate sweep (default: {DEFAULT_MIN_M:g})")
    p.add_argument("--max-m", type=float, default=DEFAULT_MAX_M, help=f"upper end of the migration-rate sweep (default: {DEFAULT_MAX_M:g})")
    p.add_argument("--outdir-template", default=DEFAULT_OUTDIR_TEMPLATE, help="output directory, with '{K}' substituted per deme count")
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS, help=f"processes to split each simulation's L replicates across (default: {DEFAULT_N_WORKERS}, i.e. cpu_count - 1)")
    p.add_argument("--dry-run", action="store_true", help="report the plan and exit without simulating")
    args = p.parse_args()
    n_per_deme_by_K = {int(k): int(n) for k, n in
                        (pair.split(":") for pair in args.n_per_deme_by_K)}

    run(K_values=args.K, num_m=args.num_m, min_m=args.min_m, max_m=args.max_m,
        Ne=args.Ne, n_per_deme=args.n_per_deme, n_per_deme_by_K=n_per_deme_by_K,
        L=args.L, outdir_template=args.outdir_template, random_seed=args.random_seed,
        n_workers=args.n_workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

# %%
