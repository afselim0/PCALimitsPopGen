#%%
"""
Simulate 1D stepping-stone haplotypes over a sweep of migration rates, for each of
several deme counts K, using msprime.Demography.stepping_stone_model with reflecting
boundaries.

The migration grid is built per K around that K's own critical rate rather than from a
fixed [min_m, max_m]: M_c grows like K^(3/2) in 1D, so a shared range cannot bracket the
transition across a K sweep -- the large-K cells end up entirely on the detectable side
and carry no information about the threshold. stepping_stone_grid.sweep_plan centres
num_m rates on m_c, spanning a factor of `span` either way.

Ne is solved per K so that m_c lands on --target-m (default 0.05). Only M = 4 Ne mbar
enters the theory, so this is free, and it is what keeps m physical: at a fixed Ne = 500
the K = 50 cell would need m_c = 0.80, i.e. near-total replacement each generation.

n_per_deme defaults to 10 rather than 2, letting the analysis subsample to n = 2, 4, 6,
10 for an N sweep at fixed K out of one set of runs.

Sizing is set by cost, which is dominated by the *top* of each migration grid: a
high-M simulation is slow because lineages wander between demes for many generations
before coalescing. Measured on this machine, the K = 20 1D cell costs ~16 ms/SNP at the
bottom of its grid and ~130 ms/SNP at the top. Since M_c ~ sqrt(n K^3 L), cutting n and L
lowers the critical rate and so the whole grid, and the saving is superlinear: n = 10,
L = 500, span = 3 runs the full sweep in ~3 h serial, where n = 20, L = 1000, span = 8
would have taken hundreds. K stops at 25 because K = 30 needs m > 1 at the top of its
span, which is no longer a migration rate.

Example:
    python SS_1D.py                                   # the default K sweep
    python SS_1D.py --dry-run                         # print the plan, simulate nothing
    python SS_1D.py --K 10 20 --num-m 20              # a quicker two-K sweep
    python SS_1D.py --K 20 --L-sweep 250 1000 4000    # vary L at fixed K

Outputs (results/simulations/SS_1D/K<K>[_L<L>] by default, one subdirectory per cell):
    simulation_parameters.csv   one row per simulation, including M and M_c
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/simulations/
from stepping_stone_grid import sweep_plan

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTDIR_ROOT = os.path.join(_PROJECT_ROOT, "results", "simulations", "SS_1D")

DEFAULT_K_VALUES = [10, 15, 20, 25]
DEFAULT_N_PER_DEME = 10
DEFAULT_L = 500
DEFAULT_NUM_M = 40
DEFAULT_SPAN = 3.0
DEFAULT_TARGET_M = 0.05

DEFAULT_RANDOM_SEED = 42
DEFAULT_N_WORKERS = max(cpu_count() - 3, 1)


def format_m(m):
    """Filename stem for a simulation: m in scientific notation to 4 decimals."""
    return f"m{m:.4e}"


def cell_name(K, L, L_is_swept):
    """Subdirectory for one cell: K<K>, with the L appended when L is the swept axis."""
    return f"K{K}_L{L}" if L_is_swept else f"K{K}"


def build_parameter_table(cell, L, n_per_deme, random_seed=DEFAULT_RANDOM_SEED):
    """
    One row per simulation for a single cell of the sweep.

    M and M_c travel with the simulations so the analysis never has to reconstruct the
    threshold the grid was built around.
    """
    m_values = cell["m_values"]
    params = pd.DataFrame({
        "simID": [format_m(m) for m in m_values],
        "K": cell["K"],
        "m": m_values,
        "M": 4 * cell["Ne"] * m_values,
        "M_c": cell["M_c"],
        "M_c_approx": cell["M_c_approx"],
        "m_c": cell["m_c"],
        "Ne": cell["Ne"],
        "n_per_deme": n_per_deme,
        "N": cell["N"],
        "L": L,
        "random_seed": random_seed + np.arange(len(m_values)),
    })

    duplicates = params.simID.duplicated()
    if duplicates.any():
        raise ValueError(
            f"{int(duplicates.sum())} m values collide after rounding to 4 decimal "
            f"places (e.g. {params.simID[duplicates].iloc[0]}); use fewer values or "
            f"a wider span."
        )
    return params


def simulate_one(simID, K, m, Ne, n_per_deme, L, outdir, random_seed, n_workers=1):
    """Simulate one migration rate and save its genotype matrix as <simID>.npy."""
    demography = msprime.Demography.stepping_stone_model(
        initial_size=[Ne] * K, migration_rate=m, boundaries=True)
    X = simulate_haplotypes(
        demography, n=n_per_deme / 2, L=L, random_seed=random_seed,
        n_workers=n_workers, show_progress=False)
    X = X.astype(np.bool_)
    np.save(os.path.join(outdir, f"{simID}.npy"), X)
    return X.shape[1]


def run(K_values=DEFAULT_K_VALUES, L_values=(DEFAULT_L,), num_m=DEFAULT_NUM_M,
        span=DEFAULT_SPAN, target_m=DEFAULT_TARGET_M, Ne=None,
        n_per_deme=DEFAULT_N_PER_DEME, outdir_root=DEFAULT_OUTDIR_ROOT,
        random_seed=DEFAULT_RANDOM_SEED, n_workers=DEFAULT_N_WORKERS,
        dry_run=False, verbose=True):
    """
    Run the sweep: one migration grid per (K, L) cell, each centred on its own m_c.

    Each simulation's L replicate coalescent trees are independent, so simulate_haplotypes
    splits them across n_workers processes; one simulation runs at a time, using all
    n_workers for it, rather than running multiple simulations concurrently.
    """
    L_values = list(L_values)
    L_is_swept = len(L_values) > 1

    for L in L_values:
        plan = sweep_plan(K_values, n_per_deme, L, dimension=1, num_m=num_m,
                          span=span, Ne=Ne, target_m=None if Ne else target_m)

        for cell in plan:
            K = cell["K"]
            outdir = os.path.join(outdir_root, cell_name(K, L, L_is_swept))
            params = build_parameter_table(cell, L, n_per_deme, random_seed=random_seed)

            if verbose:
                print(f"K={K}: {len(params)} simulations, Ne={cell['Ne']}, "
                      f"N={cell['N']}, L={L}")
                print(f"  M_c={cell['M_c']:.1f} (closed form {cell['M_c_approx']:.1f}), "
                      f"m_c={cell['m_c']:.4g}")
                print(f"  m in [{params.m.min():.4e}, {params.m.max():.4e}] -> "
                      f"M in [{params.M.min():.1f}, {params.M.max():.1f}], "
                      f"{int((params.M < cell['M_c']).sum())}/{len(params)} below M_c")
                print(f"  output -> {outdir}")

            if dry_run:
                continue

            os.makedirs(outdir, exist_ok=True)

            # smallest m first: those simulations are far cheaper (less migration to
            # simulate), so this front-loads the fast runs rather than the slow ones
            bar = tqdm(params.itertuples(), total=len(params), disable=not verbose)
            for i, r in enumerate(bar, start=1):
                if os.path.exists(os.path.join(outdir, f"{r.simID}.npy")):
                    bar.set_description(f"K={K} {i}/{len(params)} (exists), m={r.m:.4g}")
                    continue
                bar.set_description(f"K={K} {i}/{len(params)}, m={r.m:.4g}")
                simulate_one(r.simID, K, r.m, r.Ne, n_per_deme, L, outdir,
                             int(r.random_seed), n_workers=n_workers)

            params_path = os.path.join(outdir, "simulation_parameters.csv")
            params.to_csv(params_path, index=False)
            if verbose:
                print(f"  wrote {len(params)} simulations, parameters -> {params_path}\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, nargs="+", default=DEFAULT_K_VALUES, help=f"deme counts to sweep (default: {DEFAULT_K_VALUES})")
    p.add_argument("--L-sweep", type=int, nargs="+", default=[DEFAULT_L], dest="L_values", help=f"SNP counts to sweep; more than one appends _L<L> to each output directory (default: [{DEFAULT_L}])")
    p.add_argument("--n-per-deme", type=int, default=DEFAULT_N_PER_DEME, help=f"haploid samples per deme; the analysis subsamples below this for the N sweep (default: {DEFAULT_N_PER_DEME})")
    p.add_argument("--num-m", type=int, default=DEFAULT_NUM_M, help=f"number of migration rates per cell (default: {DEFAULT_NUM_M})")
    p.add_argument("--span", type=float, default=DEFAULT_SPAN, help=f"grid spans [m_c/span, m_c*span] (default: {DEFAULT_SPAN})")
    p.add_argument("--target-m", type=float, default=DEFAULT_TARGET_M, help=f"solve Ne per K so the critical rate lands here (default: {DEFAULT_TARGET_M})")
    p.add_argument("--Ne", type=float, default=None, help="fixed effective size per deme; overrides --target-m and lets m_c move with K")
    p.add_argument("--outdir-root", default=DEFAULT_OUTDIR_ROOT, help="root output directory, one subdirectory per cell")
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS, help=f"processes to split each simulation's L replicates across (default: {DEFAULT_N_WORKERS})")
    p.add_argument("--dry-run", action="store_true", help="report the plan and exit without simulating")
    args = p.parse_args()

    run(K_values=args.K, L_values=args.L_values, num_m=args.num_m, span=args.span,
        target_m=args.target_m, Ne=args.Ne, n_per_deme=args.n_per_deme,
        outdir_root=args.outdir_root, random_seed=args.random_seed,
        n_workers=args.n_workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

# %%
