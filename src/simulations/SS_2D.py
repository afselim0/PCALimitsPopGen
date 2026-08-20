#%%
"""
Simulate 2D stepping-stone haplotypes over a sweep of migration rates, for each of
several grid side lengths K, on a K x K grid of demes with nearest-neighbour (rook's
move) migration and reflecting boundaries.

msprime.Demography.stepping_stone_model only supports 1D, so the K x K grid demography
is built directly: K^2 isolated demes, each connected to its up/down/left/right
neighbours at rate m. Internal demes have 4 neighbours, edge (non-corner) demes have 3,
and the 4 corner demes have 2 -- no wraparound, matching the 1D script's boundaries=True.

As in SS_1D, the migration grid is built per K around that K's own critical rate rather
than from a fixed [min_m, max_m], so every cell brackets its transition, and Ne is
solved per K so that m_c lands on --target-m. M_c grows only like K here (against
K^(3/2) in 1D), so the Ne values needed stay far smaller than the 1D ones -- but the
solve still matters: at a fixed Ne = 500 the largest cell would run out to a migration
rate well past anything that behaves like a stepping stone.

K is capped well below the 1D deme counts because cost scales with the number of demes,
K^2: the K = 10 cell already carries 100 demes and dominates the sweep at ~2.5 h serial.
n_per_deme = 10 lets the analysis subsample to n = 2, 4, 6, 10 for an N sweep at fixed K.
As in 1D the grid's top end drives cost, so n, L and span are sized against it rather
than chosen for resolution alone; see SS_1D for the measured numbers.

The leading eigenvalue is doubly degenerate here (mu_10 = mu_01), so the recoverable
object is a 2D subspace, not a single PC. The analysis must compare subspaces -- see
results/plotting_scripts/SteppingStoneDetection.py, which passes both axes.

Example:
    python SS_2D.py                                   # the default K sweep
    python SS_2D.py --dry-run                         # print the plan, simulate nothing
    python SS_2D.py --K 5 8 --num-m 20                # a quicker two-K sweep
    python SS_2D.py --K 8 --L-sweep 250 1000 4000     # vary L at fixed K

Outputs (results/simulations/SS_2D/K<K>[_L<L>] by default, one subdirectory per cell):
    simulation_parameters.csv   one row per simulation, including M and M_c
    m<value>.npy                the (K^2 * n_per_deme) x L genotype matrix, saved as
                                 np.bool_, with m in scientific notation to 4 decimal
                                 places (e.g. m1.0000e-02.npy)
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
DEFAULT_OUTDIR_ROOT = os.path.join(_PROJECT_ROOT, "results", "simulations", "SS_2D")

DEFAULT_K_VALUES = [4, 5, 6, 8, 10]
DEFAULT_N_PER_DEME = 10
DEFAULT_L = 500
DEFAULT_NUM_M = 40
DEFAULT_SPAN = 3.0
DEFAULT_TARGET_M = 0.02

DEFAULT_RANDOM_SEED = 42
DEFAULT_N_WORKERS = max(cpu_count() - 3, 1)


def format_m(m):
    """Filename stem for a simulation: m in scientific notation to 4 decimals."""
    return f"m{m:.4e}"


def cell_name(K, L, L_is_swept):
    """Subdirectory for one cell: K<K>, with the L appended when L is the swept axis."""
    return f"K{K}_L{L}" if L_is_swept else f"K{K}"


def grid_2d_demography(K, Ne, m):
    """
    K x K grid of isolated demes (deme (row, col) named pop_{row*K+col}), connected to
    their up/down/left/right neighbours at rate m. No wraparound: corner demes get 2
    neighbours, edge demes get 3, internal demes get 4.
    """
    demography = msprime.Demography.isolated_model([Ne] * K * K)
    for row in range(K):
        for col in range(K):
            deme = row * K + col
            if col + 1 < K:
                neighbour = row * K + (col + 1)
                demography.migration_matrix[deme, neighbour] = m
                demography.migration_matrix[neighbour, deme] = m
            if row + 1 < K:
                neighbour = (row + 1) * K + col
                demography.migration_matrix[deme, neighbour] = m
                demography.migration_matrix[neighbour, deme] = m
    return demography


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
        "K_demes": cell["K_demes"],
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
    demography = grid_2d_demography(K, Ne, m)
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
        plan = sweep_plan(K_values, n_per_deme, L, dimension=2, num_m=num_m,
                          span=span, Ne=Ne, target_m=None if Ne else target_m)

        for cell in plan:
            K = cell["K"]
            outdir = os.path.join(outdir_root, cell_name(K, L, L_is_swept))
            params = build_parameter_table(cell, L, n_per_deme, random_seed=random_seed)

            if verbose:
                print(f"K={K} ({K}x{K} grid, {cell['K_demes']} demes): "
                      f"{len(params)} simulations, Ne={cell['Ne']}, "
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
    p.add_argument("--K", type=int, nargs="+", default=DEFAULT_K_VALUES, help=f"grid side lengths to sweep, i.e. K x K demes (default: {DEFAULT_K_VALUES})")
    p.add_argument("--L-sweep", type=int, nargs="+", default=[DEFAULT_L], dest="L_values", help=f"SNP counts to sweep; more than one appends _L<L> to each output directory (default: [{DEFAULT_L}])")
    p.add_argument("--target-m", type=float, default=DEFAULT_TARGET_M, help=f"solve Ne per K so the critical rate lands here (default: {DEFAULT_TARGET_M})")
    p.add_argument("--Ne", type=float, default=None, help="fixed effective size per deme; overrides --target-m and lets m_c move with K")
    p.add_argument("--n-per-deme", type=int, default=DEFAULT_N_PER_DEME, help=f"haploid samples per deme; the analysis subsamples below this for the N sweep (default: {DEFAULT_N_PER_DEME})")
    p.add_argument("--num-m", type=int, default=DEFAULT_NUM_M, help=f"number of migration rates per cell (default: {DEFAULT_NUM_M})")
    p.add_argument("--span", type=float, default=DEFAULT_SPAN, help=f"grid spans [m_c/span, m_c*span] (default: {DEFAULT_SPAN})")
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
