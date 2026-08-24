"""
2D stepping-stone haplotypes at a fixed grid side length K, over a sweep of migration
rates, on a K x K grid of demes with nearest-neighbour (rook's move) migration and
reflecting boundaries (no wraparound).

msprime.Demography.stepping_stone_model only supports 1D, so the K x K grid is built
directly here: K^2 isolated demes, each connected to its up/down/left/right neighbours
at rate m.

K, Ne and L are fixed parameters (fixed.K, fixed.Ne, fixed.L). The swept parameter is
1/M, the inverse population-scaled migration rate, log-spaced over
[sweep.inv_M_min, sweep.inv_M_max] in sweep.num_inv_M steps and converted to a migration
rate through M = 4 Ne m -- see stepping_stone_grid.inverse_M_grid.

The leading eigenvalue is doubly degenerate here (mu_10 = mu_01) -- the recoverable
object is a 2D subspace, not a single PC. See src/analysis/SteppingStoneDetection.py.

Output directory: K<K>, with its own simulation_parameters.csv and invM<value>.npy.
"""
import os
import sys

import msprime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # src/
from utils import simulate_haplotypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/simulations/
from sweep import check_no_collisions, run_table, run_one_row
from stepping_stone_grid import (PANMICTIC_M_THRESHOLD, PANMICTIC_TOLERANCE,
                                 inverse_M_grid, maybe_pool, mixing_time, sweep_plan)

DIMENSION = 2


def format_inv_M(inv_M):
    return f"invM{inv_M:.4e}"


def cell_name(K):
    return f"K{K}"


def demography(K, Ne, m, pool_threshold=PANMICTIC_M_THRESHOLD,
               pool_tolerance=PANMICTIC_TOLERANCE):
    """
    K x K isolated demes, connected to their 4-neighbours at rate m. No wraparound:
    corner demes get 2 neighbours, edge demes 3, internal demes 4.

    Above pool_threshold the grid is collapsed into a single panmictic population of size
    Ne * K^2 once lineages have mixed -- see stepping_stone_grid.add_panmictic_phase. This
    matters most here: a K x K grid has K^2 demes, so the structured phase is the expensive
    part, and a 10x10 grid runs 45x faster at m = 0.4. Pass pool_threshold=None to keep the
    fully structured model throughout.
    """
    dem = msprime.Demography.isolated_model([Ne] * K * K)
    for row in range(K):
        for col in range(K):
            deme = row * K + col
            if col + 1 < K:
                nb = row * K + (col + 1)
                dem.migration_matrix[deme, nb] = m
                dem.migration_matrix[nb, deme] = m
            if row + 1 < K:
                nb = (row + 1) * K + col
                dem.migration_matrix[deme, nb] = m
                dem.migration_matrix[nb, deme] = m
    if pool_threshold is None:
        return dem
    return maybe_pool(dem, K, m, Ne, DIMENSION, pool_threshold, pool_tolerance)[0]


def build_parameter_table(cfg):
    fixed, sweep = cfg["fixed"], cfg["sweep"]
    K, n_per_deme, L, Ne = fixed["K"], fixed["n_per_deme"], fixed["L"], fixed["Ne"]
    random_seed = cfg["random_seed"]

    # The sweep is specified in 1/M, the scaled units the theory uses; Ne turns each
    # grid point back into the migration rate msprime needs, via M = 4 Ne m.
    inv_M, m_values = inverse_M_grid(sweep["inv_M_min"], sweep["inv_M_max"],
                                     sweep["num_inv_M"], Ne,
                                     sweep.get("spacing", "log"))
    cell = sweep_plan([K], n_per_deme, L, dimension=DIMENSION, Ne=Ne)[0]
    simIDs = [format_inv_M(v) for v in inv_M]
    check_no_collisions(simIDs, "1/M values")

    # Record where the panmictic switch falls (NaN where the cell stays fully structured),
    # so the on-disk table says which simulations used the approximation.
    pool_threshold = sweep.get("pool_threshold", PANMICTIC_M_THRESHOLD)
    pool_tolerance = sweep.get("pool_tolerance", PANMICTIC_TOLERANCE)
    t_pooled = [mixing_time(K, m, DIMENSION, pool_tolerance)
                if pool_threshold is not None and m > pool_threshold else np.nan
                for m in m_values]

    return pd.DataFrame({
        "simID": simIDs,
        "outdir_suffix": cell_name(K),
        "K": K, "K_demes": cell["K_demes"], "m": m_values,
        "M": 4 * Ne * m_values, "inv_M": inv_M,
        "M_c": cell["M_c"], "M_c_approx": cell["M_c_approx"], "m_c": cell["m_c"],
        "Ne": Ne, "n_per_deme": n_per_deme, "N": cell["N"], "L": L,
        "t_pooled": t_pooled,
        "random_seed": random_seed + np.arange(len(m_values)),
    })


def simulate_one(row, cfg, outdir, n_workers):
    sweep = cfg["sweep"]
    dem = demography(row.K, row.Ne, row.m,
                     pool_threshold=sweep.get("pool_threshold", PANMICTIC_M_THRESHOLD),
                     pool_tolerance=sweep.get("pool_tolerance", PANMICTIC_TOLERANCE))
    X = simulate_haplotypes(dem, n=row.n_per_deme / 2, L=row.L,
                            random_seed=int(row.random_seed), n_workers=n_workers,
                            show_progress=False)
    row_dir = os.path.join(outdir, row.outdir_suffix)
    np.save(os.path.join(row_dir, f"{row.simID}.npy"), X)


def run(cfg, outdir, n_workers, dry_run=False, verbose=True):
    table = build_parameter_table(cfg)
    if verbose:
        print(f"{table.outdir_suffix.iloc[0]}: {len(table)} simulations "
              f"({table.K_demes.iloc[0]} demes), Ne={table.Ne.iloc[0]}, N={table.N.iloc[0]}, "
              f"L={table.L.iloc[0]}, M_c={table.M_c.iloc[0]:.1f}, m_c={table.m_c.iloc[0]:.4g}")
        print(f"output -> {outdir}")
    return run_table(table, lambda row: simulate_one(row, cfg, outdir, n_workers),
                     outdir, dry_run=dry_run, verbose=verbose, desc="stepping_stone_2d")


def run_one(task_index, cfg, outdir, n_workers, verbose=True):
    table = build_parameter_table(cfg)
    run_one_row(task_index, table, lambda row: simulate_one(row, cfg, outdir, n_workers),
               outdir, verbose=verbose)
