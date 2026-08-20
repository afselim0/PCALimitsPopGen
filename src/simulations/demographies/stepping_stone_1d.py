"""
1D stepping-stone haplotypes at a fixed deme count K, over a sweep of migration rates,
using msprime.Demography.stepping_stone_model with reflecting boundaries.

K, Ne and L are fixed parameters (fixed.K, fixed.Ne, fixed.L); only m is swept. The
migration grid is normally built around K's own critical rate
(stepping_stone_grid.sweep_plan) -- Ne solved so m_c lands on sweep.target_m if fixed.Ne
isn't given. When sweep.min and sweep.max are both given, they override this and are used
directly (log- or linear-spaced per sweep.spacing).

Output directory: K<K>, with its own simulation_parameters.csv and m<value>.npy files --
this on-disk convention is relied on by src/analysis/stepping_stone_detection.py and must
not change.
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
from sweep import axis_values, check_no_collisions, run_table, run_one_row
from stepping_stone_grid import (PANMICTIC_M_THRESHOLD, PANMICTIC_TOLERANCE, maybe_pool,
                                 mixing_time, sweep_plan)

DIMENSION = 1


def format_m(m):
    return f"m{m:.4e}"


def cell_name(K):
    return f"K{K}"


def demography(K, Ne, m, pool_threshold=PANMICTIC_M_THRESHOLD,
               pool_tolerance=PANMICTIC_TOLERANCE):
    """
    Linear chain of K demes with reflecting boundaries.

    Above pool_threshold the chain is collapsed into a single panmictic population of size
    Ne * K once lineages have mixed, which is exact to within a per-cent or so and much
    faster -- see stepping_stone_grid.add_panmictic_phase. Pass pool_threshold=None (or a
    value above every m in the sweep) to keep the fully structured model throughout.
    """
    dem = msprime.Demography.stepping_stone_model(
        initial_size=[Ne] * K, migration_rate=m, boundaries=True)
    if pool_threshold is None:
        return dem
    return maybe_pool(dem, K, m, Ne, DIMENSION, pool_threshold, pool_tolerance)[0]


def build_parameter_table(cfg):
    fixed, sweep = cfg["fixed"], cfg["sweep"]
    K, n_per_deme, L = fixed["K"], fixed["n_per_deme"], fixed["L"]
    random_seed = cfg["random_seed"]
    Ne_fixed = fixed.get("Ne")
    explicit_range = "min" in sweep and "max" in sweep

    cell = sweep_plan([K], n_per_deme, L, dimension=DIMENSION,
                      num_m=sweep.get("num_m", 40), span=sweep.get("span", 3.0),
                      Ne=Ne_fixed, target_m=None if Ne_fixed else sweep.get("target_m", 0.05))[0]
    m_values = (axis_values(sweep["min"], sweep["max"], sweep.get("num_m", 40),
                            sweep.get("spacing", "log"))
               if explicit_range else cell["m_values"])
    simIDs = [format_m(m) for m in m_values]
    check_no_collisions(simIDs, "m values")

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
        "K": K, "m": m_values,
        "M": 4 * cell["Ne"] * m_values,
        "M_c": cell["M_c"], "M_c_approx": cell["M_c_approx"], "m_c": cell["m_c"],
        "Ne": cell["Ne"], "n_per_deme": n_per_deme, "N": cell["N"], "L": L,
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
        print(f"{table.outdir_suffix.iloc[0]}: {len(table)} simulations, "
              f"Ne={table.Ne.iloc[0]}, N={table.N.iloc[0]}, L={table.L.iloc[0]}, "
              f"M_c={table.M_c.iloc[0]:.1f}, m_c={table.m_c.iloc[0]:.4g}")
        print(f"output -> {outdir}")
    return run_table(table, lambda row: simulate_one(row, cfg, outdir, n_workers),
                     outdir, dry_run=dry_run, verbose=verbose, desc="stepping_stone_1d")


def run_one(task_index, cfg, outdir, n_workers, verbose=True):
    table = build_parameter_table(cfg)
    run_one_row(task_index, table, lambda row: simulate_one(row, cfg, outdir, n_workers),
               outdir, verbose=verbose)
