"""
1D stepping-stone haplotypes over a sweep of migration rates, for each of several deme
counts K, using msprime.Demography.stepping_stone_model with reflecting boundaries.

The migration grid is normally built per K around that K's own critical rate
(stepping_stone_grid.sweep_plan), not a fixed [min, max] -- M_c grows like K^(3/2) in 1D,
so a shared range cannot bracket the transition across a K sweep in general. Ne is solved
per K so m_c lands on sweep.target_m (or fixed directly via fixed.Ne). When sweep.min and
sweep.max are both given, they override this and are used directly (log- or
linear-spaced per sweep.spacing) -- useful for a single, fixed K where the derived grid
isn't wanted.

Output directories: one per (K[, L]) cell, K<K> or K<K>_L<L> if L is swept, each with its
own simulation_parameters.csv and m<value>.npy files -- this on-disk convention is relied
on by src/analysis/stepping_stone_detection.py and must not change.
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
from stepping_stone_grid import sweep_plan

DIMENSION = 1


def format_m(m):
    return f"m{m:.4e}"


def cell_name(K, L, L_is_swept):
    return f"K{K}_L{L}" if L_is_swept else f"K{K}"


def demography(K, Ne, m):
    return msprime.Demography.stepping_stone_model(
        initial_size=[Ne] * K, migration_rate=m, boundaries=True)


def build_parameter_table(cfg):
    fixed, sweep = cfg["fixed"], cfg["sweep"]
    n_per_deme = fixed["n_per_deme"]
    random_seed = cfg["random_seed"]
    L_values = sweep.get("L_values", [fixed.get("L", 500)])
    L_is_swept = len(L_values) > 1
    Ne_fixed = fixed.get("Ne", sweep.get("Ne"))
    explicit_range = "min" in sweep and "max" in sweep

    frames = []
    for L in L_values:
        plan = sweep_plan(sweep["K_values"], n_per_deme, L, dimension=DIMENSION,
                          num_m=sweep.get("num_m", 40), span=sweep.get("span", 3.0),
                          Ne=Ne_fixed, target_m=None if Ne_fixed else sweep.get("target_m", 0.05))
        for cell in plan:
            m_values = (axis_values(sweep["min"], sweep["max"], sweep.get("num_m", 40),
                                    sweep.get("spacing", "log"))
                       if explicit_range else cell["m_values"])
            simIDs = [format_m(m) for m in m_values]
            check_no_collisions(simIDs, "m values")
            frames.append(pd.DataFrame({
                "simID": simIDs,
                "outdir_suffix": cell_name(cell["K"], L, L_is_swept),
                "K": cell["K"], "m": m_values,
                "M": 4 * cell["Ne"] * m_values,
                "M_c": cell["M_c"], "M_c_approx": cell["M_c_approx"], "m_c": cell["m_c"],
                "Ne": cell["Ne"], "n_per_deme": n_per_deme, "N": cell["N"], "L": L,
                "random_seed": random_seed + np.arange(len(m_values)),
            }))
    return pd.concat(frames, ignore_index=True)


def simulate_one(row, cfg, outdir, n_workers):
    dem = demography(row.K, row.Ne, row.m)
    X = simulate_haplotypes(dem, n=row.n_per_deme / 2, L=row.L,
                            random_seed=int(row.random_seed), n_workers=n_workers,
                            show_progress=False)
    row_dir = os.path.join(outdir, row.outdir_suffix)
    np.save(os.path.join(row_dir, f"{row.simID}.npy"), X)


def run(cfg, outdir, n_workers, dry_run=False, verbose=True):
    table = build_parameter_table(cfg)
    if verbose:
        for suffix, g in table.groupby("outdir_suffix", sort=False):
            print(f"{suffix}: {len(g)} simulations, Ne={g.Ne.iloc[0]}, N={g.N.iloc[0]}, "
                  f"L={g.L.iloc[0]}, M_c={g.M_c.iloc[0]:.1f}, m_c={g.m_c.iloc[0]:.4g}")
        print(f"output -> {outdir}")
    return run_table(table, lambda row: simulate_one(row, cfg, outdir, n_workers),
                     outdir, dry_run=dry_run, verbose=verbose, desc="stepping_stone_1d")


def run_one(task_index, cfg, outdir, n_workers, verbose=True):
    table = build_parameter_table(cfg)
    run_one_row(task_index, table, lambda row: simulate_one(row, cfg, outdir, n_workers),
               outdir, verbose=verbose)
