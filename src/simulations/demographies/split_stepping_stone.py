"""
Two populations split from a common ancestor at a fixed, recent t_split; population A
stays panmictic for its whole post-split history, population B immediately fragments
into a K x K grid of demes (nearest-neighbour migration, reflecting boundaries, no
wraparound -- identical construction to stepping_stone_2d).

K, Ne_ancestral, t_split and L are fixed parameters. The swept parameter is 1/M for
migration between B's demes, log-spaced over [sweep.inv_M_min, sweep.inv_M_max] in
sweep.num_inv_M steps. M = 4 Ne_deme m with Ne_deme = Ne_ancestral / K^2, so the deme
size implied by Ne_ancestral is what converts the scaled range into migration rates --
see stepping_stone_grid.inverse_M_grid.

Ne_B (B's size at the moment of the split, before fragmenting) is fixed to Ne_ancestral
-- simple census-size conservation, not solved for a target effective size. Each of B's
K^2 demes then gets initial_size = Ne_ancestral / K**2.

Output directory: K<K>, with its own simulation_parameters.csv and invM<value>.npy --
same convention as stepping_stone_2d.
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
from stepping_stone_grid import inverse_M_grid, sweep_plan

DIMENSION = 2


def format_inv_M(inv_M):
    return f"invM{inv_M:.4e}"


def cell_name(K):
    return f"K{K}"


def _deme_name(row, col):
    return f"B_{row}_{col}"


def demography(K, Ne_ancestral, t_split, m):
    """
    ANC splits, at t_split, into A (panmictic, Ne_ancestral) and K^2 demes of B (each
    Ne_ancestral / K**2), connected by nearest-neighbour migration at rate m from
    t_split to the present -- the same grid construction as stepping_stone_2d.demography,
    layered under a single recent split rather than being the whole history.

    msprime rejects a single population-split event that touches more than 100
    populations (derived + ancestral) at once, so K must satisfy K^2 + 1 <= 100 (K <= 9).
    """
    Ne_deme = Ne_ancestral / (K * K)
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=Ne_ancestral)
    for row in range(K):
        for col in range(K):
            dem.add_population(name=_deme_name(row, col), initial_size=Ne_deme)
    dem.add_population(name="ANC", initial_size=Ne_ancestral)

    for row in range(K):
        for col in range(K):
            deme = _deme_name(row, col)
            if col + 1 < K:
                dem.set_symmetric_migration_rate([deme, _deme_name(row, col + 1)], m)
            if row + 1 < K:
                dem.set_symmetric_migration_rate([deme, _deme_name(row + 1, col)], m)

    derived = ["A"] + [_deme_name(row, col) for row in range(K) for col in range(K)]
    dem.add_population_split(time=t_split, derived=derived, ancestral="ANC")
    dem.sort_events()
    return dem


def build_parameter_table(cfg):
    fixed, sweep = cfg["fixed"], cfg["sweep"]
    K = fixed["K"]
    Ne_ancestral, t_split, L = fixed["Ne_ancestral"], fixed["t_split"], fixed["L"]
    n_per_pop_A, n_per_deme_B = fixed["n_per_pop_A"], fixed["n_per_deme_B"]
    random_seed = cfg["random_seed"]

    # M = 4 * Ne_deme * m here: the individual deme's size, Ne_ancestral / K**2, not
    # Ne_ancestral itself. That is also the Ne sweep_plan needs.
    #
    # The plain stepping-stone m_c formula assumes migration-drift equilibrium; a
    # population that only recently fragmented (t_split) hasn't equilibrated, so m_c/M_c
    # are recorded for reference only.
    Ne_deme = Ne_ancestral / (K * K)
    inv_M, m_values = inverse_M_grid(sweep["inv_M_min"], sweep["inv_M_max"],
                                     sweep["num_inv_M"], Ne_deme,
                                     sweep.get("spacing", "log"))
    cell = sweep_plan([K], n_per_deme_B, L, dimension=DIMENSION, Ne=Ne_deme)[0]
    simIDs = [format_inv_M(v) for v in inv_M]
    check_no_collisions(simIDs, "1/M values")

    return pd.DataFrame({
        "simID": simIDs,
        "outdir_suffix": cell_name(K),
        "K": K, "K_demes": cell["K_demes"], "m": m_values,
        "M": 4 * Ne_deme * m_values, "inv_M": inv_M,
        "M_c": cell["M_c"], "M_c_approx": cell["M_c_approx"], "m_c": cell["m_c"],
        "Ne_ancestral": Ne_ancestral, "Ne_deme": Ne_deme, "t_split": t_split,
        "n_per_pop_A": n_per_pop_A, "n_per_deme_B": n_per_deme_B,
        "N": n_per_pop_A + cell["N"], "L": L,
        "random_seed": random_seed + np.arange(len(m_values)),
    })


def simulate_one(row, cfg, outdir, n_workers):
    dem = demography(row.K, row.Ne_ancestral, row.t_split, row.m)
    n = {"A": row.n_per_pop_A / 2}
    n.update({_deme_name(r, c): row.n_per_deme_B / 2
             for r in range(row.K) for c in range(row.K)})
    X = simulate_haplotypes(dem, n=n, L=row.L, random_seed=int(row.random_seed),
                            n_workers=n_workers, show_progress=False)
    row_dir = os.path.join(outdir, row.outdir_suffix)
    np.save(os.path.join(row_dir, f"{row.simID}.npy"), X)


def run(cfg, outdir, n_workers, dry_run=False, verbose=True):
    table = build_parameter_table(cfg)
    if verbose:
        print(f"{table.outdir_suffix.iloc[0]}: {len(table)} simulations "
              f"({table.K_demes.iloc[0]} B-demes), Ne_ancestral={table.Ne_ancestral.iloc[0]:g}, "
              f"t_split={table.t_split.iloc[0]:g}, N={table.N.iloc[0]}, L={table.L.iloc[0]}, "
              f"M_c={table.M_c.iloc[0]:.1f}, m_c={table.m_c.iloc[0]:.4g}")
        print(f"output -> {outdir}")
    return run_table(table, lambda row: simulate_one(row, cfg, outdir, n_workers),
                     outdir, dry_run=dry_run, verbose=verbose, desc="split_stepping_stone")


def run_one(task_index, cfg, outdir, n_workers, verbose=True):
    table = build_parameter_table(cfg)
    run_one_row(task_index, table, lambda row: simulate_one(row, cfg, outdir, n_workers),
               outdir, verbose=verbose)
