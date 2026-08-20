"""
Two-population split demography (pop A, pop B from ancestral ANC at time t_split), with
Fst = t_split / (2*Ne), swept over Fst.

Output: Fst<Fst>.npy, the (2 * n_per_pop) x L genotype matrix (np.bool_).
Sample order: pop A then pop B.
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


def format_simID(Fst):
    return f"Fst{Fst:.4e}"


def demography(Fst, Ne):
    t_split = 2 * Ne * Fst
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=Ne)
    dem.add_population(name="B", initial_size=Ne)
    dem.add_population(name="ANC", initial_size=Ne)
    dem.add_population_split(time=t_split, derived=["A", "B"], ancestral="ANC")
    dem.sort_events()
    return dem


def build_parameter_table(cfg):
    fixed, sweep = cfg["fixed"], cfg["sweep"]
    n_per_pop, L, Ne = fixed["n_per_pop"], fixed["L"], fixed["Ne"]
    random_seed = cfg["random_seed"]

    Fst_values = axis_values(sweep["min"], sweep["max"], sweep["num"], sweep.get("spacing", "log"))
    simIDs = [format_simID(f) for f in Fst_values]
    check_no_collisions(simIDs, "Fst values")

    return pd.DataFrame({
        "simID": simIDs,
        "Fst": Fst_values,
        "n_per_pop": n_per_pop, "N": 2 * n_per_pop, "L": L, "Ne": Ne,
        "random_seed": random_seed + np.arange(len(Fst_values)),
    })


def simulate_one(row, cfg, outdir, n_workers):
    dem = demography(row.Fst, row.Ne)
    X = simulate_haplotypes(dem, n=row.n_per_pop / 2, L=row.L,
                            random_seed=int(row.random_seed), n_workers=n_workers)
    np.save(os.path.join(outdir, f"{row.simID}.npy"), X)


def run(cfg, outdir, n_workers, dry_run=False, verbose=True):
    table = build_parameter_table(cfg)
    if verbose:
        print(f"split: {len(table)} simulations, Fst in "
              f"[{table.Fst.min():.4e}, {table.Fst.max():.4e}], "
              f"n_per_pop={cfg['fixed']['n_per_pop']}, N={table.N.iloc[0]}, "
              f"L={cfg['fixed']['L']}, Ne={cfg['fixed']['Ne']:g}, n_workers={n_workers}")
        print(f"output -> {outdir}")
    return run_table(table, lambda row: simulate_one(row, cfg, outdir, n_workers),
                     outdir, dry_run=dry_run, verbose=verbose, desc="split")


def run_one(task_index, cfg, outdir, n_workers, verbose=True):
    table = build_parameter_table(cfg)
    run_one_row(task_index, table, lambda row: simulate_one(row, cfg, outdir, n_workers),
               outdir, verbose=verbose)
