"""
stdpopsim's HomSap/OutOfAfrica_3G09 (Gutenkunst et al. 2009), extended with a recent
split within the African (YRI) population, over a sweep of split times t_split.

The base model has three populations (YRI, CEU, CHB) with the standard OOA / CEU-CHB
split history. A fourth population, YRI2, is added: going backward in time, YRI and
YRI2 are both sampled at present and merge into a single YRI lineage at t_split
generations ago (msprime's add_population_split), rejoining the unmodified OOA history
above that. YRI2 inherits YRI's migration rates with CEU and CHB.

The theoretical Fst between YRI and YRI2 is computed exactly from the demography via
DemographyDebugger.mean_coalescence_time, as 1 - t_within / t_between.

fixed.n_per_pop may be a scalar (uniform across CEU/CHB/YRI/YRI2) or a per-population
dict, e.g. {CEU: 5000, CHB: 5000, YRI: 2000, YRI2: 2000}.

Output: t<t_split>_Fst<Fst>.npy, the (sum of n_per_pop) x L genotype matrix (np.bool_).
Sample order is alphabetical by population: CEU, CHB, YRI, YRI2.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import stdpopsim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # src/
from utils import simulate_haplotypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/simulations/
from sweep import axis_values, check_no_collisions, run_table, run_one_row

POPS = ["CEU", "CHB", "YRI", "YRI2"]


def format_simID(t_split, Fst):
    return f"t{t_split:.4e}_Fst{Fst:.4e}"


def _n_per_pop_dict(n_per_pop):
    """Normalize a scalar or {pop: n} mapping to a full {pop: n} dict over POPS."""
    if isinstance(n_per_pop, dict):
        missing = [p for p in POPS if p not in n_per_pop]
        if missing:
            raise ValueError(f"n_per_pop is missing populations: {missing}")
        return {p: int(n_per_pop[p]) for p in POPS}
    return {p: int(n_per_pop) for p in POPS}


def demography(t_split):
    """
    stdpopsim's OutOfAfrica_3G09 with a fourth population YRI2 split off YRI at
    t_split generations ago. YRI2 mirrors YRI's migration rates with CEU/CHB.
    """
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("OutOfAfrica_3G09")
    dbg = copy.deepcopy(model.model)  # model.model is cached/shared -- copy before mutating

    yri = dbg["YRI"].id
    dbg.add_population(name="YRI2", initial_size=dbg["YRI"].initial_size, initially_active=True)
    yri2 = dbg["YRI2"].id

    dbg.migration_matrix[yri2, :] = dbg.migration_matrix[yri, :]
    dbg.migration_matrix[:, yri2] = dbg.migration_matrix[:, yri]
    dbg.migration_matrix[yri2, yri] = 0
    dbg.migration_matrix[yri, yri2] = 0

    dbg.add_population_split(time=t_split, derived=["YRI2"], ancestral="YRI")
    dbg["YRI"].initially_active = True  # still sampled at present, not just post-split
    dbg.sort_events()
    return dbg


def theoretical_Fst(dem):
    """Exact Fst(YRI, YRI2) = 1 - t_within / t_between, both via DemographyDebugger."""
    d = dem.debug()
    t_within = d.mean_coalescence_time(lineages={"YRI": 2})
    t_between = d.mean_coalescence_time(lineages={"YRI": 1, "YRI2": 1})
    return 1.0 - t_within / t_between


def build_parameter_table(cfg):
    fixed, sweep = cfg["fixed"], cfg["sweep"]
    n_dict = _n_per_pop_dict(fixed["n_per_pop"])
    N = sum(n_dict.values())
    L = fixed["L"]
    random_seed = cfg["random_seed"]

    t_values = axis_values(sweep["min"], sweep["max"], sweep["num"], sweep.get("spacing", "log"))
    Fst_values = [theoretical_Fst(demography(t)) for t in t_values]
    simIDs = [format_simID(t, f) for t, f in zip(t_values, Fst_values)]
    check_no_collisions(simIDs, "t_split values")

    return pd.DataFrame({
        "simID": simIDs,
        "t_split": t_values,
        "Fst": Fst_values,
        "n_CEU": n_dict["CEU"], "n_CHB": n_dict["CHB"],
        "n_YRI": n_dict["YRI"], "n_YRI2": n_dict["YRI2"],
        "N": N, "L": L,
        "random_seed": random_seed + np.arange(len(t_values)),
    })


def simulate_one(row, cfg, outdir, n_workers):
    n_dict = _n_per_pop_dict(cfg["fixed"]["n_per_pop"])
    dem = demography(row.t_split)
    X = simulate_haplotypes(dem, n={p: n / 2 for p, n in n_dict.items()},
                            L=cfg["fixed"]["L"], random_seed=int(row.random_seed),
                            n_workers=n_workers)
    np.save(os.path.join(outdir, f"{row.simID}.npy"), X)


def run(cfg, outdir, n_workers, dry_run=False, verbose=True):
    table = build_parameter_table(cfg)
    if verbose:
        print(f"african_split: {len(table)} simulations, t_split in "
              f"[{table.t_split.min():.4e}, {table.t_split.max():.4e}], "
              f"n_per_pop={_n_per_pop_dict(cfg['fixed']['n_per_pop'])}, "
              f"N={table.N.iloc[0]}, L={cfg['fixed']['L']}, n_workers={n_workers}")
        print(f"output -> {outdir}")
    return run_table(table, lambda row: simulate_one(row, cfg, outdir, n_workers),
                     outdir, dry_run=dry_run, verbose=verbose, desc="african_split")


def run_one(task_index, cfg, outdir, n_workers, verbose=True):
    table = build_parameter_table(cfg)
    run_one_row(task_index, table, lambda row: simulate_one(row, cfg, outdir, n_workers),
               outdir, verbose=verbose)
