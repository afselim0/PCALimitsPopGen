"""
A single constant-size panmictic population -- no structure, pure floor spectrum. Used
to characterize the variance-effective SNP count L_eff independently of any structure.

No sweep: build_parameter_table always returns one row. fixed.counts (default True)
selects allele-counts output (int32, O(L) storage) vs the full genotype matrix (np.bool_,
O(N*L)) -- see utils.simulate_allele_counts / simulate_haplotypes.

Output: N<N>_L<L>_counts.npy (counts mode) or N<N>_L<L>.npy (genotypes mode).
"""
import os
import sys

import msprime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # src/
from utils import simulate_haplotypes, simulate_allele_counts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/simulations/
from sweep import run_table, run_one_row


def format_simID(N, L, counts=False):
    return f"N{N}_L{L}_counts" if counts else f"N{N}_L{L}"


def demography(Ne):
    dem = msprime.Demography()
    dem.add_population(name="POP", initial_size=Ne)
    return dem


def build_parameter_table(cfg):
    fixed = cfg["fixed"]
    N, L, Ne = fixed["N"], fixed["L"], fixed["Ne"]
    counts = fixed.get("counts", True)
    return pd.DataFrame([dict(
        simID=format_simID(N, L, counts=counts),
        N=N, L=L, Ne=Ne, counts=counts,
        random_seed=cfg["random_seed"],
    )])


def simulate_one(row, cfg, outdir, n_workers):
    dem = demography(row.Ne)
    path = os.path.join(outdir, f"{row.simID}.npy")
    if row.counts:
        values, n_haploid = simulate_allele_counts(
            dem, n=row.N / 2, L=row.L, random_seed=int(row.random_seed), n_workers=n_workers)
        assert n_haploid == row.N, f"expected {row.N} haploid samples, got {n_haploid}"
        np.save(path, values.astype(np.int32))
    else:
        X = simulate_haplotypes(
            dem, n=row.N / 2, L=row.L, random_seed=int(row.random_seed), n_workers=n_workers)
        np.save(path, X)


def run(cfg, outdir, n_workers, dry_run=False, verbose=True):
    table = build_parameter_table(cfg)
    r = table.iloc[0]
    if verbose:
        mode = "allele counts" if r.counts else "genotype matrix"
        print(f"panmictic ({mode}): N={r.N}, L={r.L:g}, Ne={r.Ne:g}, n_workers={n_workers}")
        print(f"output -> {outdir}")
    return run_table(table, lambda row: simulate_one(row, cfg, outdir, n_workers),
                     outdir, dry_run=dry_run, verbose=verbose, desc="panmictic")


def run_one(task_index, cfg, outdir, n_workers, verbose=True):
    table = build_parameter_table(cfg)
    run_one_row(task_index, table, lambda row: simulate_one(row, cfg, outdir, n_workers),
               outdir, verbose=verbose)
