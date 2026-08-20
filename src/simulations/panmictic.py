#%%
"""
Simulate haplotypes from a single panmictic population.

One constant-size population, no structure, so the resulting genotype matrix has no
structure eigenvalue and its spectrum is pure floor. Used to characterize the
variance-effective SNP count L_eff independently of any population structure.

A single large simulation is run; analyses subsample N and L from it rather than
re-simulating, which keeps the sampling comparison within one realized allele frequency
spectrum.

Two output modes:

    --counts (default)  the derived allele count of each SNP, one int32 per locus. Any
        quantity depending on the genotype matrix only through its column frequencies --
        L_eff and the site frequency spectrum among them -- is recovered exactly from
        these, and a smaller N is obtained by hypergeometric down-sampling
        (utils.subsample_allele_counts). Costs O(L) storage instead of O(N*L): at
        N = 1e4, L = 1e6 that is 4 MB rather than a 10 GB matrix that would not fit in
        memory as float32 anyway.
    --genotypes         the full N x L boolean matrix, for analyses that need the
        genotypes themselves (a spectrum, a PCA). Only feasible at modest N * L.

Example:
    python panmictic.py                                  # counts, N=10000, L=1e6
    python panmictic.py --N 1000 --L 160000 --genotypes  # a full matrix

Outputs (results/simulations/panmictic by default):
    simulation_parameters.csv   one row per simulation
    N<N>_L<L>_counts.npy        derived allele counts (int32), --counts mode
    N<N>_L<L>.npy               the N x L genotype matrix (np.bool_), --genotypes mode
"""
import argparse
import os
import sys
from multiprocessing import cpu_count

import msprime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from utils import simulate_haplotypes, simulate_allele_counts

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTDIR = os.path.join(_PROJECT_ROOT, "results", "simulations", "panmictic")

DEFAULT_NE = 1e4
DEFAULT_N = 10000           # haploid samples
DEFAULT_L = int(1e6)
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_WORKERS = max(cpu_count() - 3, 1)


def format_simID(N, L, counts=False):
    """Filename stem for a simulation."""
    return f"N{N}_L{L}_counts" if counts else f"N{N}_L{L}"


def panmictic_demography(Ne=DEFAULT_NE):
    """A single constant-size population."""
    demography = msprime.Demography()
    demography.add_population(name="POP", initial_size=Ne)
    return demography


def run(N=DEFAULT_N, L=DEFAULT_L, Ne=DEFAULT_NE, outdir=DEFAULT_OUTDIR,
        random_seed=DEFAULT_RANDOM_SEED, n_workers=DEFAULT_N_WORKERS, counts=True,
        verbose=True):
    """Simulate one panmictic dataset and save it, as allele counts or genotypes."""
    os.makedirs(outdir, exist_ok=True)
    simID = format_simID(N, L, counts=counts)
    path = os.path.join(outdir, f"{simID}.npy")

    if os.path.exists(path):
        if verbose:
            print(f"{path} already exists, skipping")
        return path

    if verbose:
        mode = "allele counts" if counts else "genotype matrix"
        size_gb = N * L / 1e9
        print(f"panmictic ({mode}): N={N} haploid samples, L={L:g} SNPs, Ne={Ne:g}, "
              f"n_workers={n_workers}")
        print(f"  output ~{4 * L / 1e6:.0f} MB" if counts
              else f"  output ~{size_gb:.1f} GB")

    # Both simulators take diploid individuals and yield 2*n haploid samples.
    if counts:
        values, n_haploid = simulate_allele_counts(
            panmictic_demography(Ne), n=N / 2, L=L, random_seed=random_seed,
            n_workers=n_workers, show_progress=verbose)
        assert n_haploid == N, f"expected {N} haploid samples, got {n_haploid}"
        np.save(path, values.astype(np.int32))
        shape = values.shape
    else:
        X = simulate_haplotypes(panmictic_demography(Ne), n=N / 2, L=L,
                                random_seed=random_seed, n_workers=n_workers,
                                show_progress=verbose)
        np.save(path, X.astype(np.bool_))
        shape = X.shape

    row = dict(simID=simID, N=N, L=L, Ne=Ne, random_seed=random_seed,
               kind="counts" if counts else "genotypes")
    params_path = os.path.join(outdir, "simulation_parameters.csv")
    if os.path.exists(params_path):
        prev = pd.read_csv(params_path)
        prev = prev[prev.simID != simID]                 # replace any stale row
        out = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.to_csv(params_path, index=False)

    if verbose:
        print(f"wrote {path}  shape={shape}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=DEFAULT_N,
                   help=f"haploid sample size (default: {DEFAULT_N})")
    p.add_argument("--L", type=int, default=DEFAULT_L,
                   help=f"number of independent SNPs (default: {DEFAULT_L:g})")
    p.add_argument("--Ne", type=float, default=DEFAULT_NE)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    p.add_argument("--genotypes", dest="counts", action="store_false",
                   help="save the full N x L genotype matrix instead of allele counts")
    p.set_defaults(counts=True)
    args = p.parse_args()
    run(N=args.N, L=args.L, Ne=args.Ne, outdir=args.outdir,
        random_seed=args.random_seed, n_workers=args.n_workers, counts=args.counts)


if __name__ == "__main__":
    main()

# %%
