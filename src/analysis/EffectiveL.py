#%%
"""
The variance-effective SNP count L_eff against the raw count L, on panmictic genotypes.

The GRM is built from unnormalized genotypes, so column l enters with weight
v_l = p_l(1-p_l) and the aspect ratio governing the bulk is N/L_eff rather than N/L, with

    L_eff = (sum_l v_l)^2 / sum_l v_l^2                (participation ratio)

L_eff counts how many loci *effectively* contribute: it equals L only when every locus
carries the same weight, and falls short of it whenever the allele frequency spectrum is
skewed, as a coalescent spectrum is. This script measures that shortfall.

Panmictic simulations are used deliberately: with no population structure there is no
spike, so L_eff is characterized on its own rather than entangled with a structure
eigenvalue.

Works from allele counts, not genotypes. L_eff depends on the genotype matrix only through
its column frequencies, and a subsample of n out of N_full haploid samples has a
hypergeometric derived allele count, so the whole sweep is computable from one length-L
integer array. At N = 1e4 and L = 1e6 that is 4 MB instead of a 10 GB matrix which would
need 40 GB to hold as float32. Both quantities are exact, not approximations: the
hypergeometric draw *is* the distribution of a subsample's allele count.

A single simulation is subsampled rather than re-simulated per cell, so every (N, L) cell
is drawn from one realized frequency spectrum and the comparison across cells is not
confounded by simulation noise.

Both N and L are swept. N matters because a locus polymorphic in the full sample can be
monomorphic in a subsample, which changes the frequency spectrum L_eff is computed from;
monomorphic loci are dropped, as everywhere else in the project.

Getting L exactly. Loci are filtered to the polymorphic ones *first* and L drawn from that
pool, rather than drawing L and then filtering: L is the x-axis here, so the curves must be
comparable across N. The simulation therefore needs enough loci that even the smallest N
leaves max(L_values) of them segregating; the sweep raises a clear error if it does not.

Outputs (results/analysis/EffectiveL):
    L_eff.csv    one row per (N, L, replicate): L_requested, L_raw, L_eff and their ratio
    summary.csv  the same aggregated to a median per (N, L)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from pc_recovery import SIMULATION_ROOT
from sweeps import ANALYSIS_ROOT
from utils import subsample_allele_counts

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "panmictic")
DEFAULT_NAME = "EffectiveL"

N_VALUES = np.array([200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000])
# Capped at 5e5 rather than the simulated 1e6: only ~59% of loci still segregate once the
# sample is cut to N = 200, so a 1e6 simulation leaves ~594k polymorphic there. Every N
# must draw from the same L grid for the curves to be comparable, so the smallest N sets
# the ceiling. Raising it means simulating more SNPs, not changing this list.
L_VALUES = np.array([1e4, 2e4, 5e4, 1e5, 1.5e5, 2e5, 3e5, 4e5, 5e5], dtype=int)
DEFAULT_NUM_REPLICATES = 5
DEFAULT_SEED = 0


def load_counts(simdir):
    """
    Derived allele counts and the haploid sample size they are counts out of.

    Prefers the widest *_counts.npy in the directory: several panmictic runs may coexist,
    and only the largest has enough loci for the top of the L sweep.
    """
    paths = sorted(glob.glob(os.path.join(simdir, "*_counts.npy")))
    if not paths:
        raise FileNotFoundError(
            f"no *_counts.npy in {simdir}. Run:\n"
            f"    python src/simulations/run_simulation.py "
            f"src/simulations/configs/panmictic.yaml")
    arrays = [np.load(p, mmap_mode="r") for p in paths]
    widest = int(np.argmax([a.size for a in arrays]))
    path = paths[widest]
    N_full = int(re.search(r"N(\d+)_", os.path.basename(path)).group(1))
    return np.asarray(arrays[widest]), N_full, os.path.basename(path)


def effective_L_from_counts(counts, n):
    """
    L_eff from derived allele counts out of n haploid samples.

    Monomorphic loci (count 0 or n) carry v = 0 and are excluded, matching the polymorphic
    filter applied to genotype matrices elsewhere.
    """
    p = np.asarray(counts, dtype=np.float64) / n
    v = p * (1.0 - p)
    v = v[v > 0]
    return float(v.sum() ** 2 / (v ** 2).sum()), int(v.size)


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, N_values=N_VALUES, L_values=L_VALUES,
        num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, verbose=True):
    """Measure L_eff for every (N, L) cell, replicated over independent subsamples."""
    counts_full, N_full, simID = load_counts(simdir)
    outdir = os.path.join(ANALYSIS_ROOT, name)

    if verbose:
        print(f"using {simID}  ({N_full} haploid samples, {counts_full.size:,} SNPs)")

    if max(N_values) > N_full:
        raise ValueError(f"simulation has {N_full} haploid samples, fewer than the "
                         f"largest requested N={max(N_values)}")

    rows = []
    bar = tqdm(total=len(N_values) * num_replicates, disable=not verbose,
               desc="L_eff sweep")
    for N in N_values:
        for rep in range(num_replicates):
            rng = np.random.default_rng([seed, int(N), rep])
            # Hypergeometric down-sample: exactly the allele count distribution of a
            # subsample of N out of N_full haploid samples.
            sub_counts = subsample_allele_counts(counts_full, N_full, int(N), rng)
            poly = np.flatnonzero((sub_counts > 0) & (sub_counts < int(N)))

            if poly.size < max(L_values):
                raise ValueError(
                    f"N={N} leaves only {poly.size:,} polymorphic loci, fewer than the "
                    f"largest requested L={max(L_values):,}. Simulate more SNPs "
                    f"(retention falls as N falls, so the smallest N is the binding case).")

            # One shuffle per replicate; each L takes a prefix, so the column sets are
            # nested and increasing L strictly adds loci rather than resampling.
            order = rng.permutation(poly)
            for L in L_values:
                L_eff, L_raw = effective_L_from_counts(sub_counts[order[:int(L)]], int(N))
                rows.append(dict(N=int(N), L_requested=int(L), rep=rep,
                                 L_raw=L_raw, L_eff=L_eff,
                                 L_polymorphic_available=int(poly.size)))
            bar.update(1)
    bar.close()

    df = pd.DataFrame(rows)
    df["L_eff_over_L_raw"] = df.L_eff / df.L_raw
    df["L_eff_over_L_requested"] = df.L_eff / df.L_requested

    summary = df.groupby(["N", "L_requested"]).agg(
        L_raw=("L_raw", "median"), L_eff=("L_eff", "median"),
        L_eff_sd=("L_eff", "std"),
        L_eff_over_L_raw=("L_eff_over_L_raw", "median"),
        L_eff_over_L_requested=("L_eff_over_L_requested", "median"),
        L_polymorphic_available=("L_polymorphic_available", "median"),
        num_replicates=("rep", "count")).reset_index()

    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "L_eff.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    if verbose:
        print(f"\nL_eff / L: {df.L_eff_over_L_raw.min():.3f} .. "
              f"{df.L_eff_over_L_raw.max():.3f}")
        piv = summary.pivot(index="N", columns="L_requested", values="L_eff")
        print("\nmedian L_eff by N (rows) and L (columns):")
        print(piv.round(0).to_string())
        print(f"\n-> {outdir}")
    return df, summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    run(simdir=args.simdir, name=args.name, num_replicates=args.num_replicates,
        seed=args.seed)


if __name__ == "__main__":
    main()

# %%
