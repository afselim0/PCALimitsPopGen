#%%
"""
Simulate haplotypes under a two-population split demography (pop A, pop B from ancestral
ANC at time t_split), with Fst = t_split / (2*Ne).

Outputs (results/simulations/single_split by default):
    Fst<Fst>.npy   the (2 * n_per_pop) x L genotype matrix, saved as np.bool_, with Fst
                    in scientific notation to 4 decimal places (e.g. Fst4.0255e-03.npy).
                    Sample order: pop A then pop B.
"""
import argparse
import os
import sys
from multiprocessing import cpu_count

import msprime
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from utils import simulate_haplotypes

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTDIR = os.path.join(_PROJECT_ROOT, "results", "simulations", "single_split")

DEFAULT_NE = 1e4
DEFAULT_N_PER_POP = 500
DEFAULT_L = int(5e4)
DEFAULT_NUM_FST = 100
DEFAULT_FST_MIN = 5e-4
DEFAULT_FST_MAX = 5e-2

DEFAULT_RANDOM_SEED = 42
DEFAULT_N_WORKERS = max(cpu_count() - 3, 1)


def format_simID(Fst):
    """Filename stem for a simulation: Fst in scientific notation to 4 decimals."""
    return f"Fst{Fst:.4e}"


def split_demography(Fst, Ne=DEFAULT_NE):
    """Two-population split demography with t_split = 2*Ne*Fst."""
    t_split = 2 * Ne * Fst
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=Ne)
    demography.add_population(name="B", initial_size=Ne)
    demography.add_population(name="ANC", initial_size=Ne)
    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="ANC")
    demography.sort_events()
    return demography


def simulate_one(Fst, n_per_pop, L, outdir, random_seed, Ne=DEFAULT_NE, n_workers=1):
    """Simulate one Fst value and save its genotype matrix as Fst<Fst>.npy."""
    demography = split_demography(Fst, Ne=Ne)
    X = simulate_haplotypes(
        demography, n=n_per_pop / 2, L=L, random_seed=random_seed, n_workers=n_workers)
    X = X.astype(np.bool_)
    np.save(os.path.join(outdir, f"{format_simID(Fst)}.npy"), X)


def run(num_fst=DEFAULT_NUM_FST, fst_min=DEFAULT_FST_MIN, fst_max=DEFAULT_FST_MAX,
        n_per_pop=DEFAULT_N_PER_POP, L=DEFAULT_L, Ne=DEFAULT_NE, outdir=DEFAULT_OUTDIR,
        random_seed=DEFAULT_RANDOM_SEED, n_workers=DEFAULT_N_WORKERS, verbose=True):
    """Run the full Fst sweep, log-spaced from fst_min to fst_max."""
    fst_values = np.geomspace(fst_min, fst_max, num_fst)

    if verbose:
        print(f"{num_fst} simulations, Fst in [{fst_min:.4e}, {fst_max:.4e}], "
              f"n_per_pop={n_per_pop}, N={2 * n_per_pop}, L={L}, Ne={Ne:g}, "
              f"n_workers={n_workers}")
        print(f"output -> {outdir}")

    os.makedirs(outdir, exist_ok=True)

    bar = tqdm(fst_values, disable=not verbose)
    for i, Fst in enumerate(bar, start=1):
        simID = format_simID(Fst)
        if os.path.exists(os.path.join(outdir, f"{simID}.npy")):
            bar.set_description(f"skipping {i}/{num_fst} (exists), Fst={Fst:.4g}")
            continue
        bar.set_description(f"running {i}/{num_fst}, Fst={Fst:.4g}")
        simulate_one(Fst, n_per_pop, L, outdir, random_seed + i, Ne=Ne, n_workers=n_workers)

    if verbose:
        print(f"wrote {num_fst} simulations to {outdir}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per-pop", type=int, default=DEFAULT_N_PER_POP, help=f"haploid samples per population (default: {DEFAULT_N_PER_POP})")
    p.add_argument("--L", type=int, default=DEFAULT_L, help=f"number of independent SNPs (default: {DEFAULT_L:g})")
    p.add_argument("--Ne", type=float, default=DEFAULT_NE, help=f"effective population size of A, B, ANC (default: {DEFAULT_NE:g})")
    p.add_argument("--num-fst", type=int, default=DEFAULT_NUM_FST, help=f"number of log-spaced Fst values (default: {DEFAULT_NUM_FST})")
    p.add_argument("--fst-min", type=float, default=DEFAULT_FST_MIN, help=f"lower end of the Fst sweep (default: {DEFAULT_FST_MIN:g})")
    p.add_argument("--fst-max", type=float, default=DEFAULT_FST_MAX, help=f"upper end of the Fst sweep (default: {DEFAULT_FST_MAX:g})")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS, help=f"processes to split each simulation's L replicates across (default: {DEFAULT_N_WORKERS}, i.e. cpu_count - 1)")
    args = p.parse_args()

    run(num_fst=args.num_fst, fst_min=args.fst_min, fst_max=args.fst_max,
        n_per_pop=args.n_per_pop, L=args.L, Ne=args.Ne, outdir=args.outdir,
        random_seed=args.random_seed, n_workers=args.n_workers)


if __name__ == "__main__":
    main()

# %%
