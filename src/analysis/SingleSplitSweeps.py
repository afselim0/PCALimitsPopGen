#%%
"""
PC1 recovery across the two-population split sweep, as a function of N and of phi.

Two sweeps over results/simulations/split, both at L_input = 2e4 SNP columns per
replicate, both recording the PC1 overlap against the theoretical structure axis for every
one of the 100 Fst values:

    vary_N.csv     10 values of N from 100 to 1000, at phi = 0.5
    vary_phi.csv   10 values of phi from 0.05 to 0.5, at N = 200

Each row carries the measured cosine similarity (median over replicates, with a MAD-based
standard error) beside analytical_cos, the BBP prediction

    |cos| = sqrt((1 - gamma/nu^2) / (1 + gamma/nu)),  gamma = N/L_eff,  nu = 2*N*phi*(1-phi)*Fst

built from the measured L_eff and the theoretical nu, so nothing in it is fitted and the two
columns can be compared directly.

Fst is always the theoretical value from the filename, never re-estimated.

Outputs (results/analysis/SingleSplitSweeps): vary_N.csv, vary_phi.csv
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_recovery import SIMULATION_ROOT
from sweeps import ANALYSIS_ROOT, run_sweep, split_T

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "split")
DEFAULT_NAME = "SingleSplitSweeps"

DEFAULT_L = int(2e4)
DEFAULT_NUM_REPLICATES = 10
DEFAULT_SEED = 0

# 10 values each, per the sweep design.
N_VALUES = np.linspace(100, 1000, 10).astype(int)
PHI_VALUES = np.linspace(0.05, 0.5, 10)
PHI_FIXED_N = 200


def n_from_phi(N, phi):
    """
    Per-population sample sizes for a total N split at fraction phi.

    The smaller population is rounded to at least 1 and the remainder goes to the larger,
    so sum(n) == N exactly and no cell is degenerate at the small-phi end.
    """
    nA = max(int(round(phi * N)), 1)
    return np.array([nA, N - nA])


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, L=DEFAULT_L,
        num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, verbose=True):
    """Run both sweeps and write one CSV each."""
    outdir = os.path.join(ANALYSIS_ROOT, name)
    T_from = lambda row: split_T(row["Fst"])

    # ---- Sweep 1: vary N at phi = 0.5.
    schemes_N = [n_from_phi(int(N), 0.5) for N in N_VALUES]
    if verbose:
        print(f"N sweep: N in {list(N_VALUES)}, phi = 0.5, L = {L:g}")
    df_N = run_sweep(simdir, T_from, schemes_N, indices=[0], name=name, outdir=outdir,
                     L=L, num_replicates=num_replicates, seed=seed, verbose=verbose,
                     filename="vary_N.csv", desc="N sweep")

    # ---- Sweep 2: vary phi at N = 200.
    schemes_phi = [n_from_phi(PHI_FIXED_N, p) for p in PHI_VALUES]
    if verbose:
        print(f"\nphi sweep: phi in {np.round(PHI_VALUES, 3).tolist()}, "
              f"N = {PHI_FIXED_N}, L = {L:g}")
    df_phi = run_sweep(simdir, T_from, schemes_phi, indices=[0], name=name, outdir=outdir,
                       L=L, num_replicates=num_replicates, seed=seed, verbose=verbose,
                       filename="vary_phi.csv", desc="phi sweep")

    if verbose:
        print(f"\nvary_N.csv   {len(df_N)} rows ({len(N_VALUES)} N x {df_N.simID.nunique()} Fst)")
        print(f"vary_phi.csv {len(df_phi)} rows ({len(PHI_VALUES)} phi x "
              f"{df_phi.simID.nunique()} Fst)")
        print(f"-> {outdir}")
    return df_N, df_phi


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME,
                   help=f"subdirectory of results/analysis (default: {DEFAULT_NAME})")
    p.add_argument("--L", type=int, default=DEFAULT_L,
                   help=f"SNP columns per replicate (default: {DEFAULT_L:g})")
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES,
                   help=f"subsampling replicates per cell (default: {DEFAULT_NUM_REPLICATES})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    run(simdir=args.simdir, name=args.name, L=args.L,
        num_replicates=args.num_replicates, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
