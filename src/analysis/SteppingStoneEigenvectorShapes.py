#%%
"""
PC coordinates at three regimes -- no structure, weak, strong -- 1D and 2D, even sampling.

For each dimension, pick the simulated m closest to three targets in M/M_c (well below,
near, well above 1), at the largest available n_per_deme, and record the sample PC1 (1D)
or PC1+PC2 (2D) coordinates for every sample, along with each sample's deme index (for
coloring by position along the chain / grid).

2D's PC1/PC2 are procrustes-aligned to the closed-form (1,0)/(0,1) pair before saving, so
the plotted axes are oriented consistently with theory rather than by an arbitrary
rotation inside the degenerate eigenspace -- the eigenspace itself is what's recoverable,
but a fixed choice of basis inside it is needed to draw two separate scatter axes.

Outputs (results/analysis/SteppingStoneEigenvectorShapes):
    pcs.csv    one row per (dimension, regime, sample): deme index, PC1 (, PC2)
    meta.csv   one row per (dimension, regime): m, M, M_over_Mc, simID
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import load_simulations, subsample, sample_eigenvectors, SIMULATION_ROOT
from sweeps import ANALYSIS_ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import (closed_form_eigenvector_1d, closed_form_eigenvector_2d,
                                 critical_M, num_demes)

DEFAULT_NAME = "SteppingStoneEigenvectorShapes"
N_PER_DEME = 64                 # largest available -- cleanest picture of each regime
REGIME_TARGETS = {"no_structure": 5.0, "weak": 1.0, "strong": 0.2}   # M / M_c targets
SEED = 0


def pick_regime_rows(params, M_c):
    """The simulated row whose M/M_c is closest to each regime's target."""
    ratio = params.M.values / M_c
    picked = {}
    for regime, target in REGIME_TARGETS.items():
        i = int(np.argmin(np.abs(ratio - target)))
        picked[regime] = i
    return picked


def pcs_1d(Z, K, n_per_deme, L, rng):
    n = np.full(K, n_per_deme, dtype=int)
    sub = subsample(Z, n, L, rng)
    pc1 = sample_eigenvectors(sub, [0])[:, 0]
    u_theory = closed_form_eigenvector_1d(1, K, n_per_deme)
    pc1 = pc1 if pc1 @ u_theory >= 0 else -pc1
    return pd.DataFrame(dict(deme=np.repeat(np.arange(K), n_per_deme), pc1=pc1))


def pcs_2d(Z, K, n_per_deme, L, rng):
    K_demes = num_demes(K, 2)
    n = np.full(K_demes, n_per_deme, dtype=int)
    sub = subsample(Z, n, L, rng)
    V_s = sample_eigenvectors(sub, [0, 1])

    V_theory = np.column_stack([closed_form_eigenvector_2d(1, 0, K, n_per_deme),
                                closed_form_eigenvector_2d(0, 1, K, n_per_deme)])
    R = orthogonal_procrustes(V_s, V_theory)[0]
    V_aligned = V_s @ R

    deme = np.repeat(np.arange(K_demes), n_per_deme)
    return pd.DataFrame(dict(deme=deme, row=deme // K, col=deme % K,
                             pc1=V_aligned[:, 0], pc2=V_aligned[:, 1]))


def run(name=DEFAULT_NAME, n_per_deme=N_PER_DEME, L=None, seed=SEED, verbose=True):
    """Pick and record the three-regime PCs for both dimensions."""
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    pc_rows, meta_rows = [], []
    for dimension, pc_func in [(1, pcs_1d), (2, pcs_2d)]:
        simroot = os.path.join(SIMULATION_ROOT, f"stepping_stone_{dimension}d")
        cells = sorted(d for d in os.listdir(simroot) if d.startswith("K"))
        simdir = os.path.join(simroot, cells[0])
        params, arrays = load_simulations(simdir)
        K = int(params["K"].iloc[0])

        # M_c from a probe at this n_per_deme and the simulation's own L (or override),
        # so the regime targets are evaluated at the actual detectability threshold.
        L_probe = L if L is not None else int(params.L.iloc[0])
        M_c = critical_M(K, n_per_deme, L_probe, dimension)
        picked = pick_regime_rows(params, M_c)

        for regime, i in picked.items():
            row = params.iloc[i]
            L_cell = L if L is not None else int(row.L)
            rng = np.random.default_rng([seed, dimension, i])
            df = pc_func(arrays[i], K, n_per_deme, L_cell, rng)
            df["dimension"], df["regime"] = dimension, regime
            pc_rows.append(df)
            meta_rows.append(dict(dimension=dimension, regime=regime, simID=row.simID,
                                  m=float(row.m), M=float(row.M), M_c=M_c,
                                  M_over_Mc=float(row.M) / M_c, K=K,
                                  n_per_deme=n_per_deme, L=L_cell))
            if verbose:
                print(f"{dimension}D {regime}: simID={row.simID}, "
                      f"M/M_c={float(row.M) / M_c:.3f}")

    pd.concat(pc_rows, ignore_index=True).to_csv(os.path.join(outdir, "pcs.csv"),
                                                 index=False)
    pd.DataFrame(meta_rows).to_csv(os.path.join(outdir, "meta.csv"), index=False)
    if verbose:
        print(f"-> {outdir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--n-per-deme", type=int, default=N_PER_DEME)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(name=args.name, n_per_deme=args.n_per_deme, L=args.L, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
