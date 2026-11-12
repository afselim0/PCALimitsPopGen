#%%
"""
PC coordinates at three regimes -- no structure, weak, strong -- 1D and 2D, even sampling.

For each dimension, pick three simulated m by the MEASURED overlap already recorded in
results/analysis/SteppingStoneEvenSampling (median_cos over replicates), rather than by
M/M_c: no structure = the M whose overlap is smallest, weak = the M whose overlap is
closest to 0.5, strong = the largest M at which the overlap first reaches 0.9. The stored
median_cos is noisy at the low-overlap tail, so it is smoothed along M with a short
rolling median before these picks. Record the sample PC1 (1D) or PC1+PC2 (2D) coordinates
for every sample, along with each sample's deme index (for coloring by position along the
chain / grid).

2D's PC1/PC2 are procrustes-aligned to the closed-form (1,0)/(0,1) pair before saving, so
the plotted axes are oriented consistently with theory rather than by an arbitrary
rotation inside the degenerate eigenspace -- the eigenspace itself is what's recoverable,
but a fixed choice of basis inside it is needed to draw two separate scatter axes.

Outputs (results/analysis/SteppingStoneEigenvectorShapes):
    pcs.csv    one row per (dimension, regime, sample): deme index, PC1 (, PC2)
    meta.csv   one row per (dimension, regime): m, M, M_over_Mc, median_cos, simID
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import load_simulations, subsample, sample_eigenvectors, SIMULATION_ROOT
import preprocess
from sweeps import ANALYSIS_ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import (closed_form_eigenvector_1d, closed_form_eigenvector_2d,
                                 critical_M, num_demes)

DEFAULT_NAME = "SteppingStoneEigenvectorShapes"
N_PER_DEME = 4                  # haplotypes (two diploid individuals): the largest n whose swept
                                # M range still starts near no structure in 2D
OVERLAP_SWEEP = "SteppingStoneEvenSampling"   # supplies the measured median_cos per (n, M)
MID_TARGET = 0.7                # "weak": squared cosine closest to this
SMOOTH_WINDOW = 5               # rolling-median window over M, to de-noise median_cos
SEED = 0


def pick_regime_rows(params, dimension, n_per_deme, sweep_name=OVERLAP_SWEEP):
    """Index into `params` of the three regimes, chosen by measured overlap.

    Reads median_cos vs M from the even-sampling sweep at this (dimension, n_per_deme),
    smooths it along M, then takes the minimum-overlap M (no structure), the M closest to
    MID_TARGET (weak), and the lowest simulated M (strong) -- the most strongly structured
    cell in the sweep, where the overlap has saturated near 1.
    Returns {regime: (index into params, measured overlap)}.
    """
    sweep = pd.read_csv(os.path.join(ANALYSIS_ROOT, sweep_name, f"even_{dimension}D.csv"))
    sweep = sweep[sweep.n_per_deme == n_per_deme].sort_values("M")
    if sweep.empty:
        raise ValueError(f"no {dimension}D overlap sweep at n_per_deme={n_per_deme}")
    cos = sweep.median_cos.rolling(SMOOTH_WINDOW, center=True, min_periods=1).median().values

    chosen = {"no_structure": int(np.argmin(cos)),
              "weak": int(np.argmin(np.abs(cos ** 2 - MID_TARGET))),
              "strong": 0}   # sorted ascending by M, so row 0 is the lowest simulated M

    # The simulation directory holds two sweeps under different naming conventions
    # (an older invM* set, which owns the parameters CSV, and the m* set this overlap
    # sweep actually used). Only the m* stems match, and their K / L / M survive only in
    # the sweep table, so the sweep row is returned alongside the array index.
    simID_to_index = {s: i for i, s in enumerate(params.simID)}
    picked = {}
    for regime, j in chosen.items():
        row = sweep.iloc[j]
        if row.simID not in simID_to_index:
            raise KeyError(f"{dimension}D: sweep simID {row.simID} has no simulation file")
        picked[regime] = (simID_to_index[row.simID], row)
    return picked


def pcs_1d(Z, K, n_per_deme, L, rng):
    n = np.full(K, n_per_deme, dtype=int)
    sub = subsample(Z, n, L, rng)
    pc1 = sample_eigenvectors(sub, [0])[:, 0]
    # cos(pi/K * (j+1/2)) DECREASES in j, so aligning to it as-is leaves PC1 running
    # backwards along the chain and the plotted colors mirrored relative to the deme
    # schematic. Negate the reference so the fixed axis increases with deme index.
    u_theory = -closed_form_eigenvector_1d(1, K, n_per_deme)
    pc1 = pc1 if pc1 @ u_theory >= 0 else -pc1
    return pd.DataFrame(dict(deme=np.repeat(np.arange(K), n_per_deme), pc1=pc1))


def pcs_2d(Z, K, n_per_deme, L, rng):
    K_demes = num_demes(K, 2)
    n = np.full(K_demes, n_per_deme, dtype=int)
    sub = subsample(Z, n, L, rng)
    V_s = sample_eigenvectors(sub, [0, 1])

    # Column first, then row: the schematic draws column along x and row along y, so
    # (0,1) -- which varies with the column index -- has to be the axis plotted as PC1,
    # otherwise the scatter comes out as the schematic's transpose. Both are negated
    # because cos(pi/K * (i+1/2)) decreases in i, which would otherwise mirror each axis.
    V_theory = np.column_stack([-closed_form_eigenvector_2d(0, 1, K, n_per_deme),
                                -closed_form_eigenvector_2d(1, 0, K, n_per_deme)])
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
        picked = pick_regime_rows(params, dimension, n_per_deme)

        # K and L come from the sweep rows, not from params: the m* simulations this
        # sweep used carry neither in their filename nor in the directory's CSV.
        any_row = next(iter(picked.values()))[1]
        K = int(any_row.K)

        # M_c is no longer what selects the regimes (the measured overlap is), but keep it
        # so M/M_c stays available alongside each pick.
        L_probe = L if L is not None else int(any_row.L)
        M_c = critical_M(K, n_per_deme, L_probe, dimension)

        for regime, (i, row) in picked.items():
            L_cell = L if L is not None else int(row.L)
            rng = np.random.default_rng([seed, dimension, i])
            df = pc_func(arrays[i], K, n_per_deme, L_cell, rng)
            df["dimension"], df["regime"] = dimension, regime
            pc_rows.append(df)
            meta_rows.append(dict(dimension=dimension, regime=regime, simID=row.simID,
                                  m=float(row.m), M=float(row.M), M_c=M_c,
                                  M_over_Mc=float(row.M) / M_c,
                                  median_cos=float(row.median_cos),
                                  K=K, n_per_deme=n_per_deme, L=L_cell))
            if verbose:
                print(f"{dimension}D {regime}: simID={row.simID}, M={float(row.M):.4g}, "
                      f"overlap={float(row.median_cos):.3f}")

    preprocess.individuals(pd.concat(pc_rows, ignore_index=True)).to_csv(os.path.join(outdir, "pcs.csv"),
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
