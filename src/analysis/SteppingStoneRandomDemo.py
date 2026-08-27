#%%
"""
A demonstration of random deme sampling and its effect on the recovered PCs, 1D and 2D.

A demonstration that, at a FIXED migration rate, how densely the demes are sampled decides
whether PCA recovers the spatial structure at all.

One M per dimension (M_TARGETS), chosen to sit in the regime where sparse sampling fails
and dense sampling succeeds, and three sampling fractions phi in [0.2, 0.4, 0.6] at
n_per_deme = 8. Migration is held fixed across the whole figure so the only thing varying
along a row is the sampling scheme.

PCs are sign-fixed (1D) and procrustes-aligned (2D) to the closed-form eigenvectors
oriented so that PC1 increases with chain position / column and PC2 increases with row,
i.e. the scatter colors run the same way as the schematic above them.

For each (dimension, phi): which demes were sampled (for the schematic rows), and the PC1
(1D) / PC1+PC2 (2D) coordinates of the sampled individuals (for the PC rows). A single
fixed random deme draw per (dimension, phi), not a replicate sweep -- this figure
demonstrates the sampling scheme's effect, it does not measure detectability (that's
SteppingStoneRandomSampling.py / figure 10).

Outputs (results/analysis/SteppingStoneRandomDemo):
    sampled_demes.csv   one row per (dimension, phi, deme): whether it was sampled
    pcs.csv             one row per (dimension, phi, sample): deme, PC1 (, PC2), and
                        `cos`, the cosine similarity of that panel's recovered PC(s)
                        with the closed-form axis / axes (constant within a panel)
    meta.csv            one row per dimension: simID, m, M, M_c, K
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import (load_simulations, subsample, sample_eigenvectors,
                         principal_cosines, SIMULATION_ROOT)
from sweeps import ANALYSIS_ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import (closed_form_eigenvector_1d, closed_form_eigenvector_2d,
                                 critical_M, num_demes)

DEFAULT_NAME = "SteppingStoneRandomDemo"
SWEEP_NAME = "SteppingStoneRandomSampling"   # supplies the M grid this demo picks from
PHI_VALUES = [0.2, 0.4, 0.6]
N_PER_DEME = 8
SEED = 0
# One target M per dimension, in the regime where sparse sampling misses the structure
# that dense sampling recovers.
M_TARGETS = {1: 500.0, 2: 50.0}


def sample_demes(K_demes, phi, rng):
    n_sampled = max(int(round(phi * K_demes)), 1)
    return np.sort(rng.choice(K_demes, n_sampled, replace=False))


def pick_m_row(params, dimension, sweep_name=SWEEP_NAME):
    """The simulated M closest to this dimension's M_TARGETS, as (array index, sweep row).

    The simulation directory holds two sweeps under different naming conventions (an older
    invM* set, which owns the parameters CSV, and the m* set the overlap sweeps actually
    used, whose M and L survive only in the sweep table). Picking from the sweep table
    keeps this demo on the same M grid as figure 10.
    """
    sweep = pd.read_csv(os.path.join(ANALYSIS_ROOT, sweep_name, f"random_{dimension}D.csv"))
    sweep = sweep.drop_duplicates("simID")
    row = sweep.iloc[int(np.argmin(np.abs(sweep.M.values - M_TARGETS[dimension])))]
    simID_to_index = {s: i for i, s in enumerate(params.simID)}
    if row.simID not in simID_to_index:
        raise KeyError(f"{dimension}D: sweep simID {row.simID} has no simulation file")
    return simID_to_index[row.simID], row


def demo_1d(Z, K, n_per_deme, phi, L, rng):
    demes = sample_demes(K, phi, rng)
    n = np.zeros(K, dtype=int)
    n[demes] = n_per_deme
    sub = subsample(Z, n, L, rng)
    pc1 = sample_eigenvectors(sub, [0])[:, 0]

    row_mask = np.zeros(K, dtype=bool)
    row_mask[demes] = True
    # cos(pi/K * (j+1/2)) DECREASES in j, so aligning to it as-is leaves PC1 running
    # backwards along the chain and the colors mirrored relative to the schematic.
    u_theory = -closed_form_eigenvector_1d(1, K, n_per_deme)
    u_sub = u_theory[np.repeat(row_mask, n_per_deme)]
    u_sub = u_sub / np.linalg.norm(u_sub)
    pc1 = pc1 if pc1 @ u_sub >= 0 else -pc1
    cos = float(principal_cosines(u_sub[:, None], pc1[:, None])[0])

    pc_df = pd.DataFrame(dict(deme=np.repeat(demes, n_per_deme), pc1=pc1))
    deme_df = pd.DataFrame(dict(deme=np.arange(K), sampled=row_mask))
    return deme_df, pc_df, cos


def demo_2d(Z, K, n_per_deme, phi, L, rng):
    K_demes = num_demes(K, 2)
    demes = sample_demes(K_demes, phi, rng)
    n = np.zeros(K_demes, dtype=int)
    n[demes] = n_per_deme
    sub = subsample(Z, n, L, rng)
    V_s = sample_eigenvectors(sub, [0, 1])

    row_mask = np.zeros(K_demes, dtype=bool)
    row_mask[demes] = True
    sample_mask = np.repeat(row_mask, n_per_deme)
    # Column first, then row: the schematic draws column along x and row along y, so
    # (0,1) -- which varies with the column index -- has to be the axis plotted as PC1,
    # otherwise the scatter comes out as the schematic's transpose. Both are negated
    # because cos(pi/K * (i+1/2)) decreases in i, which would otherwise mirror each axis.
    V_theory = np.column_stack([
        -closed_form_eigenvector_2d(0, 1, K, n_per_deme)[sample_mask],
        -closed_form_eigenvector_2d(1, 0, K, n_per_deme)[sample_mask]])
    # Restricting to a deme subset destroys the orthonormality the closed forms have on
    # the full lattice, so re-orthonormalize before comparing subspaces.
    V_theory = np.linalg.qr(V_theory)[0]
    R = orthogonal_procrustes(V_s, V_theory)[0]
    V_aligned = V_s @ R
    cos = float(principal_cosines(V_theory, V_s).mean())

    deme = np.repeat(demes, n_per_deme)
    pc_df = pd.DataFrame(dict(deme=deme, row=deme // K, col=deme % K,
                              pc1=V_aligned[:, 0], pc2=V_aligned[:, 1]))
    deme_df = pd.DataFrame(dict(deme=np.arange(K_demes), row=np.arange(K_demes) // K,
                                col=np.arange(K_demes) % K, sampled=row_mask))
    return deme_df, pc_df, cos


def run(name=DEFAULT_NAME, n_per_deme=N_PER_DEME, phi_values=PHI_VALUES, L=None,
        seed=SEED, verbose=True):
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    deme_rows, pc_rows, meta_rows = [], [], []
    for dimension, demo_func in [(1, demo_1d), (2, demo_2d)]:
        simroot = os.path.join(SIMULATION_ROOT, f"stepping_stone_{dimension}d")
        cells = sorted(d for d in os.listdir(simroot) if d.startswith("K"))
        simdir = os.path.join(simroot, cells[0])
        params, arrays = load_simulations(simdir)
        K = int(params["K"].iloc[0])

        i, row = pick_m_row(params, dimension)
        L_cell = L if L is not None else int(row.L)
        M_c = critical_M(K, n_per_deme, L_cell, dimension)
        meta_rows.append(dict(dimension=dimension, simID=row.simID, m=float(row.m),
                              M=float(row.M), M_c=M_c, M_over_Mc=float(row.M) / M_c, K=K))
        if verbose:
            print(f"{dimension}D: simID={row.simID}, M={float(row.M):.0f}, "
                  f"M/M_c={float(row.M) / M_c:.3f}")

        for phi in phi_values:
            rng = np.random.default_rng([seed, dimension, int(round(phi * 100))])
            deme_df, pc_df, cos = demo_func(arrays[i], K, n_per_deme, phi, L_cell, rng)
            deme_df["dimension"], deme_df["phi"] = dimension, phi
            pc_df["dimension"], pc_df["phi"] = dimension, phi
            pc_df["cos"] = cos
            deme_rows.append(deme_df)
            pc_rows.append(pc_df)
            if verbose:
                print(f"  phi={phi:g}: cos={cos:.3f}")

    pd.concat(deme_rows, ignore_index=True).to_csv(
        os.path.join(outdir, "sampled_demes.csv"), index=False)
    pd.concat(pc_rows, ignore_index=True).to_csv(
        os.path.join(outdir, "pcs.csv"), index=False)
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
