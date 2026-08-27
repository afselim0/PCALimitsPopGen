#%%
"""
Uniform deme dilution and its effect on the recovered PCs, 1D and 2D.

The counterpart of SteppingStoneRandomDemo: there the deme subset is a random draw of a
given size, here it is every p-th deme of the chain (1D) or every p-th row AND column of
the grid (2D). The total sample size N is held FIXED across p, so the samples freed by
dropping demes are redistributed over the demes that remain -- what varies along a row is
spatial coverage alone, not sample size.

Migration is held fixed at one M per dimension (M_TARGETS), chosen to sit where dense
coverage recovers the spatial axis; three strides per dimension.

PCs are sign-fixed (1D) and procrustes-aligned (2D) to the closed-form eigenvectors,
oriented so PC1 increases with chain position / column and PC2 with row, i.e. the scatter
colors run the same way as the schematic above them.

Outputs (results/analysis/SteppingStoneStrideDemo):
    sampled_demes.csv   one row per (dimension, p, deme): whether the stride kept it
    pcs.csv             one row per (dimension, p, sample): deme, PC1 (, PC2), and `cos`,
                        the cosine similarity of that panel's recovered PC(s) with the
                        closed-form axis / axes restricted to the sampled demes
                        (constant within a panel)
    meta.csv            one row per (dimension, p): simID, m, M, K, n_per_deme, N
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
from SteppingStoneRandomDemo import pick_m_row

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import (closed_form_eigenvector_1d, closed_form_eigenvector_2d,
                                 num_demes, strided_demes)

DEFAULT_NAME = "SteppingStoneStrideDemo"
# Three strides per dimension, from full coverage to the sparsest the lattice allows.
# 2D skips p=3, whose retained rows {0,3,6} are asymmetric about the lattice centre and
# break the (1,0)/(0,1) degeneracy; the demo is about coverage, and p=4 keeps the same
# number of demes without that confound.
P_VALUES = {1: [1, 3, 5], 2: [1, 2, 4]}
# Total sample size held fixed across strides, matching configs 11 and 12.
N_TOTAL = {1: 100, 2: 324}
SEED = 0
M_TARGETS = {1: 500.0, 2: 50.0}


def stride_sampling(K, p, dimension, N_total):
    """(deme indices kept, n-per-deme) for a stride, spreading N_total over them."""
    demes = strided_demes(K, p, dimension)
    return demes, max(int(N_total // len(demes)), 1)


def demo_1d(Z, K, p, N_total, L, rng):
    demes, n_per_deme = stride_sampling(K, p, 1, N_total)
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
    return deme_df, pc_df, cos, n_per_deme


def demo_2d(Z, K, p, N_total, L, rng):
    K_demes = num_demes(K, 2)
    demes, n_per_deme = stride_sampling(K, p, 2, N_total)
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
    return deme_df, pc_df, cos, n_per_deme


def run(name=DEFAULT_NAME, p_values=None, n_total=None, L=None, seed=SEED, verbose=True):
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    p_values = p_values or P_VALUES
    n_total = n_total or N_TOTAL

    deme_rows, pc_rows, meta_rows = [], [], []
    for dimension, demo_func in [(1, demo_1d), (2, demo_2d)]:
        simroot = os.path.join(SIMULATION_ROOT, f"stepping_stone_{dimension}d")
        cells = sorted(d for d in os.listdir(simroot) if d.startswith("K"))
        simdir = os.path.join(simroot, cells[0])
        params, arrays = load_simulations(simdir)
        K = int(params["K"].iloc[0])

        i, row = pick_m_row(params, dimension)
        L_cell = L if L is not None else int(row.L)
        if verbose:
            print(f"{dimension}D: simID={row.simID}, M={float(row.M):.0f}, K={K}")

        for p in p_values[dimension]:
            rng = np.random.default_rng([seed, dimension, p])
            deme_df, pc_df, cos, n_per = demo_func(
                arrays[i], K, p, n_total[dimension], L_cell, rng)
            deme_df["dimension"], deme_df["p"] = dimension, p
            pc_df["dimension"], pc_df["p"] = dimension, p
            pc_df["cos"] = cos
            deme_rows.append(deme_df)
            pc_rows.append(pc_df)
            meta_rows.append(dict(dimension=dimension, p=p, simID=row.simID,
                                  m=float(row.m), M=float(row.M), K=K,
                                  n_demes_sampled=int(deme_df.sampled.sum()),
                                  n_per_deme=n_per, N=len(pc_df), cos=cos))
            if verbose:
                print(f"  p={p}: {int(deme_df.sampled.sum())} demes x {n_per} "
                      f"= {len(pc_df)} samples, cos={cos:.3f}")

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
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(name=args.name, L=args.L, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
