#%%
"""
A demonstration of random deme sampling and its effect on the recovered PCs, 1D and 2D.

Three sampling fractions phi in [0.2, 0.4, 0.6], n_per_deme = 8, at two representative M
levels per dimension: "near_Mc" (M/M_c ~ 1, structure visible but not saturated) and
"high" (M/M_c ~ 5, effectively no structure) -- the figure's low- and high-migration
columns.

For each (dimension, phi, m_level): which demes were sampled (for the schematic in
column 1 of the figure), and the PC1 (1D) / PC1+PC2 (2D) coordinates of the sampled
individuals (for the low-/high-migration PC columns). A single fixed random deme draw per
(phi, m_level), not a replicate sweep -- this figure demonstrates the sampling scheme's
effect, it does not measure detectability (that's SteppingStoneRandomSampling.py /
figure 10).

Outputs (results/analysis/SteppingStoneRandomDemo):
    sampled_demes.csv   one row per (dimension, phi, deme): whether it was sampled
    pcs.csv             one row per (dimension, phi, m_level, sample): deme, PC1 (, PC2)
    meta.csv            one row per (dimension, m_level): simID, m, M, M_c, K
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

DEFAULT_NAME = "SteppingStoneRandomDemo"
PHI_VALUES = [0.2, 0.4, 0.6]
N_PER_DEME = 8
SEED = 0
M_LEVEL_TARGETS = {"near_Mc": 1.0, "high": 5.0}   # M / M_c targets


def sample_demes(K_demes, phi, rng):
    n_sampled = max(int(round(phi * K_demes)), 1)
    return np.sort(rng.choice(K_demes, n_sampled, replace=False))


def pick_m_rows(params, M_c, targets=M_LEVEL_TARGETS):
    """The simulated row index closest to each M/M_c target, as {level: row_index}."""
    ratio = params.M.values / M_c
    return {level: int(np.argmin(np.abs(ratio - target))) for level, target in targets.items()}


def demo_1d(Z, K, n_per_deme, phi, L, rng):
    demes = sample_demes(K, phi, rng)
    n = np.zeros(K, dtype=int)
    n[demes] = n_per_deme
    sub = subsample(Z, n, L, rng)
    pc1 = sample_eigenvectors(sub, [0])[:, 0]

    row_mask = np.zeros(K, dtype=bool)
    row_mask[demes] = True
    u_theory = closed_form_eigenvector_1d(1, K, n_per_deme)
    u_sub = u_theory[np.repeat(row_mask, n_per_deme)]
    pc1 = pc1 if pc1 @ u_sub >= 0 else -pc1

    pc_df = pd.DataFrame(dict(deme=np.repeat(demes, n_per_deme), pc1=pc1))
    deme_df = pd.DataFrame(dict(deme=np.arange(K), sampled=row_mask))
    return deme_df, pc_df


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
    V_theory = np.column_stack([
        closed_form_eigenvector_2d(1, 0, K, n_per_deme)[sample_mask],
        closed_form_eigenvector_2d(0, 1, K, n_per_deme)[sample_mask]])
    V_theory = V_theory / np.linalg.norm(V_theory, axis=0, keepdims=True)
    R = orthogonal_procrustes(V_s, V_theory)[0]
    V_aligned = V_s @ R

    deme = np.repeat(demes, n_per_deme)
    pc_df = pd.DataFrame(dict(deme=deme, row=deme // K, col=deme % K,
                              pc1=V_aligned[:, 0], pc2=V_aligned[:, 1]))
    deme_df = pd.DataFrame(dict(deme=np.arange(K_demes), row=np.arange(K_demes) // K,
                                col=np.arange(K_demes) % K, sampled=row_mask))
    return deme_df, pc_df


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

        L_probe = L if L is not None else int(params.L.iloc[0])
        M_c = critical_M(K, n_per_deme, L_probe, dimension)
        rows_by_level = pick_m_rows(params, M_c)
        level_index = {level: k for k, level in enumerate(M_LEVEL_TARGETS)}

        for level, i in rows_by_level.items():
            row = params.iloc[i]
            L_cell = L if L is not None else int(row.L)
            meta_rows.append(dict(dimension=dimension, m_level=level, simID=row.simID,
                                  m=float(row.m), M=float(row.M), M_c=M_c,
                                  M_over_Mc=float(row.M) / M_c, K=K))
            if verbose:
                print(f"{dimension}D {level}: simID={row.simID}, "
                      f"M/M_c={float(row.M) / M_c:.3f}")

            for phi in phi_values:
                rng = np.random.default_rng([seed, dimension, int(round(phi * 100)),
                                             level_index[level]])
                deme_df, pc_df = demo_func(arrays[i], K, n_per_deme, phi, L_cell, rng)
                deme_df["dimension"], deme_df["phi"], deme_df["m_level"] = \
                    dimension, phi, level
                pc_df["dimension"], pc_df["phi"], pc_df["m_level"] = dimension, phi, level
                deme_rows.append(deme_df)
                pc_rows.append(pc_df)

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
