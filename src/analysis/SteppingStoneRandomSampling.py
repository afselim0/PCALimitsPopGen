#%%
"""
Random subsets of demes, 1D and 2D stepping-stone: PC recovery of the true spatial axis
against a partial, randomly-chosen sample of demes.

phi is the fraction of demes sampled (chosen once per replicate, without replacement),
n_per_deme the number of samples drawn from each chosen deme. Two sweeps:

    vary_phi   n_per_deme = 8 fixed, phi in [0.2, 0.4, 0.6, 0.8, 1.0]
    vary_n     phi = 0.4 fixed, n_per_deme in [2, 4, 6, 7, 16, 32, 64]

The "true spatial axis" for a partial sample is the full-K closed-form eigenvector
(stepping_stone_grid.closed_form_eigenvector_1d/_2d), restricted to the rows of the
sampled demes and renormalized -- not recomputed from a resistance-distance matrix on the
reduced deme set. This is what "the true spatial axis" means for a subset here: the
direction the full population's structure predicts, evaluated where it was actually
observed, rather than a different axis that would maximize the structure eigenvalue of
the reduced sample considered on its own.

The theoretical nu used for the BBP prediction is therefore also the full-K closed-form
value at the reduced N = phi * K_demes * n_per_deme, matching the axis it is paired with.

L is fixed at the simulation's own L (or an override) across every cell within one sweep;
only the sampling design (phi, n_per_deme) varies.

subsample() (pc_recovery) accepts a zero count for an unsampled deme -- rng.choice(...,
0) returns an empty draw -- so a random deme subset is just a per-subpopulation size
vector n with zeros at the unsampled demes, no special-casing needed there.

Outputs (results/analysis/SteppingStoneRandomSampling):
    random_1D.csv, random_2D.csv   one row per (dimension, sweep, level, m) cell
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import (load_simulations, subsample, sample_eigenvectors,
                         principal_cosines, procrustes_distance, effective_L,
                         analytical_cosine, SIMULATION_ROOT)
from sweeps import ANALYSIS_ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import (closed_form_eigenvector_1d, closed_form_eigenvector_2d,
                                 closed_form_spectrum, num_demes)

DEFAULT_NAME = "SteppingStoneRandomSampling"
DEFAULT_NUM_REPLICATES = 20
DEFAULT_SEED = 0

PHI_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
N_FIXED_FOR_PHI = 8
N_VALUES = [2, 4, 6, 7, 16, 32, 64]
PHI_FIXED_FOR_N = 0.4


def sample_demes(K_demes, phi, rng):
    """A random subset of deme indices, without replacement."""
    n_sampled = max(int(round(phi * K_demes)), 1)
    return np.sort(rng.choice(K_demes, n_sampled, replace=False))


def true_axis_restricted(K, dimension, n_per_deme, sampled_demes, K_demes):
    """
    The full-K leading closed-form eigenvector(s), restricted to the sampled demes'
    sample-space rows and renormalized. 1D returns one column; 2D returns the degenerate
    (1,0)/(0,1) pair, each restricted and renormalized independently.
    """
    if dimension == 1:
        cols = [closed_form_eigenvector_1d(1, K, n_per_deme)]
    else:
        cols = [closed_form_eigenvector_2d(1, 0, K, n_per_deme),
                closed_form_eigenvector_2d(0, 1, K, n_per_deme)]

    row_mask = np.zeros(K_demes, dtype=bool)
    row_mask[sampled_demes] = True
    sample_mask = np.repeat(row_mask, n_per_deme)

    out = [u[sample_mask] / np.linalg.norm(u[sample_mask]) for u in cols]
    return np.column_stack(out)


def measure_cell(Z, K, dimension, n_per_deme, phi, L, num_replicates, seed, M):
    """
    Median PC recovery over replicates, each drawing its own random deme subset and
    individuals within them -- both sources of randomness contribute to the spread.
    """
    k_lead = 1 if dimension == 1 else 2
    K_demes = num_demes(K, dimension)
    _, nu_full, _ = closed_form_spectrum(K, M, n_per_deme, dimension)
    nu = nu_full[:k_lead]

    cos_all, dists, L_effs, L_raws = [], [], [], []
    for rep in range(num_replicates):
        rng = np.random.default_rng([seed, rep])
        demes = sample_demes(K_demes, phi, rng)
        n = np.zeros(K_demes, dtype=int)
        n[demes] = n_per_deme

        sub = subsample(Z, n, L, rng)
        V_theory = true_axis_restricted(K, dimension, n_per_deme, demes, K_demes)
        by_magnitude = np.any(nu < 0)
        V_s = sample_eigenvectors(sub, list(range(k_lead)), by_magnitude=by_magnitude)

        cos_all.append(principal_cosines(V_theory, V_s))
        dists.append(procrustes_distance(V_theory, V_s))
        L_effs.append(effective_L(sub))
        L_raws.append(sub.shape[1])

    cos_all = np.array(cos_all)
    mean_cos = cos_all.mean(axis=1)
    L_eff_med = float(np.median(L_effs))
    N_eff = int(n.sum())            # every replicate samples the same total N
    gamma = N_eff / L_eff_med
    _, analytical_mean = analytical_cosine(nu, gamma, N=N_eff)

    return dict(median_cos=float(np.median(mean_cos)),
               analytical_cos=analytical_mean,
               procrustes_d=float(np.median(dists)),
               L_eff=L_eff_med, L_raw=float(np.median(L_raws)), gamma=gamma,
               N=N_eff, n_demes_sampled=len(demes), nu_1=float(nu[0]))


def analyse_sweep(simdir, dimension, sweep_name, levels, level_key,
                  fixed, L, num_replicates, seed, verbose=True):
    """One sweep (vary_phi or vary_n): every simulated m, at every level of the swept
    sampling parameter, the other parameter held at `fixed`."""
    params, arrays = load_simulations(simdir)
    K = int(params["K"].iloc[0])

    rows = []
    cells = [(r, lvl) for r in params.itertuples() for lvl in levels]
    bar = tqdm(cells, disable=not verbose, desc=f"{sweep_name} ({dimension}D)")
    for r, lvl in bar:
        design = dict(fixed)
        design[level_key] = lvl
        L_cell = L if L is not None else int(r.L)

        res = measure_cell(arrays[r.Index], K, dimension, design["n_per_deme"],
                           design["phi"], L_cell, num_replicates, seed, r.M)
        rows.append(dict(
            simID=r.simID, dimension=dimension, K=K, sweep=sweep_name,
            m=r.m, Ne=r.Ne, M=r.M, L=L_cell,
            phi=design["phi"], n_per_deme=design["n_per_deme"], **res,
            num_replicates=num_replicates))
    return pd.DataFrame(rows)


def run(dimension, simulation_root=None, L=None, num_replicates=DEFAULT_NUM_REPLICATES,
        seed=DEFAULT_SEED, name=DEFAULT_NAME, verbose=True):
    """Both sweeps for one dimension's single K simulation set."""
    simulation_root = simulation_root or os.path.join(
        SIMULATION_ROOT, f"stepping_stone_{dimension}d")
    cells = sorted(d for d in os.listdir(simulation_root)
                   if d.startswith("K") and
                   os.path.isdir(os.path.join(simulation_root, d)))
    if not cells:
        raise FileNotFoundError(f"no K* subdirectories in {simulation_root}")
    simdir = os.path.join(simulation_root, cells[0])
    if len(cells) > 1 and verbose:
        print(f"warning: {len(cells)} K cells found in {simulation_root}, using {cells[0]}")

    df_phi = analyse_sweep(simdir, dimension, "vary_phi", PHI_VALUES, "phi",
                           dict(n_per_deme=N_FIXED_FOR_PHI), L, num_replicates, seed,
                           verbose=verbose)
    df_n = analyse_sweep(simdir, dimension, "vary_n", N_VALUES, "n_per_deme",
                        dict(phi=PHI_FIXED_FOR_N), L, num_replicates, seed,
                        verbose=verbose)
    df = pd.concat([df_phi, df_n], ignore_index=True)

    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"random_{dimension}D.csv")
    df.to_csv(path, index=False)
    if verbose:
        print(f"wrote {len(df)} rows -> {path}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dimension", type=int, choices=(1, 2), required=True)
    p.add_argument("--simulation-root", default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args()
    run(args.dimension, simulation_root=args.simulation_root, L=args.L,
        num_replicates=args.num_replicates, seed=args.seed, name=args.name)


if __name__ == "__main__":
    main()

# %%
