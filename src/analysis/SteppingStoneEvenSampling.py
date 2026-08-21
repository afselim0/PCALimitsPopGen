#%%
"""
Even sampling of every deme, 1D and 2D stepping-stone: eigenvalue verification and PC
recovery vs M/M_c, over a sweep of per-deme sample sizes n.

Every deme is sampled at n_per_deme in [2, 4, 8, 16, 32, 64], L held fixed across n (the
simulation's full L, since every simulated m already shares one L). For each (m, n) cell:

    nu_meas       the sample eigenvalue(s) of the leading structure mode(s) -- one value
                  in 1D, the mean of the degenerate pair in 2D
    nu_theory     the closed-form eigenvalue (stepping_stone_grid.closed_form_spectrum),
                  verified elsewhere to match the resistance-distance F_hat exactly
    median_cos    PC recovery: |cos| in 1D, mean principal-angle cosine (procrustes-
                  aligned subspace) in 2D, against the closed-form leading eigenvector(s)
    M_over_Mc     M / critical_M(K, n, L_eff, dimension), the detectability axis

1D uses PC1 against the single leading eigenvector. 2D procrustes-aligns the top two PCs
against the degenerate (1,0)/(0,1) pair and reports the subspace cosine -- no canonical
basis exists inside a 2-fold degenerate eigenspace, so a per-axis comparison would be
ill-posed (see pc_recovery.theoretical_eigenvectors's degeneracy handling for the same
convention elsewhere in this project).

Outputs (results/analysis/SteppingStoneEvenSampling):
    even_1D.csv   one row per (m, n) cell
    even_2D.csv   one row per (m, n) cell
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import (load_simulations, pc_recovery, subsample, effective_L,
                         _median_se, SIMULATION_ROOT)
from sweeps import ANALYSIS_ROOT
from utils import compute_sample_covariance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import closed_form_spectrum, critical_M, num_demes

SPECTRUM_SCALE = 0.5   # compute_sample_covariance's floor sits at 2; the theory's floor
                        # (structure_eigenvalues' nu=0 convention) is at 1, so rescale.

DEFAULT_NAME = "SteppingStoneEvenSampling"
DEFAULT_N_VALUES = [2, 4, 8, 16, 32, 64]
DEFAULT_NUM_REPLICATES = 20
DEFAULT_SEED = 0


def n_leading(dimension):
    """How many leading structure axes to compare: 1 in 1D, 2 (the degenerate pair) in 2D."""
    return 1 if dimension == 1 else 2


def measure_eigenvalue(Z, n, L, V_lead, k_lead, rng):
    """
    Sample eigenvalue(s) of the leading structure mode(s), by overlap with V_lead.

    Ranked-by-magnitude selection would work in 1D, but in 2D the two nearly-degenerate
    sample eigenvectors are not guaranteed to be exactly PC1/PC2 once noise breaks the
    tie, so each theoretical column is matched to its own best-overlapping sample
    eigenvector (see pc_recovery.eigenvectors_by_overlap) and their eigenvalues averaged
    -- consistent with reporting the subspace's eigenvalue rather than an arbitrary
    per-axis split.

    Returns nu_meas = lambda_meas / floor_meas - 1, the same convention structure_
    eigenvalues() uses for the theoretical nu (floor at nu = 0).
    """
    sub = subsample(Z, n, L, rng)
    G = compute_sample_covariance(sub) * SPECTRUM_SCALE
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    K_demes = len(n)
    floor = float(vals[K_demes - 1:-1].mean())

    overlaps = np.abs(vecs.T @ V_lead)          # (N, k_lead)
    picked = [int(np.argmax(overlaps[:, j])) for j in range(k_lead)]
    lam_meas = float(np.mean(vals[picked]))
    return lam_meas / floor - 1.0, float(effective_L(sub)), sub.shape[1]


def analyse(simdir, dimension, n_values=DEFAULT_N_VALUES, L=None,
           num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, verbose=True):
    """One row per (m, n) cell for a single K simulation directory."""
    params, arrays = load_simulations(simdir)
    K = int(params["K"].iloc[0])
    K_demes = num_demes(K, dimension)
    n_available = int(params["n_per_deme"].iloc[0])
    k_lead = n_leading(dimension)

    rows = []
    cells = [(r, n) for r in params.itertuples() for n in n_values if n <= n_available]
    bar = tqdm(cells, disable=not verbose, desc=os.path.basename(simdir))
    for r, n_per in bar:
        Z = arrays[r.Index]
        n = np.full(K_demes, n_per, dtype=int)
        N = int(n.sum())
        L_cell = L if L is not None else int(r.L)

        modes, nu_theory, V_theory = closed_form_spectrum(K, r.M, n_per, dimension)
        V_lead, nu_lead = V_theory[:, :k_lead], nu_theory[:k_lead]

        res = pc_recovery(Z, n, V_lead, indices=list(range(k_lead)), L=L_cell,
                          num_replicates=num_replicates, seed=seed, nu=nu_lead)

        nu_meass, L_effs = [], []
        for rep in range(num_replicates):
            rng = np.random.default_rng([seed, rep, 1])
            nu_m, L_eff, _ = measure_eigenvalue(Z, n, L_cell, V_lead, k_lead, rng)
            nu_meass.append(nu_m)
            L_effs.append(L_eff)

        M_c = critical_M(K, n_per, res["L_eff"], dimension)
        rows.append(dict(
            simID=r.simID, dimension=dimension, K=K, K_demes=K_demes,
            m=r.m, Ne=r.Ne, M=r.M, n_per_deme=n_per, N=N,
            L=L_cell, L_eff=res["L_eff"], L_raw=res["L_raw"], gamma=res["gamma"],
            nu_theory=float(np.mean(nu_lead)),
            nu_meas=float(np.median(nu_meass)), nu_meas_se=_median_se(nu_meass),
            M_c=M_c, M_over_Mc=r.M / M_c if np.isfinite(M_c) and M_c > 0 else np.nan,
            median_cos=res["median_cos"], se_cos=res["se_cos"],
            analytical_cos=res["analytical_cos"], procrustes_d=res["procrustes_d"],
            num_replicates=num_replicates))
    return pd.DataFrame(rows)


def run(dimension, simulation_root=None, n_values=DEFAULT_N_VALUES, L=None,
        num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, name=DEFAULT_NAME,
        verbose=True):
    """Even-sampling analysis for one dimension's single K simulation set."""
    simulation_root = simulation_root or os.path.join(
        SIMULATION_ROOT, f"stepping_stone_{dimension}d")
    cells = sorted(d for d in os.listdir(simulation_root)
                   if d.startswith("K") and
                   os.path.isdir(os.path.join(simulation_root, d)))
    if not cells:
        raise FileNotFoundError(f"no K* subdirectories in {simulation_root}")
    if len(cells) > 1 and verbose:
        print(f"warning: {len(cells)} K cells found in {simulation_root}, "
              f"using all of them")

    frames = [analyse(os.path.join(simulation_root, c), dimension, n_values=n_values,
                      L=L, num_replicates=num_replicates, seed=seed, verbose=verbose)
             for c in cells]
    df = pd.concat(frames, ignore_index=True)

    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"even_{dimension}D.csv")
    df.to_csv(path, index=False)
    if verbose:
        print(f"wrote {len(df)} rows -> {path}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dimension", type=int, choices=(1, 2), required=True)
    p.add_argument("--simulation-root", default=None)
    p.add_argument("--n", type=int, nargs="+", default=DEFAULT_N_VALUES, dest="n_values")
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args()
    run(args.dimension, simulation_root=args.simulation_root, n_values=args.n_values,
        L=args.L, num_replicates=args.num_replicates, seed=args.seed, name=args.name)


if __name__ == "__main__":
    main()

# %%
