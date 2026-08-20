#%%
"""
Does PCA recover the leading spatial axis exactly where the theory says it should?

For each stepping-stone simulation, measure how well the sample PCs recover the leading
theoretical structure axes, and record it against M / M_c. The prediction under test is
that recovery collapses as M / M_c crosses 1, in both dimensions and for every K, L and
N -- so a sweep that works is one where curves at different K collapse onto each other
once the x axis is rescaled by M_c.

Degeneracy is the thing to get right in 2D. The leading eigenvalue there is doubly
degenerate (mu_10 = mu_01), and inside a degenerate eigenspace no canonical basis
exists: the two axes are only defined up to a rotation, so a per-PC correlation would
report spurious failure even well above threshold. The recoverable object is the plane
they span. leading_multiplet() finds the whole multiplet and pc_recovery compares
subspaces by principal angles, which gives one well-posed verdict for the pair. In 1D
the leading eigenvalue is simple and the same code path reduces to |cos| on one axis.

The N sweep comes from subsampling, not from new simulations: the .npy matrices carry
n_per_deme = 10 and cost scales with the number of demes, so n = 2, 4, 6, 10 all come
out of the same runs.

Outputs (results/analysis/stepping_stone_detection):
    detection.csv   one row per (simulation, n, L) cell, long format
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import (load_simulations, pc_recovery, theoretical_eigenvectors,
                         SIMULATION_ROOT)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import (critical_M, critical_M_approx, laplacian_eigenvalues,
                                 num_demes, structure_eigenvalues)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_ROOT = os.path.join(_PROJECT_ROOT, "results", "analysis")

DEFAULT_NUM_REPLICATES = 20
DEFAULT_SEED = 0
DEFAULT_N_VALUES = [2, 4, 6, 10]


def leading_multiplet(K, dimension, rtol=1e-8):
    """
    Indices of the structure axes sharing the largest eigenvalue.

    The eigenvalues nu_k = N / (M K mu_k) are ordered by ascending mu, so the leading
    multiplet is the set of smallest-mu modes that tie. In 1D that is always a single
    index; on a square 2D grid mu_10 = mu_01 makes it a pair. The tie is tested with a
    *relative* tolerance because the Kronecker sum leaves ~1e-16 of spread on an exact
    degeneracy.

    Returned as indices into the descending-nu ordering that theoretical_eigenvectors
    uses, so they can be passed straight through as `indices`.
    """
    mu = laplacian_eigenvalues(K, dimension)
    mu_1 = mu[0]
    return np.flatnonzero(np.abs(mu - mu_1) <= rtol * max(abs(mu_1), 1.0))


def stepping_stone_T(K, m, Ne, dimension):
    """
    Population-level coalescent time matrix, from the resistance-distance approximation.

    F_2 = T - (tau_i + tau_j)/2 is proportional to the resistance distance R, and under
    conservative isotropic migration with equal deme sizes every within-deme coalescent
    time is tau = 2 Ne K_demes (Strobeck). Rebuilding T from those two facts, rather than
    integrating the structured coalescent numerically, keeps the analysis in closed form
    and matches the theory the simulations are testing.
    """
    K_demes = num_demes(K, dimension)
    tau = 2.0 * Ne * K_demes

    # Scaled Laplacian of the migration graph; mbar = m cancels, so build it unscaled.
    A = np.zeros((K_demes, K_demes))
    if dimension == 1:
        for i in range(K_demes - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
    else:
        for row in range(K):
            for col in range(K):
                i = row * K + col
                if col + 1 < K:
                    j = row * K + col + 1
                    A[i, j] = A[j, i] = 1.0
                if row + 1 < K:
                    j = (row + 1) * K + col
                    A[i, j] = A[j, i] = 1.0
    Lap = np.diag(A.sum(axis=1)) - A

    Lap_plus = np.linalg.pinv(Lap)
    d = np.diag(Lap_plus)
    R = d[:, None] + d[None, :] - 2 * Lap_plus

    # F_2 = (K_demes / 4) R in units where the Laplacian is scaled by mbar = m; the
    # migration rate enters by dividing, since resistance is inverse conductance.
    F_2 = (K_demes / 4.0) * R / m
    T = F_2 + tau
    np.fill_diagonal(T, tau)
    return T


def analyse_cell(simdir, dimension, n_values=DEFAULT_N_VALUES, L=None,
                 num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, verbose=True):
    """
    One row per (simulation, n) for a single K cell.

    n_values are per-deme sample sizes to subsample down to; each gives a different N at
    fixed K, and therefore a different M_c, so the theoretical threshold is recomputed
    per row rather than taken from the simulation's CSV.
    """
    params, arrays = load_simulations(simdir)
    K = int(params["K"].iloc[0])
    K_demes = num_demes(K, dimension)
    n_available = int(params["n_per_deme"].iloc[0])
    indices = leading_multiplet(K, dimension)

    rows = []
    cells = [(r, n) for r in params.itertuples() for n in n_values if n <= n_available]
    bar = tqdm(cells, disable=not verbose, desc=os.path.basename(simdir))
    for r, n_per in bar:
        Z = arrays[r.Index]
        n = np.full(K_demes, n_per, dtype=int)
        N = int(n.sum())

        T = stepping_stone_T(K, r.m, r.Ne, dimension)
        V_theory, nu = theoretical_eigenvectors(T, n, indices=indices)

        L_cell = L if L is not None else int(r.L)
        res = pc_recovery(Z, n, V_theory, indices, L=L_cell,
                          num_replicates=num_replicates, seed=seed, nu=nu)

        M = 4.0 * r.Ne * r.m
        M_c = critical_M(K, n_per, res["L_eff"], dimension)
        rows.append(dict(
            simID=r.simID, dimension=dimension, K=K, K_demes=K_demes,
            m=r.m, Ne=r.Ne, M=M, n_per_deme=n_per, N=N,
            L=L_cell, L_eff=res["L_eff"], L_raw=res["L_raw"], gamma=res["gamma"],
            M_c=M_c, M_c_approx=critical_M_approx(K, n_per, res["L_eff"], dimension),
            M_over_Mc=M / M_c if np.isfinite(M_c) else np.nan,
            nu_1=float(nu[0]), multiplicity=len(indices),
            median_cos=res["median_cos"], se_cos=res["se_cos"],
            analytical_cos=res["analytical_cos"],
            procrustes_d=res["procrustes_d"],
            num_replicates=num_replicates))
    return pd.DataFrame(rows)


def run(dimension, K_values=None, simulation_root=None, n_values=DEFAULT_N_VALUES,
        L=None, num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED,
        outdir=None, filename=None, verbose=True):
    """Analyse every K cell of a stepping-stone sweep and write one long-format table."""
    simulation_root = simulation_root or os.path.join(
        SIMULATION_ROOT, f"SS_{dimension}D")
    if not os.path.isdir(simulation_root):
        raise FileNotFoundError(f"no such simulation root: {simulation_root}")

    cells = sorted(d for d in os.listdir(simulation_root)
                   if d.startswith("K") and
                   os.path.isdir(os.path.join(simulation_root, d)))
    if K_values is not None:
        keep = {f"K{K}" for K in K_values}
        cells = [c for c in cells if c.split("_")[0] in keep]
    if not cells:
        raise FileNotFoundError(f"no K* subdirectories in {simulation_root}")

    frames = [analyse_cell(os.path.join(simulation_root, c), dimension,
                           n_values=n_values, L=L, num_replicates=num_replicates,
                           seed=seed, verbose=verbose)
              for c in cells]
    out = pd.concat(frames, ignore_index=True)

    outdir = outdir or os.path.join(ANALYSIS_ROOT, "stepping_stone_detection")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename or f"detection_{dimension}D.csv")
    out.to_csv(path, index=False)
    if verbose:
        print(f"wrote {len(out)} rows -> {path}")
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dimension", type=int, choices=(1, 2), default=1, help="1D chain or 2D grid (default: 1)")
    p.add_argument("--K", type=int, nargs="+", default=None, help="restrict to these K cells (default: every K* directory found)")
    p.add_argument("--simulation-root", default=None, help="directory of K* simulation cells (default: results/simulations/SS_<dim>D)")
    p.add_argument("--n", type=int, nargs="+", default=DEFAULT_N_VALUES, dest="n_values", help=f"per-deme sample sizes to subsample to, giving the N sweep (default: {DEFAULT_N_VALUES})")
    p.add_argument("--L", type=int, default=None, help="SNP columns per replicate; default keeps each simulation's full L")
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    run(args.dimension, K_values=args.K, simulation_root=args.simulation_root,
        n_values=args.n_values, L=args.L, num_replicates=args.num_replicates,
        seed=args.seed, outdir=args.outdir)


if __name__ == "__main__":
    main()

# %%
