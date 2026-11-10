#%%
"""
The maximum realizable cosine similarity when only a fraction of demes is sampled.

Sampling a fraction phi of the demes does not merely add noise to PC recovery -- it changes
which axis the theoretical covariance HAS. So even a perfect estimate of that axis cannot
match the full-lattice spatial axis that stands in for geographic position, and the measured
cosine similarity is capped at

    ceiling(phi) = < u_phi , u_1 >^2      mean over random deme subsets

    u_1     the full-lattice axis (every deme sampled), restricted to the sampled demes --
            the geographic proxy
    u_phi   the axis the sparse sample's own theoretical covariance really has

This is a property of the DEMOGRAPHY alone: no genotype data, no simulation, no RMT. It is
one number per (dimension, phi), and it is what the phase-transition panels get as a
horizontal ceiling.

Cheap by construction. F_hat comes straight from the resistance distance,

    F_hat = -(N / 2M) P_hat D_psi R D_psi P_hat

(stepping_stone_grid.f_hat_from_resistance), with R cached per habitat -- so a value costs
one small eigendecomposition. M enters only as an overall scale and therefore cannot move
the eigenvectors: the ceiling is independent of M, which is checked and reported rather
than assumed. 16 random deme configurations per phi, matching the replicate count of the
sweeps this annotates.

2D compares the degenerate (1,0)/(0,1) pair as a SUBSPACE (mean squared principal cosine,
equivalently the Procrustes-aligned average), since no canonical basis exists inside a
degenerate eigenspace. 1D reduces to cos^2 between two vectors, so both land on the same
[0, 1] scale as the measured curves.

Outputs (results/analysis/SparseSamplingCeiling):
    ceiling.csv        one row per (dimension, phi, replicate)
    ceiling_mean.csv   one row per (dimension, phi): median, mean, sd and quantiles
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import principal_cosines

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import (f_hat_from_resistance, num_demes, resistance_distance)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sweeps_analysis", "models"))  # models/
from stepping_stone import sample_demes, true_axis_restricted

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_ROOT = os.path.join(_PROJECT_ROOT, "results", "analysis")

DEFAULT_NAME = "SparseSamplingCeiling"

# Mirrors configs 06 and 08, the sweeps these ceilings annotate.
SPEC = {
    1: dict(K=25, N_total=200),
    2: dict(K=9, N_total=600),
}
PHI_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_N_SCHEMES = 16
DEFAULT_SEED = 0
# Any positive value gives the same eigenvectors; M is an overall scale on F_hat.
M_REF = 1.0


def n_per_deme_for(K, phi, dimension, N_total):
    """n-per-deme spreading N_total over the demes a fraction phi retains (configs 06/08)."""
    n_demes = max(int(round(float(phi) * num_demes(K, dimension))), 1)
    return max(int(N_total // n_demes), 1)


def orthonormalize(V):
    """An orthonormal basis for the span of V's columns.

    Restricting the full-lattice closed forms to a deme subset destroys the orthogonality
    they have on the whole lattice (at K=9 two restricted columns can correlate at 0.10),
    which would bias the principal cosines. QR fixes that without changing the span, and the
    comparison was always a subspace comparison.
    """
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if V.shape[1] == 1:
        return V / np.linalg.norm(V)
    return np.linalg.qr(V)[0]


def sparse_axis(K, dimension, demes, n_per_deme, M=M_REF):
    """
    u_phi: the leading structure axis (1D) or degenerate pair (2D) of the sparse sample's
    own theoretical covariance, in population space.

    R is the FULL-habitat resistance distance restricted to the sampled demes -- the habitat
    is unchanged by whom we sampled, so the Laplacian must not be rebuilt on the sub-graph.
    """
    k = 1 if dimension == 1 else 2
    R = resistance_distance(K, dimension)[np.ix_(demes, demes)]
    n = np.full(len(demes), float(n_per_deme))
    w, V = np.linalg.eigh(f_hat_from_resistance(R, n, M))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    # Drop the trivial psi = sqrt(phi) direction that mean-centering removes.
    psi = np.sqrt(n / n.sum())
    trivial = int(np.argmax(np.abs(psi @ V)))
    keep = [j for j in range(len(w)) if j != trivial]
    return V[:, keep][:, :k], w[keep][:k]


def lift(V_pop, n):
    """Population-space axes -> sample space; 1/sqrt(n_a) keeps the columns orthonormal."""
    n = np.asarray(n, dtype=int)
    V_pop = np.atleast_2d(np.asarray(V_pop, dtype=float))
    return np.repeat(V_pop / np.sqrt(n)[:, None], n, axis=0)


def ceiling_for(K, dimension, phi, n_per_deme, n_schemes=DEFAULT_N_SCHEMES,
                seed=DEFAULT_SEED, M=M_REF):
    """
    <u_phi, u_1>^2 for `n_schemes` random deme configurations at one phi.

    The rng call order (default_rng([seed, rep]) -> sample_demes) matches the sweeps'
    replicate loop, so these configurations are the ones the measurement actually drew.

    Returns a list of per-scheme values.
    """
    K_demes = num_demes(K, dimension)
    vals = []
    for rep in range(n_schemes):
        rng = np.random.default_rng([seed, rep])
        demes = sample_demes(K_demes, phi, rng)
        n = np.full(len(demes), n_per_deme, dtype=int)

        V_proxy = orthonormalize(true_axis_restricted(K, dimension, n_per_deme, demes,
                                                      K_demes))
        V_sparse, _ = sparse_axis(K, dimension, demes, n_per_deme, M=M)
        c = principal_cosines(V_proxy, orthonormalize(lift(V_sparse, n)))
        vals.append(float(np.mean(c ** 2)))
    return vals


def verify_m_independence(K, dimension, phi=0.4, n_per_deme=8, seed=0,
                          M_values=(0.05, 1.0, 500.0)):
    """Confirm the ceiling does not move with M, as F_hat's 1/M scaling requires."""
    vals = [ceiling_for(K, dimension, phi, n_per_deme, n_schemes=4, seed=seed, M=M)
            for M in M_values]
    vals = np.array(vals)
    return float(np.max(np.abs(vals - vals[0])))


def run(phi_values=None, n_schemes=DEFAULT_N_SCHEMES, seed=DEFAULT_SEED,
        out_name=DEFAULT_NAME, verbose=True):
    """Both dimensions, every phi -> ceiling.csv and ceiling_mean.csv."""
    phi_values = PHI_VALUES if phi_values is None else phi_values

    rows = []
    for dimension, spec in SPEC.items():
        K, N_total = spec["K"], spec["N_total"]
        dev = verify_m_independence(K, dimension, seed=seed)
        if verbose:
            print(f"  {dimension}D ceiling M-independence: max |Delta| = {dev:.2e}")
        for phi in phi_values:
            n_per = n_per_deme_for(K, phi, dimension, N_total)
            for rep, val in enumerate(ceiling_for(K, dimension, phi, n_per,
                                                  n_schemes=n_schemes, seed=seed)):
                rows.append(dict(dimension=dimension, K=K, K_demes=num_demes(K, dimension),
                                 phi=phi, n_per_deme=n_per, N_total=N_total,
                                 replicate=rep, ceiling=val, n_schemes=n_schemes,
                                 seed=seed))

    df = pd.DataFrame(rows)
    outdir = os.path.join(ANALYSIS_ROOT, out_name)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ceiling.csv")
    df.to_csv(path, index=False)

    # The single number per (dimension, phi) the figure draws, with its spread over
    # configurations: different subsets of the same size give genuinely different axes.
    # `mean` is what the figure draws. Replicates are never collapsed before fitting
    # (CLAUDE.md), so the fitted curve the ceiling is drawn over converges to the MEAN of
    # the replicate scatter, not its median. The per-subset distribution is strongly
    # left-skewed (1D phi=0.2: mean 0.8918, median 0.9469), so the two are not
    # interchangeable -- `median` is kept in the file for reference but the ceiling line
    # must match the curve's own statistic.
    summary = (df.groupby(["dimension", "K", "phi", "n_per_deme"], as_index=False)["ceiling"]
               .agg(median="median", mean="mean", sd="std",
                    lo=lambda s: s.quantile(0.025),
                    hi=lambda s: s.quantile(0.975), min="min", max="max"))
    mean_path = os.path.join(outdir, "ceiling_mean.csv")
    summary.to_csv(mean_path, index=False)

    if verbose:
        print(f"wrote {len(df)} rows -> {path}")
        print(f"wrote {len(summary)} rows -> {mean_path}\n")
        print(summary.to_string(index=False,
                                float_format=lambda v: f"{v:.4f}"))
    return df, summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-schemes", type=int, default=DEFAULT_N_SCHEMES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--phi", type=float, nargs="+", default=None)
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args()
    run(phi_values=args.phi, n_schemes=args.n_schemes, seed=args.seed,
        out_name=args.name)


if __name__ == "__main__":
    main()

# %%
