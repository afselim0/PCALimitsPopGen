#%%
"""
Is the rank K-1 eigenvalue a clean bulk edge on OOA3, near the sister transition?

The eigenvalue-based threshold (eigenvalue_threshold.py) divides the sister eigenvalue by
lambda_bulk = rank K-1 = the 4th highest, and fits the BBP liftoff of that ratio. On the
K = 2 split it recovers nu_c to 1.02 +/- 0.01; on OOA3 it lands at 0.75-0.82 of the exact
nu_c, rising with N but extrapolating to ~0.85-0.9, not 1.

The suspicion this script tests: rank 3 is NOT the bulk edge here. K = 4, and the CEU/CHB
structure eigenvalues (nu ~ 30-50) have detached far above the sister spike, so rank 3 is
whatever sits below them -- which is the true edge only if exactly 3 eigenvalues have
separated. If the sister spike itself is at rank 2 and a fourth mode is partially detached,
rank 3 is an outlier and the ratio's denominator is too big, biasing nu_c low.

SisterPCAlignment.py records the sub-threshold baseline as 1.009, which says rank 3 tracks
the sister eigenvalue well BELOW threshold. That is consistent with both stories, because
below threshold the sister eigenvalue is itself pinned to the edge. The discriminating
measurement is the edge itself, independent of rank: fit the Marchenko-Pastur edge to the
BULK of the spectrum (ranks well below any structure) and compare.

Outputs (results/analysis/SisterBulkEdge/spectra.csv), one row per (Fst, N, replicate):
    lam0..lam5      the six largest eigenvalues
    lam_bulk_rank   rank K-1 = lam3, what the estimator currently uses
    mp_edge         MP edge implied by fitting gamma_eff to the bulk quantiles
    edge_ratio      lam_bulk_rank / mp_edge   -- 1 means rank 3 IS the edge
    lam_sister      the sister eigenvalue (rank 2), for context
    n_detached      how many eigenvalues exceed mp_edge
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from pc_recovery import subsample
from sweeps import ANALYSIS_ROOT
from utils import compute_sample_covariance

DEFAULT_NAME = "SisterBulkEdge"
SIMDIR = os.path.join("results", "simulations", "african_split")
POPS = ["CEU", "CHB", "YRI1", "YRI2"]
N_PER_POP_FULL = 5000
PHI_SISTER = 0.20      # matches 03_african_split_vary_N.yaml
L_DRAW = 50000


def n_vector(N, phi_sister=PHI_SISTER):
    """[CEU, CHB, YRI1, YRI2] counts at total N with each sister at phi_sister/2 of N."""
    n_sis = int(round(N * phi_sister / 2.0))
    n_oth = int(round((N - 2 * n_sis) / 2.0))
    return np.array([n_oth, n_oth, n_sis, n_sis], dtype=int)


def mp_edge_from_bulk(w, gamma_grid=None, lo=0.2, hi=0.8):
    """
    The MP edge implied by the bulk, with gamma_eff fitted to interior quantiles.

    Uses the quantile range [lo, hi] of the spectrum -- interior, so it is untouched by
    both the detached spikes at the top and the small-eigenvalue tail at the bottom. The
    MP quantile function has no closed form, so gamma is chosen on a grid by matching the
    bulk's own quantiles to those of the standard MP law scaled to the bulk's mean.
    """
    w = np.sort(np.asarray(w, dtype=float))[::-1]
    if gamma_grid is None:
        gamma_grid = np.exp(np.linspace(np.log(1e-4), np.log(0.9), 240))
    q = np.linspace(lo, hi, 25)
    emp = np.quantile(w, q)
    scale = emp.mean()
    best, best_err = np.nan, np.inf
    for g in gamma_grid:
        a, b = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2
        x = np.linspace(a, b, 4000)
        dens = np.sqrt(np.clip((b - x) * (x - a), 0, None)) / (2 * np.pi * g * x)
        cdf = np.cumsum(dens); cdf /= cdf[-1]
        theo = np.interp(q, cdf, x)
        theo = theo * (scale / theo.mean())          # match location, fit only shape
        err = float(np.mean((theo - emp) ** 2))
        if err < best_err:
            best, best_err = g, err
    a, b = (1 - np.sqrt(best)) ** 2, (1 + np.sqrt(best)) ** 2
    x = np.linspace(a, b, 4000)
    dens = np.sqrt(np.clip((b - x) * (x - a), 0, None)) / (2 * np.pi * best * x)
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    theo = np.interp(q, cdf, x)
    edge = b * (scale / theo.mean())
    return float(edge), float(best), float(np.sqrt(best_err))


def run_one(path, N, n_replicates, seed, L=L_DRAW):
    """Full spectra for one simulation file at one N, over replicates."""
    Z = np.load(path, mmap_mode="r")
    n = n_vector(N)
    rows = []
    for r in range(n_replicates):
        rng = np.random.default_rng([seed, r])
        Zs = subsample(Z, n, L=L, rng=rng,
                       n_available=np.full(len(POPS), N_PER_POP_FULL))
        G = compute_sample_covariance(Zs)
        w = np.sort(np.linalg.eigvalsh(G))[::-1]
        w = w[: len(w) - 1]                      # drop the mean-centering zero at the end
        edge, g_fit, q_rms = mp_edge_from_bulk(w)
        row = dict(N=N, replicate=r, L_used=int(Zs.shape[1]))
        for i in range(6):
            row[f"lam{i}"] = float(w[i])
        row.update(lam_bulk_rank=float(w[3]), lam_sister=float(w[2]),
                   mp_edge=edge, gamma_fit=g_fit, quantile_rms=q_rms,
                   edge_ratio=float(w[3]) / edge,
                   n_detached=int(np.sum(w > edge)))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", default=SIMDIR)
    ap.add_argument("--outdir", default=os.path.join(ANALYSIS_ROOT, DEFAULT_NAME))
    ap.add_argument("--N", type=int, nargs="+", default=[200, 600, 1000])
    ap.add_argument("--n-replicates", type=int, default=8)
    ap.add_argument("--n-sims", type=int, default=0, help="0 = all Fst values")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.simdir) if f.endswith(".npy"))
    if a.n_sims:
        files = files[:: max(1, len(files) // a.n_sims)][: a.n_sims]
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for i, f in enumerate(files):
        fst = float(f.replace("Fst", "").replace(".npy", ""))
        for N in a.N:
            for row in run_one(os.path.join(a.simdir, f), N, a.n_replicates, a.seed):
                row["Fst"] = fst
                rows.append(row)
        print(f"  [{i+1}/{len(files)}] Fst={fst:.4g} done", flush=True)

    out = os.path.join(a.outdir, "spectra.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
