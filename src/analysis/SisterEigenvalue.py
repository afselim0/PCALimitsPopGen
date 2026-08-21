#%%
"""
Test nu_1 = N * H(phi_1, phi_2) * Fst for a recently split sister pair, H the harmonic mean.

Hypothesis. When K subpopulations are sampled and two of them (the "sister" pair) split
recently, with equal within-population coalescent times and equal coalescent times to every
other population, the eigenvector separating the pair is

    u_1 = (sqrt(phi_2), -sqrt(phi_1), 0, ..., 0) / sqrt(phi_1 + phi_2)

with eigenvalue nu_1 = eps_0 + (2*N*phi_1*phi_2/s) * f_12, s = phi_1 + phi_2. Taking
tau/tau_bar = 1 and eps_0 = 0 -- the approximation under test, appropriate when the
subpopulations have broadly similar diversity -- this reduces to

    nu_1 = N * H * Fst,      H = 2*phi_1*phi_2/(phi_1+phi_2) = harmonic mean(phi_1, phi_2)

The sampling factor is exactly the harmonic mean of the two sister sampling fractions,
which is worth stating that way rather than as a bare ratio:

  * It reduces to the K = 2 result. When the sisters are the whole sample, phi_1+phi_2 = 1
    and H(phi, 1-phi) = 2*phi*(1-phi), giving nu = 2*N*phi*(1-phi)*Fst -- the form verified
    exactly against -H T H / T_bar. At phi_1 = phi_2 = 1/2 that is nu = N*Fst/2.
  * It says the penalty for other populations being in the sample is precisely that
    phi_1 + phi_2 < 1 now, rather than = 1.
  * A harmonic mean is dominated by its smaller argument (H <= 2*min(phi_1, phi_2)), so
    under-sampling one sister costs detectability in proportion to *that* fraction however
    many samples the other sister contributes. That is the design consequence.

results/simulations/african_split realizes exactly this structure: YRI2 is
built from YRI by copying its migration rates, so T_{YRI,c} = T_{YRI2,c} for c in
{CEU, CHB} and tau_YRI = tau_YRI2 by symmetry.

How the four variables are tested without a 4-D grid. The prediction is a *product*,
nu = N * H * Fst, so it factorizes into claims that can be tested separately:

    collapse  nu_meas vs nu_pred over every cell, however generated -- one line, slope 1
    Fst       nu vs Fst at fixed N, H          -- slope 1 on log-log
    N         nu vs N   at fixed Fst, H        -- slope 1
    H         nu vs H   at fixed N, Fst        -- slope 1

The H panel is the discriminating one: naive alternatives (phi_1*phi_2, or the K=2 form
phi_1*(1-phi_1) applied as if the sisters were the whole sample) predict different curves.
Two sampling families are therefore included -- one holding s = phi_1+phi_2 fixed while
varying phi_1*phi_2, one holding phi_1 = phi_2 while varying s -- which separate H from
those alternatives rather than merely being consistent with it.

Cells are a "star": a centre point plus one-factor-at-a-time excursions, plus a scattered
set for the collapse panel. That is ~100 cells rather than the ~10^4 a full grid would need.

Measurement. The sample GRM's structure eigenvalues are (1+nu)/(1+alpha) with alpha =
O(1/N), so nu_meas = lambda - 1 in units where the floor is 1. The sister axis is *not* the
leading eigenvalue -- the deep OOA splits dominate -- so it is located by projection onto
the predicted u_1. That is the ansatz under test, so the overlap is reported alongside: a
cell only supports the hypothesis if the eigenvector matches *and* carries the predicted
eigenvalue.

Sample order in the simulations is alphabetical: CEU, CHB, YRI, YRI2, so the sister pair is
blocks 2 and 3.

nu_exact is also computed per cell (exact_sister_nu.exact_sister_nu, from the full 4x4
coalescent-time matrix, not the eps_a=0 approximation), so nu_meas can be checked against
both the exact theoretical eigenvalue and the N*H*Fst approximation separately -- the first
tests whether measurement recovers theory, the second tests what the approximation costs
(see SisterApproximation.py for that comparison in isolation, theory vs theory).

Outputs (results/analysis/SisterEigenvalue):
    cells.csv    one row per (simulation, design, replicate)
    summary.csv  medians per cell, with the ratios nu_meas/nu_pred and nu_meas/nu_exact
    coal_times.npz  cached coalescent-time matrices, keyed by t_split
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from pc_recovery import load_simulations, subsample, effective_L, SIMULATION_ROOT
from exact_sister_nu import cached_coal_times, exact_sister_nu
from sweeps import ANALYSIS_ROOT
from utils import compute_sample_covariance

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "african_split")
DEFAULT_NAME = "SisterEigenvalue"

POPS = ["CEU", "CHB", "YRI", "YRI2"]
SISTER = (2, 3)                  # YRI, YRI2 in alphabetical block order
SPECTRUM_SCALE = 0.5             # compute_sample_covariance places the floor at 2
N_PER_POP_FULL = 5000            # per-population sample count in african_split config

DEFAULT_NUM_REPLICATES = 5
DEFAULT_SEED = 0
DEFAULT_L = None                 # None keeps every polymorphic locus


def harmonic_phi(n):
    """
    H = 2*phi_1*phi_2/(phi_1+phi_2), the harmonic mean of the sister sampling fractions.

    Equivalently 2*n_1*n_2/((n_1+n_2)*N): the harmonic mean of the two sister *counts*,
    divided by N.
    """
    n = np.asarray(n, dtype=float)
    phi = n / n.sum()
    a, b = SISTER
    return float(2.0 * phi[a] * phi[b] / (phi[a] + phi[b]))


def predicted_nu(Fst, n):
    """nu_1 = N*H*Fst, with eps_0 = 0 and tau/tau_bar = 1."""
    return np.asarray(n, dtype=float).sum() * harmonic_phi(n) * Fst


def sister_axis(n):
    """
    The predicted eigenvector u_1, lifted from population space to sample space.

    Population-space entry w_a is spread over that population's n_a samples as
    w_a/sqrt(n_a), which keeps the lifted vector unit-norm.
    """
    n = np.asarray(n, dtype=int)
    phi = n / n.sum()
    a, b = SISTER
    s = phi[a] + phi[b]

    w = np.zeros(len(n))
    w[a] = np.sqrt(phi[b] / s)
    w[b] = -np.sqrt(phi[a] / s)
    return np.repeat(w / np.sqrt(n), n)


def measure(Z, n, L, rng, K=len(POPS)):
    """
    The sister eigenvalue and its eigenvector's overlap with the prediction.

    Returns the eigenvalue of whichever eigenvector has the largest |overlap| with u_1,
    that overlap, its rank in the descending spectrum, and the realized SNP counts.

    The floor is *measured*, not assumed. The theoretical spectrum is
    lambda_k = (1+nu_k)/(1+alpha), so the bulk sits at 1/(1+alpha) rather than at 1 and
    nu = lambda/floor - 1 = lambda*(1+alpha) - 1. SPECTRUM_SCALE alone would place the
    floor at 1 only when alpha = 0; on this demography the deep OOA splits make alpha
    ~ 0.1, so assuming it away biases every nu by that same ~10%. Taking the ratio to the
    measured floor cancels 1+alpha without needing to compute it.

    The floor is the mean of the bulk: drop the K-1 leading structure eigenvalues and the
    trailing zero from mean-centering. The mean (not the median) is the right estimator --
    the MP density is left-skewed, so its median sits a couple of percent low.
    """
    sub = subsample(Z, n, L, rng)
    G = compute_sample_covariance(sub) * SPECTRUM_SCALE
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    floor = float(vals[K - 1:-1].mean())
    overlaps = np.abs(vecs.T @ sister_axis(n))
    k = int(np.argmax(overlaps))
    return dict(nu_meas=float(vals[k] / floor - 1.0),
                nu_meas_unnormed=float(vals[k] - 1.0),
                floor=floor, alpha_meas=float(1.0 / floor - 1.0),
                overlap=float(overlaps[k]), rank=k,
                L_eff=effective_L(sub), L_raw=sub.shape[1])


def _design(n_other, n1, n2):
    """Sample-size vector [CEU, CHB, YRI, YRI2] from the sister sizes and the others'."""
    n = [n_other, n_other, 0, 0]
    n[SISTER[0]], n[SISTER[1]] = n1, n2
    return np.array(n, dtype=int)


def build_designs():
    """
    Sampling designs, grouped by which factor each family isolates.

    Every family fixes all but one of (N, H) so the corresponding panel is a clean
    one-factor sweep; the "scatter" family varies both, for the collapse panel.
    """
    designs = []

    # Centre point, shared by the Fst family (which varies Fst across simulations).
    designs.append(("centre", "Fst", _design(250, 250, 250)))

    # N family: H fixed at 1/4 (equal sister sizes, sisters half the sample), N varied.
    for N4 in (50, 100, 150, 200, 300, 400, 500):
        designs.append((f"N{4 * N4}", "N", _design(N4, N4, N4)))

    # H family A: phi_1*phi_2 varied at fixed s (sisters always half the sample).
    # Distinguishes H from phi_1*phi_2, which would predict a different shape here.
    for n1 in (50, 100, 150, 200, 250):
        n2 = 500 - n1
        designs.append((f"H_A_{n1}", "H", _design(250, n1, n2)))

    # H family B: s varied at fixed phi_1 = phi_2 (sisters equal, their share varied).
    # Together with family A this pins H = 2*phi_1*phi_2/s rather than either factor alone.
    for n_s in (75, 150, 250, 350, 450):
        designs.append((f"H_B_{n_s}", "H", _design(250, n_s, n_s)))

    # Scatter: N and H both varied, for the collapse panel.
    rng = np.random.default_rng(12345)
    for j in range(10):
        n_other = int(rng.integers(60, 460))
        n1 = int(rng.integers(60, 460))
        n2 = int(rng.integers(60, 460))
        designs.append((f"scatter{j}", "scatter", _design(n_other, n1, n2)))

    return [(label, family, n) for label, family, n in designs
            if n.max() <= N_PER_POP_FULL and n.min() >= 2]


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, L=DEFAULT_L,
        num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, n_fst=8, verbose=True):
    """Measure the sister eigenvalue over every (simulation, design) cell."""
    params, arrays = load_simulations(simdir)
    outdir = os.path.join(ANALYSIS_ROOT, name)
    designs = build_designs()

    # The sister spike must clear the bulk to be measurable, so take the largest Fst
    # values; the Fst family then sweeps across them at the centre design.
    order = np.argsort(params.Fst.values)[::-1]
    fst_idx = order[:n_fst]
    # The Fst panel wants the whole usable range, the other families only need one Fst
    # each -- running all designs at all Fst would multiply the cost for no extra claim.
    centre_only = {"N", "H", "scatter"}

    # The exact structure eigenvalue (from the full 4x4 coalescent-time matrix), for
    # comparison against both the measurement and the N*H*Fst approximation. Only depends
    # on t_split, so it's computed once per distinct t_split among the Fst values used.
    t_splits = sorted(params.t_split.iloc[fst_idx].unique())
    coal_cache = os.path.join(outdir, "coal_times.npz")
    Tmap = cached_coal_times(t_splits, coal_cache, verbose=verbose)

    rows = []
    total = sum(len(fst_idx) if fam == "Fst" else 1 for _, fam, _ in designs)
    bar = tqdm(total=total * num_replicates, disable=not verbose, desc="cells")
    for label, family, n in designs:
        idxs = fst_idx if family == "Fst" else fst_idx[:1]
        for i in idxs:
            Fst = float(params.Fst.iloc[i])
            t_split = float(params.t_split.iloc[i])
            nu_pred = predicted_nu(Fst, n)
            nu_exact = exact_sister_nu(Tmap[t_split], n, sister_axis(n))["nu_exact"]
            for rep in range(num_replicates):
                rng = np.random.default_rng([seed, int(i), rep, len(label)])
                m = measure(arrays[i], n, L, rng)
                rows.append(dict(
                    simID=params.simID.iloc[i], t_split=t_split,
                    Fst=Fst, design=label, family=family, rep=rep,
                    N=int(n.sum()), n_sister1=int(n[SISTER[0]]), n_sister2=int(n[SISTER[1]]),
                    n_other=int(n[0]), H=harmonic_phi(n), nu_pred=nu_pred,
                    nu_exact=nu_exact, **m))
                bar.update(1)
    bar.close()

    df = pd.DataFrame(rows)
    df["ratio"] = df.nu_meas / df.nu_pred
    df["ratio_unnormed"] = df.nu_meas_unnormed / df.nu_pred
    df["ratio_exact"] = df.nu_meas / df.nu_exact
    df["nu_c"] = np.sqrt(df.N / df.L_eff)
    df["above_threshold"] = df.nu_pred > df.nu_c

    summary = df.groupby(["design", "family", "simID", "Fst", "N", "H",
                          "nu_pred", "nu_exact"]).agg(
        nu_meas=("nu_meas", "median"), nu_meas_sd=("nu_meas", "std"),
        nu_meas_unnormed=("nu_meas_unnormed", "median"),
        ratio=("ratio", "median"), ratio_unnormed=("ratio_unnormed", "median"),
        ratio_exact=("ratio_exact", "median"),
        floor=("floor", "median"), alpha_meas=("alpha_meas", "median"),
        overlap=("overlap", "median"),
        rank=("rank", "median"), L_eff=("L_eff", "median"),
        nu_c=("nu_c", "median"), num_replicates=("rep", "count")).reset_index()

    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "cells.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    if verbose:
        print(f"\n{len(summary)} cells, {num_replicates} replicates each")
        print(f"ratio nu_meas/nu_pred: median {summary.ratio.median():.4f}, "
              f"IQR {summary.ratio.quantile(.25):.4f}-{summary.ratio.quantile(.75):.4f}")
        print(f"  (floor-normalized; without it: "
              f"{summary.ratio_unnormed.median():.4f})")
        print(f"ratio nu_meas/nu_exact: median {summary.ratio_exact.median():.4f}, "
              f"IQR {summary.ratio_exact.quantile(.25):.4f}-"
              f"{summary.ratio_exact.quantile(.75):.4f}")
        print(f"measured floor 1/(1+alpha): median {summary.floor.median():.4f} "
              f"-> alpha = {summary.alpha_meas.median():.4f}")
        print(f"overlap with predicted u_1: median {summary.overlap.median():.4f}, "
              f"min {summary.overlap.min():.4f}")
        print(f"below threshold (excluded from the claim): "
              f"{int((~df.above_threshold).sum())} of {len(df)} rows")
        for fam, g in summary.groupby("family"):
            print(f"  {fam:8s} {len(g):3d} cells  ratio median {g.ratio.median():.4f}")
        print(f"\n-> {outdir}")
    return df, summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--L", type=int, default=DEFAULT_L,
                   help="SNP columns per replicate (default: all polymorphic)")
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--n-fst", type=int, default=8,
                   help="how many of the largest-Fst simulations to use (default: 8)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    run(simdir=args.simdir, name=args.name, L=args.L,
        num_replicates=args.num_replicates, n_fst=args.n_fst, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
