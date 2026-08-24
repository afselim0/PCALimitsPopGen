#%%
"""
Discriminate critical-window vs multi-spike-interaction as the source of the near-threshold
BBP bias (bbp_predicted_cos - cos_median), using the existing SisterPCAlignment sweeps.

Two candidate mechanisms for the +0.31 (near nu_c) -> 0 (nu/nu_c > 2) bias:

    critical-window   finite-N crossover around nu_c, shrinks with N at fixed nu/nu_c
    spike interaction correction from the CEU/CHB structure eigenvalues (nu ~ 30-50),
                      entering through gamma/(nu_1 - nu_other) -- an N-independent term
                      set by the GAP to the nearest other spike, not by N itself

Discriminating design: hold N fixed, vary only Fst (so nu_1/nu_c sweeps) while
nu_other (CEU/CHB-driven) and hence spike_gap stay ~fixed, since those are set by the
deep OOA splits and don't depend on the recent YRI1/YRI2 t_split. vary_phi.csv already has
this structure *within* each `design` (fixed N=1000, fixed H, fixed n_other=400, Fst
swept over all 100 t_split values) -- see SisterPCAlignment.py. vary_N.csv sweeps N at
fixed Fst, giving the complementary bias-vs-N-at-fixed-nu_1/nu_c panel.

This script adds nu_other_min / spike_gap (not in the original CSVs -- they only kept the
sister eigenvalue) by recomputing the full F_hat spectrum per row, then reports the
partial-correlation tests requested: bias vs N at fixed nu_1/nu_c, and bias vs spike_gap
at fixed nu_1/nu_c, both within the near-threshold band and in the nu_1/nu_c > 2 tail.

Outputs (results/analysis/SpikeInteraction):
    vary_phi_spikes.csv   vary_phi.csv rows + nu_other_min, spike_gap, bias
    vary_N_spikes.csv     vary_N.csv rows + nu_other_min, spike_gap, bias
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from exact_sister_nu import cached_coal_times, expand
from sweeps import ANALYSIS_ROOT
from utils import compute_F_hat

DEFAULT_NAME = "SpikeInteraction"
SISTER_POP = 3  # YRI2, index into the K=4 [CEU, CHB, YRI1, YRI2] population vector
CACHE = os.path.join(ANALYSIS_ROOT, "SisterEigenvalue", "coal_times.npz")


def bbp_overlap(nu, gamma):
    """Unfloored BBP |cos| prediction, matching results/plotting_scripts/SisterPCAlignment.py."""
    nu = np.asarray(nu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ov = np.where(nu > np.sqrt(gamma), (1 - gamma / nu ** 2) / (1 + gamma / nu), 0.0)
    return np.sqrt(np.clip(ov, 0.0, 1.0))


def spike_spectrum(T, n):
    """
    All K-1 structure eigenvalues of F_hat (descending), with the sister eigenvalue
    identified as the one whose eigenvector best matches the sister axis.

    Returns (nu_sister, nu_other_min): nu_other_min is the smallest of the remaining K-2
    structure eigenvalues (the CEU/CHB-driven ones on OOA3).
    """
    n = np.asarray(n, dtype=int)
    K = len(n)
    phi = n / n.sum()
    psi = np.sqrt(phi)

    w, V = np.linalg.eigh(compute_F_hat(T, n))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    trivial = int(np.argmax(np.abs(psi @ V)))
    keep = [j for j in range(K) if j != trivial]
    w, V = w[keep], V[:, keep]

    # sister axis in population space: nonzero only on the sister pair, weighted by sqrt(phi)
    sister_pop_axis = np.zeros(K)
    a, b = 2, 3  # YRI1, YRI2
    s = phi[a] + phi[b]
    sister_pop_axis[a] = np.sqrt(phi[b] / s)
    sister_pop_axis[b] = -np.sqrt(phi[a] / s)

    overlaps = np.abs(V.T @ sister_pop_axis)
    k_sister = int(np.argmax(overlaps))
    nu_sister = float(w[k_sister])
    nu_other = np.delete(w, k_sister)
    return nu_sister, float(nu_other.min())


def add_spike_columns(df, Tmap, verbose=True):
    """Recompute nu_other_min / spike_gap / bias for every row of a SisterPCAlignment CSV."""
    rows = []
    for _, r in df.iterrows():
        n = np.array([int(r.n_other), int(r.n_other), int(r.n_sister1), int(r.n_sister2)])
        T = Tmap[float(r.t_split)]
        nu_sister, nu_other_min = spike_spectrum(T, n)
        rows.append(dict(nu_sister_check=nu_sister, nu_other_min=nu_other_min))
    add = pd.DataFrame(rows, index=df.index)

    out = df.copy()
    out["nu_other_min"] = add["nu_other_min"]
    out["spike_gap"] = out["nu_other_min"] - out["nu"]
    out["nu_ratio"] = out["nu"] / out["nu_c"]
    out["bbp_pred"] = bbp_overlap(out["nu"], out["gamma"])
    out["bias"] = out["bbp_pred"] - out["cos_median"]

    if verbose:
        check_err = (add["nu_sister_check"] - out["nu"]).abs()
        print(f"  sister-eigenvalue cross-check: max |diff| = {check_err.max():.2e} "
              f"(should be ~0, confirms nu_other_min uses the same F_hat convention)")
    return out


def run(name=DEFAULT_NAME, verbose=True):
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    src = os.path.join(ANALYSIS_ROOT, "SisterPCAlignment")
    vary_phi = pd.read_csv(os.path.join(src, "vary_phi.csv"))
    vary_N = pd.read_csv(os.path.join(src, "vary_N.csv"))

    t_splits = sorted(set(vary_phi.t_split.unique()) | set(vary_N.t_split.unique()))
    Tmap = cached_coal_times(t_splits, CACHE, verbose=verbose)

    if verbose:
        print("vary_phi:")
    vp = add_spike_columns(vary_phi, Tmap, verbose=verbose)
    if verbose:
        print("vary_N:")
    vn = add_spike_columns(vary_N, Tmap, verbose=verbose)

    vp.to_csv(os.path.join(outdir, "vary_phi_spikes.csv"), index=False)
    vn.to_csv(os.path.join(outdir, "vary_N_spikes.csv"), index=False)
    if verbose:
        print(f"\n-> {outdir}")
    return vp, vn


if __name__ == "__main__":
    run()

# %%
