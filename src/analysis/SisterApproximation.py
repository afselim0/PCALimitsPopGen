#%%
"""
The harmonic-mean approximation nu_1 = N*H*Fst against the exact eigenvalue.

nu_1 = N*H*Fst was derived under eps_a = 0 and tau_a/tau_bar = 1. This measures what those
assumptions cost, with no simulation in the loop: both sides are theoretical, so the
comparison is exact and noise-free, and the parameter range is limited only by arithmetic
rather than by the 5000-samples-per-population simulation cap.

The exact value is the structure eigenvalue of G = -H T H / T_bar built from the full
4 x 4 OOA3 coalescent-time matrix, normalized by the measured floor (so both sides are in
the same 1/(1+alpha) units). On this demography the assumption is visibly false --
tau/tau_bar = [0.855, 0.811, 1.166, 1.168] -- and the sisters sit ~17% above the mean,
which is why the approximation runs low.

Three panels' worth of structure, swept here:

    vary_N    H fixed, N from 100 to 5000. The ratio decays toward an asymptote, so part
              of the error is a finite-N term that never fully vanishes.
    vary_H    N fixed, the sisters sharing a fixed budget, split from lopsided to even.
              The ratio diverges as H -> 0: a harmonic mean is dominated by its smaller
              argument, and that is exactly where eps_a matters most.
    vary_Fst  N and H fixed, across all 8 cached t_split values. Fst enters both sides
              linearly, so this isolates whether the error depends on split depth.

Outputs (results/analysis/SisterApproximation):
    approximation.csv   one row per (sweep, cell): nu_pred, nu_exact and their ratio
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_sister_nu import cached_coal_times, exact_sister_nu
from sweeps import ANALYSIS_ROOT

_spec = importlib.util.spec_from_file_location(
    "_sister_eigenvalue",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "SisterEigenvalue.py"))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)
harmonic_phi, sister_axis, predicted_nu = _se.harmonic_phi, _se.sister_axis, _se.predicted_nu

DEFAULT_NAME = "SisterApproximation"
CACHE = os.path.join(ANALYSIS_ROOT, "SisterEigenvalue", "coal_times.npz")

# The 8 cached demographies. t_split -> Fst, from the simulation filenames.
T_SPLIT_FST = {722.1: 2.7396e-02, 756.5: 2.8609e-02, 792.5: 2.9871e-02, 830.2: 3.1185e-02,
               869.8: 3.2688e-02, 911.2: 3.4375e-02, 954.5: 3.6139e-02, 1000.0: 3.7983e-02}
REF_T = 1000.0

N_VALUES = np.array([100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4000, 5000])
PHI_SISTER = 0.125                  # H = 0.125 when both sisters are at this fraction
H_SWEEP_N = 1000
H_SWEEP_TOTAL = 200                 # sisters share this many samples
H_SPLITS = [2, 3, 5, 8, 10, 15, 20, 30, 40, 55, 70, 85, 100]


def design_even(N, phi=PHI_SISTER):
    """[n_o, n_o, n_s, n_s] with each sister at fraction phi of N."""
    n_s = int(round(phi * N))
    n_o = (N - 2 * n_s) // 2
    n = np.array([n_o, n_o, n_s, n_s], dtype=int)
    n[3] += N - n.sum()
    return n


def design_split(n1, total=H_SWEEP_TOTAL, N=H_SWEEP_N):
    """[n_o, n_o, n1, total-n1]: the sisters share a fixed budget, split unevenly."""
    n2 = total - n1
    n_o = (N - total) // 2
    n = np.array([n_o, n_o, n1, n2], dtype=int)
    n[0] += N - n.sum()
    return n


def cell(T, n, Fst, sweep, **extra):
    """One comparison: exact eigenvalue vs the harmonic-mean approximation."""
    e = exact_sister_nu(T, n, sister_axis(n))
    nu_pred = predicted_nu(Fst, n)
    return dict(sweep=sweep, N=int(n.sum()), H=harmonic_phi(n), Fst=Fst,
                n_sister1=int(n[2]), n_sister2=int(n[3]), n_other=int(n[0]),
                nu_pred=nu_pred, nu_exact=e["nu_exact"],
                ratio=e["nu_exact"] / nu_pred, alpha=e["alpha_exact"],
                overlap=e["overlap_exact"], **extra)


def run(name=DEFAULT_NAME, verbose=True):
    """Three theory-only sweeps of the approximation error."""
    Tmap = cached_coal_times(sorted(T_SPLIT_FST), CACHE, verbose=verbose)
    T_ref, Fst_ref = Tmap[REF_T], T_SPLIT_FST[REF_T]

    rows = [cell(T_ref, design_even(int(N)), Fst_ref, "vary_N") for N in N_VALUES]
    rows += [cell(T_ref, design_split(n1), Fst_ref, "vary_H") for n1 in H_SPLITS]
    rows += [cell(Tmap[t], design_even(H_SWEEP_N), T_SPLIT_FST[t], "vary_Fst", t_split=t)
             for t in sorted(T_SPLIT_FST)]

    df = pd.DataFrame(rows)
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "approximation.csv"), index=False)

    if verbose:
        tau = np.diag(T_ref)
        print(f"\ntau/tau_bar = {np.round(tau / tau.mean(), 4)}  (eps_a = 0 assumes all 1)")
        for sweep, g in df.groupby("sweep"):
            print(f"  {sweep:9s} ratio {g.ratio.min():.4f} .. {g.ratio.max():.4f}")
        print(f"\n-> {outdir}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args()
    run(name=args.name)


if __name__ == "__main__":
    main()

# %%
