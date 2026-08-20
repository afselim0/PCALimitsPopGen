#%%
"""
Replace the sqrt(gamma) threshold in the sweep outputs with the general BBP criterion.

nu_c = sqrt(gamma) is the special case of g(nu) = 1 when every eps_a = 0. On OOA3 the
within-population coalescent times are heterogeneous -- eps = [-0.145, -0.189, 0.166, 0.168]
-- so sqrt(gamma) is only an approximation, and a poor one: it runs from +36% at N = 200 to
-11% at N = 1000, crossing over near N = 600, so it is not even a consistent bias.

The general criterion is

    g(nu) = gamma * sum_a phi_a ((1+eps_a)/(nu-eps_a))^2 < 1

evaluated pointwise at nu, with the scalar threshold nu_c being the largest root of g = 1.
Both come from src/utils.

The floor does not fragment here: the four eps_a cluster into two tight pairs, so g = 1 has
only two roots and the largest-root scalar remains meaningful. That is checked rather than
assumed -- n_roots is written out so a demography where it fails is visible.

This is a post-hoc recomputation. nu, gamma and the sample sizes are already in the CSVs, so
nothing needs re-simulating.

Adds columns: nu_c (exact), nu_c_sqrt_gamma (the old approximation), detectable,
g_at_nu, H_c (exact), n_roots.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exact_sister_nu import cached_coal_times
from sweeps import ANALYSIS_ROOT
from utils import critical_eigenvalue, detection_g, is_detectable

SWEEP_DIR = os.path.join(ANALYSIS_ROOT, "SisterPCAlignment")
CACHE = os.path.join(ANALYSIS_ROOT, "SisterEigenvalue", "coal_times.npz")

# From the 900-cell fit nu_exact = SLOPE*(N H Fst) + PEDESTAL, R^2 = 0.99988.
SLOPE = 1.3788
PEDESTAL = 0.3441


def eps_of(T):
    """eps_a = tau_a/tau_bar - 1, the deviation of each within-population coalescent time."""
    tau = np.diag(T)
    return tau / tau.mean() - 1.0


def count_roots(phi, eps, gamma, lo=-0.9, hi=3.0, num=20000):
    """Roots of g = 1. Two means a single bulk; more means the floor has fragmented."""
    grid = np.linspace(lo, hi, num)
    sign = np.sign(detection_g(grid, phi, eps, gamma) - 1.0)
    return int(np.sum(np.abs(np.diff(sign)) == 2))


def correct(path, Tmap, verbose=True):
    """Recompute the threshold columns of one sweep CSV in place."""
    d = pd.read_csv(path)
    out = []
    for _, r in d.iterrows():
        n = np.array([int(r.n_other), int(r.n_other), int(r.n_sister1), int(r.n_sister2)])
        phi = n / n.sum()
        eps = eps_of(Tmap[float(r.t_split)])
        nu_c = critical_eigenvalue(phi, eps, r.gamma)
        out.append(dict(
            nu_c=nu_c,
            detectable=bool(is_detectable(r.nu, phi, eps, r.gamma)),
            g_at_nu=float(detection_g(r.nu, phi, eps, r.gamma)),
            n_roots=count_roots(phi, eps, r.gamma),
            # H_c inverts nu_c through the *fitted* relation nu = 1.379*(N H Fst) + 0.344
            # rather than nu = N H Fst, since the latter is what the pedestal invalidates.
            H_c=max((nu_c - PEDESTAL) / (SLOPE * r.N * r.Fst), 0.0)))
    add = pd.DataFrame(out, index=d.index)

    d["nu_c_sqrt_gamma"] = d["nu_c"]
    for c in add.columns:
        d[c] = add[c]
    d.to_csv(path, index=False)

    if verbose:
        ratio = d.nu_c / d.nu_c_sqrt_gamma
        flip = (d.detectable != (d.nu > d.nu_c_sqrt_gamma)).sum()
        print(f"  {os.path.basename(path)}: {len(d)} rows")
        print(f"    nu_c exact / sqrt(gamma): {ratio.min():.3f} .. {ratio.max():.3f}")
        print(f"    verdict flips vs sqrt(gamma): {flip} ({100*flip/len(d):.1f}%)")
        print(f"    max roots of g=1: {d.n_roots.max()} (2 = single bulk)")
    return d


def run(sweep_dir=SWEEP_DIR, verbose=True):
    paths = [os.path.join(sweep_dir, f) for f in ("vary_N.csv", "vary_phi.csv")]
    paths = [p for p in paths if os.path.exists(p)]
    t_splits = sorted(set(np.concatenate([pd.read_csv(p).t_split.unique() for p in paths])))
    Tmap = cached_coal_times(t_splits, CACHE, verbose=False)
    return [correct(p, Tmap, verbose=verbose) for p in paths]


if __name__ == "__main__":
    run()

# %%
