#%%
"""
PC alignment for a recently split sister pair, across sample size and sampling balance.

Measures how well PCA recovers the sister-separating axis

    u_1 = (sqrt(phi_2), -sqrt(phi_1), 0, ..., 0) / sqrt(phi_1 + phi_2)

on results/simulations/OOA3_recent_african_split (K = 4: CEU, CHB, YRI, YRI2, in that
alphabetical block order; YRI/YRI2 are the sisters). Detectability is governed by

    nu_1 = N * H * Fst,   H = 2*phi_1*phi_2/(phi_1+phi_2) = harmonic mean(phi_1, phi_2)

so the axis separates when nu_1 > sqrt(N/L_eff), i.e. when

    H > H_c = sqrt(N/L_eff) / (N * Fst)

Two sweeps:

    vary_N.csv    phi_1 = phi_2 = 0.05 fixed, N from 200 to 1000
    vary_phi.csv  N = 1000 fixed, the sisters supplying 200 samples between them, split
                  from (5, 195) through (100, 100)

Why these values. The point is to place the phase transition *inside* the grid rather than
sweeping a regime where the answer is always the same. With L_input = 5e4, the OOA3
matrices give L_eff ~ 0.161 * L_input = 8050 (much lower than a panmictic sample's ~40%,
because the four-population frequency spectrum is far richer in rare variants), so at
N = 1000 and Fst = 1e-2 the critical harmonic mean is H_c = 0.035. Both grids straddle it:

  * the N sweep at H = 0.05 predicts |cos| = 0 at N = 400 rising to 0.64 at N = 1000
  * the phi sweep crosses H_c at n_1 ~ 20 of the 200 sister samples

The phi sweep is the substantive one. Every cell spends exactly the same 200 samples on the
sister pair, but H depends on their *balance*: (5, 195) gives H = 0.0097 and (100, 100)
gives H = 0.100, a factor of ten. So Fst = 1e-2 -- not a small value -- is predicted to be
completely undetectable under the lopsided design and cleanly recovered under the balanced
one, with the same sequencing effort. A harmonic mean is dominated by its smaller argument,
so the 195 extra samples in the first design buy essentially nothing.

Sample-size cap. Each population was simulated with 500 haploid samples, so N <= 2000 and
any design needs every n_a <= 500. The grids below respect that; run() asserts it.

Fst is always the theoretical value from the filename, never re-estimated.

Outputs (results/analysis/SisterPCAlignment):
    vary_N.csv    one row per (simulation, N cell)
    vary_phi.csv  one row per (simulation, sampling-balance cell)
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_recovery import load_simulations, pc_recovery, SIMULATION_ROOT
from exact_sister_nu import cached_coal_times, exact_sister_nu
from sweeps import ANALYSIS_ROOT

# Loaded by path, not by bare module name: results/plotting_scripts holds a *different*
# SisterEigenvalue.py (the figure script for the same analysis), so a plain
# `from SisterEigenvalue import ...` resolves to whichever directory happens to come first
# on sys.path. Every analysis module here has a same-named plotting counterpart, so this is
# a general hazard, not a one-off.
_spec = importlib.util.spec_from_file_location(
    "_sister_eigenvalue",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "SisterEigenvalue.py"))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)
SISTER, POPS = _se.SISTER, _se.POPS
harmonic_phi, sister_axis = _se.harmonic_phi, _se.sister_axis
N_PER_POP_FULL = _se.N_PER_POP_FULL

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "OOA3_recent_african_split")
DEFAULT_NAME = "SisterPCAlignment"

DEFAULT_L = 50000
DEFAULT_NUM_REPLICATES = 10
DEFAULT_SEED = 0

# --- N sweep: sisters held at phi_1 = phi_2 = 0.05, so H = 0.05 throughout.
PHI_SISTER = 0.05
N_VALUES = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])

# --- phi sweep: N = 1000, the sisters supplying SISTER_TOTAL samples between them.
PHI_SWEEP_N = 1000
SISTER_TOTAL = 200
SISTER_SPLITS = [(5, 195), (10, 190), (15, 185), (20, 180), (30, 170),
                 (40, 160), (50, 150), (70, 130), (85, 115), (100, 100)]


def design_N(N, phi=PHI_SISTER):
    """n = [n_other, n_other, n_sister, n_sister] with each sister at fraction phi of N."""
    n_s = int(round(phi * N))
    n_o = (N - 2 * n_s) // 2
    n = np.array([n_o, n_o, n_s, n_s], dtype=int)
    n[SISTER[1]] += N - n.sum()          # absorb the rounding residual
    return n


def design_phi(n1, n2, N=PHI_SWEEP_N):
    """n = [n_other, n_other, n1, n2], the other two populations splitting the remainder."""
    n_o = (N - n1 - n2) // 2
    n = np.array([n_o, n_o, 0, 0], dtype=int)
    n[SISTER[0]], n[SISTER[1]] = n1, n2
    n[0] += N - n.sum()
    return n


def check_feasible(designs):
    """Every population was simulated with N_PER_POP_FULL samples; nothing may exceed it."""
    for label, n in designs:
        if n.max() > N_PER_POP_FULL:
            raise ValueError(f"design {label} asks for {n.max()} samples from one "
                             f"population but only {N_PER_POP_FULL} were simulated: {list(n)}")
        if n.min() < 2:
            raise ValueError(f"design {label} has a population with < 2 samples: {list(n)}")


def measure_cell(Z, n, L, num_replicates, seed, nu=None):
    """
    PC alignment with the predicted sister axis, median over replicates.

    select_by_overlap is essential here: the sister axis is *not* PC1. The deep OOA splits
    carry far larger eigenvalues, so the sister axis sits around rank 2, and rank-based
    selection returns a near-orthogonal vector -- measured overlap 2e-4 where the true axis
    gives 0.97.

    nu is passed so analytical_cos is populated and floored at the finite-N random overlap
    rather than left NaN below threshold.
    """
    V = sister_axis(n)[:, None]
    return pc_recovery(Z, n, V, indices=[0], L=L, num_replicates=num_replicates,
                       seed=seed, nu=[nu] if nu is not None else None,
                       select_by_overlap=True)


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, L=DEFAULT_L,
        num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, verbose=True):
    """Run both sweeps and write one CSV each."""
    from tqdm import tqdm

    params, arrays = load_simulations(simdir)
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    # The BBP prediction needs the *exact* structure eigenvalue, not N*H*Fst. The latter
    # assumes eps_a = 0, which is false on this demography (tau/tau_bar spans 0.81-1.17)
    # and understates nu by ~1.3x -- which would understate the predicted overlap and make
    # a correct measurement look like an overshoot.
    Tmap = cached_coal_times(sorted(params.t_split.unique()),
                             os.path.join(ANALYSIS_ROOT, "SisterEigenvalue",
                                          "coal_times.npz"), verbose=verbose)

    designs_N = [(f"N{N}", design_N(int(N))) for N in N_VALUES]
    designs_phi = [(f"n1_{a}", design_phi(a, b)) for a, b in SISTER_SPLITS]
    check_feasible(designs_N)
    check_feasible(designs_phi)

    for sweep_name, designs in [("vary_N", designs_N), ("vary_phi", designs_phi)]:
        rows = []
        bar = tqdm(total=len(designs) * len(arrays), disable=not verbose, desc=sweep_name)
        for label, n in designs:
            for i, (_, prow) in enumerate(params.iterrows()):
                Fst = float(prow["Fst"])
                N = int(n.sum())
                H = harmonic_phi(n)
                nu_approx = N * H * Fst
                nu = exact_sister_nu(Tmap[float(prow["t_split"])], n,
                                     sister_axis(n))["nu_exact"]
                res = measure_cell(arrays[i], n, L, num_replicates, seed, nu=nu)
                gamma = N / res["L_eff"]
                rows.append(dict(
                    simID=prow["simID"], t_split=float(prow["t_split"]), Fst=Fst,
                    design=label, N=N, H=H,
                    n_sister1=int(n[SISTER[0]]), n_sister2=int(n[SISTER[1]]),
                    n_other=int(n[0]),
                    phi1=n[SISTER[0]] / N, phi2=n[SISTER[1]] / N,
                    nu=nu, nu_approx=nu_approx, gamma=gamma, nu_c=np.sqrt(gamma),
                    H_c=np.sqrt(gamma) / (N * Fst),
                    cos_median=res["median_cos"], se_cos=res["se_cos"],
                    analytical_cos=res["analytical_cos"],
                    L_raw=res["L_raw"], L_eff=res["L_eff"],
                    num_replicates=num_replicates, seed=seed))
                bar.update(1)
        bar.close()

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, f"{sweep_name}.csv"), index=False)
        if verbose:
            print(f"  wrote {len(df)} rows -> {sweep_name}.csv")

    if verbose:
        print(f"\n-> {outdir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--L", type=int, default=DEFAULT_L)
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    run(simdir=args.simdir, name=args.name, L=args.L,
        num_replicates=args.num_replicates, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
