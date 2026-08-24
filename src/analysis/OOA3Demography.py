#%%
"""
Demography preview and the YRI1/YRI2 sister-split PC, at three split depths.

Three t_split values, all shallow (15, 50, 100 generations ago) -- the recent end of the
swept range in results/simulations/african_split, where the sister split is fine enough to
be worth showing at this resolution. For each:

    demography   the msprime Demography as a demes.Graph, for demesdraw
    pc_phi05     sample PC1 (by overlap with the predicted sister axis) at N=2000,
                 phi_1 = phi_2 = 0.05 (100 YRI1 + 100 YRI2, remainder split CEU/CHB)
    pc_phi20     the same at phi_1 = phi_2 = 0.20 (400 + 400)

The PC is located by overlap with the predicted sister axis (see SisterEigenvalue.sister_axis)
rather than by rank: the deep OOA splits carry larger eigenvalues than the recent YRI1/YRI2
split, so the sister axis is not PC1 of the sample covariance.

Outputs (results/analysis/OOA3Demography):
    pc1.csv    one row per (t_split, phi, sample): population, pc1 value
    meta.csv   one row per t_split: t_split, Fst, simID
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from pc_recovery import load_simulations, eigenvectors_by_overlap, SIMULATION_ROOT
from sweeps import ANALYSIS_ROOT

_spec = importlib.util.spec_from_file_location(
    "_sister_eigenvalue",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "SisterEigenvalue.py"))
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)
SISTER, POPS, sister_axis = _se.SISTER, _se.POPS, _se.sister_axis

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))
from demographies.african_split import demography as ooa3_demography

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "african_split")
DEFAULT_NAME = "OOA3Demography"

N_TOTAL = 2000
PHI_VALUES = (0.05, 0.20)

# Three shallow t_split values, chosen as the closest available simIDs to these targets
# rather than hardcoded indices, since which rows finished simulating can vary run to run.
T_SPLIT_TARGETS = {"t15": 15.0, "t50": 50.0, "t100": 100.0}


def design(phi, n_total=N_TOTAL):
    """n = [n_CEU, n_CHB, n_YRI1, n_YRI2], sisters at phi each, remainder split CEU/CHB."""
    n_s = int(round(phi * n_total))
    n_o = (n_total - 2 * n_s) // 2
    n = np.array([n_o, n_o, n_s, n_s], dtype=int)
    n[SISTER[1]] += n_total - n.sum()
    return n


def pick_rows(params, targets):
    """The simID closest to each target t_split, preserving the recent/intermediate/deep
    label order."""
    picked = {}
    for label, t in targets.items():
        i = int((params.t_split - t).abs().values.argmin())
        picked[label] = i
    return picked


def sample_pc1(Z, n, rng):
    """Sample PC located by overlap with the predicted sister axis, sign-fixed against it."""
    from pc_recovery import subsample
    sub = subsample(Z, n, None, rng)
    V = sister_axis(n)[:, None]
    pc1 = eigenvectors_by_overlap(sub, V)[:, 0]
    return pc1 if pc1 @ V[:, 0] >= 0 else -pc1


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, seed=0, verbose=True):
    params, arrays = load_simulations(simdir)
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    rows_idx = pick_rows(params, T_SPLIT_TARGETS)

    meta_rows, pc_rows = [], []
    for label, i in rows_idx.items():
        t_split = float(params.t_split.iloc[i])
        Fst = float(params.Fst.iloc[i])
        meta_rows.append(dict(label=label, t_split=t_split, Fst=Fst,
                              simID=params.simID.iloc[i]))
        if verbose:
            print(f"{label}: t_split={t_split:.1f}, Fst={Fst:.4e}")

        for phi in PHI_VALUES:
            n = design(phi)
            rng = np.random.default_rng([seed, i, int(round(phi * 100))])
            pc1 = sample_pc1(arrays[i], n, rng)
            population = np.repeat(POPS, n)
            pc_rows.append(pd.DataFrame(dict(
                label=label, t_split=t_split, phi=phi, population=population, pc1=pc1)))

    pc1_df = pd.concat(pc_rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)

    pc1_df.to_csv(os.path.join(outdir, "pc1.csv"), index=False)
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)

    if verbose:
        print(f"-> {outdir}")
    return pc1_df, meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(simdir=args.simdir, name=args.name, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
