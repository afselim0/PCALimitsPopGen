#%%
"""
Demography preview and the YRI1/YRI2 sister-split PC, at two split depths.

Rows are chosen by Fst(YRI1, YRI2) rather than by t_split -- Fst is the quantity the
theory is written in, and it is what makes the two rows comparable to every other figure
in the set. Two targets, Fst = 1e-2 and 1e-3, spanning the swept range in
results/simulations/african_split from a clearly resolved split down to a marginal one.
For each:

    demography   the msprime Demography as a demes.Graph, for demesdraw
    pc_phi05     sample PC1 (by overlap with the predicted sister axis) at N=1000,
                 L = 70,000 polymorphic SNPs, phi_1 = phi_2 = 0.05 (50 YRI1 + 50 YRI2,
                 remainder split CEU/CHB)
    pc_phi20     the same at phi_1 = phi_2 = 0.20 (200 + 200)

The PC is located by overlap with the predicted sister axis (see SisterEigenvalue.sister_axis)
rather than by rank: the deep OOA splits carry larger eigenvalues than the recent YRI1/YRI2
split, so the sister axis is not PC1 of the sample covariance.

Outputs (results/analysis/OOA3Demography):
    pc1.csv    one row per (Fst, phi, sample): label, t_split, Fst, phi, rank (0-indexed
               rank of the selected PC), overlap (|cos| against the predicted axis),
               population, pc1
    meta.csv   one row per Fst target: t_split, Fst, simID
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

N_TOTAL = 1000
# L is capped by the data, not chosen freely: the simulations carry 2e5 loci, but only
# ~73k-80k of them still segregate once the matrix is cut to N=1000 samples (measured
# across all four cells; the binding one is Fst=1e-2 at phi=0.05, at 73,024). 1e5
# polymorphic SNPs is therefore unreachable here -- 70,000 is the largest round number
# every cell can supply, which keeps L identical across the whole figure.
N_SNPS = 70_000
PHI_VALUES = (0.05, 0.20)

# Two Fst values, chosen as the closest available simIDs to these targets rather than
# hardcoded indices, since which rows finished simulating can vary run to run.
FST_TARGETS = {"fst1e-2": 1e-2, "fst1e-3": 1e-3}


def design(phi, n_total=N_TOTAL):
    """n = [n_CEU, n_CHB, n_YRI1, n_YRI2], sisters at phi each, remainder split CEU/CHB."""
    n_s = int(round(phi * n_total))
    n_o = (n_total - 2 * n_s) // 2
    n = np.array([n_o, n_o, n_s, n_s], dtype=int)
    n[SISTER[1]] += n_total - n.sum()
    return n


def pick_rows(params, targets):
    """The simID whose Fst is closest to each target, on a log scale, keeping label order."""
    picked = {}
    logFst = np.log(params.Fst.values)
    for label, target in targets.items():
        picked[label] = int(np.abs(logFst - np.log(target)).argmin())
    return picked


def sample_pc1(Z, n, rng, L=N_SNPS):
    """Sample PC located by overlap with the predicted sister axis, sign-fixed against it.

    L is drawn from the polymorphic columns of the subsample, so every panel is built on
    exactly the same number of segregating sites regardless of n.

    Returns (pc, rank, overlap): the sign-fixed PC, its 0-indexed rank in the sample
    spectrum, and |cos| against the predicted axis. The rank is recorded rather than
    assumed -- the sister axis normally sits at rank 2 (PC3), but at the marginal Fst it
    need not, and the figure's axis label should follow what was actually selected.
    """
    from pc_recovery import subsample, sample_eigenvectors
    sub = subsample(Z, n, L, rng, L_polymorphic=True)
    v = sister_axis(n)
    pc = eigenvectors_by_overlap(sub, v[:, None])[:, 0]
    overlap = abs(float(pc @ v))

    V = sample_eigenvectors(sub, indices=range(min(10, sub.shape[0])))
    rank = int(np.argmax(np.abs(V.T @ pc)))
    return (pc if pc @ v >= 0 else -pc), rank, overlap


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, seed=0, verbose=True):
    params, arrays = load_simulations(simdir)
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    rows_idx = pick_rows(params, FST_TARGETS)

    meta_rows, pc_rows = [], []
    for label, i in rows_idx.items():
        t_split = float(params.t_split.iloc[i])
        Fst = float(params.Fst.iloc[i])
        meta_rows.append(dict(label=label, t_split=t_split, Fst=Fst,
                              simID=params.simID.iloc[i]))
        if verbose:
            print(f"{label}: Fst={Fst:.4e}, t_split={t_split:.1f}")

        for phi in PHI_VALUES:
            n = design(phi)
            rng = np.random.default_rng([seed, i, int(round(phi * 100))])
            pc1, rank, overlap = sample_pc1(arrays[i], n, rng)
            population = np.repeat(POPS, n)
            pc_rows.append(pd.DataFrame(dict(
                label=label, t_split=t_split, Fst=Fst, phi=phi, rank=rank,
                overlap=overlap, population=population, pc1=pc1)))
            if verbose:
                print(f"  phi={phi:.2f}: PC{rank + 1}, overlap={overlap:.3f}")

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
