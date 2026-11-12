#%%
"""
Data for the RMT phenomena figure, from the two-population split simulations.

N haplotypes (N/2 per population, phi = 0.5) and L SNP columns, both read from the figure's
sweep config, src/sweeps_analysis/configs/17_split_rmt_phenomena.yaml.

    a     replicate spectra at three Fst values (the smallest simulated, ABOVE_FACTOR times
          the detection threshold, the largest): per replicate the largest sample
          eigenvalue and the squared cosine of PC1 with the true axis, plus the bulk
          eigenvalues of the first SPIKE_BULK_REPS replicates. Each spectrum is rescaled so
          its bulk has mean 1, the Marchenko-Pastur mean.
    b     the gap lambda_1 / lambda_2 across the Fst sweep, the top two eigenvalues computed
          once per Fst -- no replicates.
    c     PC1 at the same three Fst values. The cosine-similarity curve itself is drawn
          from results/sweep_summaries/rmt_phenomena, not computed here.

Fst is always the theoretical value from the filename, never re-estimated.

Outputs (results/analysis/RMTPhenomena):
    spike_top.csv   panel a    : per replicate lam1, cos2, L_raw, L_eff
    spike_bulk.csv  panel a    : bulk eigenvalues of the first replicates
    spike_meta.csv  panel a    : the Fst and simID of each
    gap.csv         panel b: per Fst, lambda_1 / lambda_2, L_raw, L_eff
    pc1.csv         PC1 coordinates for the three scatters under panel c
    meta.csv        one row: N, L_input, the two Fst_c
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
import preprocess
from pc_recovery import (load_simulations, subsample, effective_L, raw_L, SIMULATION_ROOT,
                         theoretical_eigenvectors)
from sweeps import ANALYSIS_ROOT, split_T
from utils import compute_sample_covariance

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "split")
DEFAULT_NAME = "RMTPhenomena"

SETTING_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sweeps_analysis", "configs", "17_split_rmt_phenomena.yaml")


def figure_setting(path=SETTING_PATH):
    """(N, L) of the figure, from its sweep config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return int(cfg["sampling"]["vary"]["values"][0]), int(cfg["fixed"]["L"])


SEED = 0
SPECTRUM_SCALE = 0.5

# Top row: many replicate subsamples at each of the three inset Fst values (INSET_KEY).
SPIKE_REPS = 400
SPIKE_BULK_REPS = 10

# The three Fst shown: the smallest simulated, ABOVE_FACTOR x the detection threshold (PC1
# partly aligned), and the largest simulated.
INSET_KEY = ("small", "threshold", "large")
ABOVE_FACTOR = 1.4


def subsample_first_L(Z, n, L, rng):
    """
    Row-subsample, filter to polymorphic loci, then take the first L of them.

    pc_recovery.subsample's default L draws L columns *before* filtering out the
    monomorphic ones, scattered at random across the full width -- at N=1000, L=1e5 out
    of 200000 columns that was ~1e5 non-sequential reads per Fst on a 4 GB memory-mapped
    file, which dominated runtime (the dense Z@Z.T covariance and its eigh are the real,
    unavoidable cost at this N and L -- ~95s/Fst -- but the scattered read was worse still).
    Row-subsetting first is one contiguous-ish read (Z[idx, :], all columns, few rows);
    the polymorphic filter and the first-L column slice are both then in-memory and cheap.
    Deterministic in which L columns are kept (not random), which is fine here since L is
    a subsample-size choice, not a quantity the theory is sensitive to the identity of.
    """
    sub_all = subsample(Z, n, L=None, rng=rng)  # rows only, already polymorphic-filtered
    if L is not None and L < sub_all.shape[1]:
        return sub_all[:, :L]
    return sub_all


def spectrum(Z):
    """Descending eigenvalues and eigenvectors of the rescaled sample GRM."""
    G = compute_sample_covariance(np.asarray(Z, dtype=np.float64)) * SPECTRUM_SCALE
    vals, vecs = np.linalg.eigh(G)
    return vals[::-1], vecs[:, ::-1]


def top2_spectrum(Z):
    """
    The two largest eigenvalues/vectors of the rescaled sample GRM, descending.

    Just spectrum() sliced to the top two: at N=1000, scipy.linalg.eigh's subset_by_index
    (LAPACK partial solver) measured no faster than a full eigh (~65s either way), so there
    is nothing to gain from the partial route here -- the Z@Z.T covariance build (~30s) and
    the O(N^3) eigh (~65s) are both real, unavoidable cost at this N and L=1e5.
    """
    vals, vecs = spectrum(Z)
    return vals[:2], vecs[:, :2]


def spike_replicates(params, arrays, targets, n, L, seed, verbose=True):
    """Replicate spectra at each Fst in targets {label: Fst}: (top, bulk, meta) tables."""
    u = np.repeat([1.0, -1.0], n) / np.sqrt(np.sum(n))
    top, bulk, meta = [], [], []
    for label, target in targets.items():
        i = int((np.log(params.Fst) - np.log(target)).abs().argmin())
        meta.append(dict(label=label, Fst=float(params.Fst.iloc[i]), simID=params.simID.iloc[i],
                         ploidy=preprocess.ploidy()))
        for rep in tqdm(range(SPIKE_REPS), disable=not verbose, desc=f"spike replicates ({label})"):
            sub = subsample_first_L(arrays[i], n, L, np.random.default_rng([seed, i, rep]))
            vals, vecs = spectrum(sub)
            bulk_vals = vals[1:preprocess.rank(len(vals))]   # non-zero, minus the spike
            scale = bulk_vals.mean()
            top.append(dict(label=label, rep=rep, lam1=vals[0] / scale,
                            cos2=float(vecs[:, 0] @ u) ** 2,
                            L_raw=raw_L(sub), L_eff=effective_L(sub)))
            if rep < SPIKE_BULK_REPS:
                bulk.append(pd.DataFrame(dict(label=label, rep=rep, eigenvalue=bulk_vals / scale)))
    return pd.DataFrame(top), pd.concat(bulk, ignore_index=True), pd.DataFrame(meta)


def pick_inset_fst(fst, fst_c):
    """The three Fst shown: the smallest, the one nearest ABOVE_FACTOR x threshold, the largest."""
    fst = np.sort(np.asarray(fst, dtype=float))
    near = fst[np.abs(np.log(fst) - np.log(ABOVE_FACTOR * fst_c)).argmin()]
    return dict(zip(INSET_KEY, (float(fst[0]), float(near), float(fst[-1]))))


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, n_total=None, L=None,
        seed=SEED, verbose=True):
    """Compute everything the figure needs and write the four tables."""
    N_cfg, L_cfg = figure_setting()
    n_total, L = n_total or N_cfg, L or L_cfg
    params, arrays = load_simulations(simdir)
    n = [n_total // 2, n_total - n_total // 2]
    outdir = os.path.join(ANALYSIS_ROOT, name)

    # ---- panel b: gap lambda_1 / lambda_2 across the Fst sweep, one pass per Fst.
    gap_rows = []
    for i, (_, prow) in enumerate(tqdm(list(params.iterrows()), total=len(params),
                                       disable=not verbose, desc="Fst sweep (panel b)")):
        Fst = float(prow["Fst"])
        rng = np.random.default_rng([seed, i])
        sub = subsample_first_L(arrays[i], n, L, rng)
        vals, _ = top2_spectrum(sub)
        gap_rows.append(dict(Fst=Fst, simID=prow["simID"],
                             gap=float(vals[0] / vals[1]),
                             L_raw=float(raw_L(sub)), L_eff=float(effective_L(sub))))
    gap = pd.DataFrame(gap_rows).sort_values("Fst").reset_index(drop=True)

    # ---- the two critical Fst, from the median L_raw and L_eff of the sweep above.
    L_raw_med = float(gap.L_raw.median())
    L_eff_med = float(gap.L_eff.median())
    meta = pd.DataFrame([dict(
        N=n_total, L_input=L, seed=seed,
        L_raw=L_raw_med, L_eff=L_eff_med,
        gamma_raw=n_total / L_raw_med, gamma_eff=n_total / L_eff_med,
        Fst_c_raw=2.0 / np.sqrt(n_total * L_raw_med),
        Fst_c_eff=2.0 / np.sqrt(n_total * L_eff_med))])

    # ---- top row: replicate spectra below, just above and far above the threshold.
    targets = pick_inset_fst(params.Fst, float(meta.Fst_c_eff.iloc[0]))
    spike_top, spike_bulk, spike_meta = spike_replicates(params, arrays, targets, n, L, seed,
                                                         verbose)

    # ---- PC1 at the same three Fst.
    pc_rows = []
    for key, Fst in targets.items():
        i = int(params.index[np.isclose(params.Fst.values, Fst, rtol=1e-9)][0])
        rng = np.random.default_rng([seed, i])
        sub = subsample_first_L(arrays[i], n, L, rng)
        V, _ = theoretical_eigenvectors(split_T(Fst), n, indices=[0])
        vals, vecs = top2_spectrum(sub)
        pc1 = vecs[:, 0]
        pc1 = pc1 if pc1 @ V[:, 0] >= 0 else -pc1        # fix the arbitrary sign
        pc_rows.append(preprocess.individuals(pd.DataFrame(dict(
            inset=key, Fst=Fst, population=np.repeat([0, 1], n), pc1=pc1))))
    pc1_df = pd.concat(pc_rows, ignore_index=True)


    os.makedirs(outdir, exist_ok=True)
    gap.to_csv(os.path.join(outdir, "gap.csv"), index=False)
    spike_top.to_csv(os.path.join(outdir, "spike_top.csv"), index=False)
    spike_bulk.to_csv(os.path.join(outdir, "spike_bulk.csv"), index=False)
    spike_meta.to_csv(os.path.join(outdir, "spike_meta.csv"), index=False)
    pc1_df.to_csv(os.path.join(outdir, "pc1.csv"), index=False)
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)

    if verbose:
        m = meta.iloc[0]
        print(f"\nN={n_total}  L_input={L:g}  L_raw={m.L_raw:.0f}  L_eff={m.L_eff:.0f}")
        print(f"Fst_c: raw L = {m.Fst_c_raw:.4e},  L_eff = {m.Fst_c_eff:.4e}")
        print(spike_top.groupby("label")[["lam1", "cos2", "L_eff"]].mean().round(4).to_string())
        print("Fst shown: " + ", ".join(f"{k}: {v:.3e}" for k, v in targets.items()))
        print(f"-> {outdir}")
    return gap, spike_top, pc1_df, meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--N", type=int, default=None, help="default: the sweep config's")
    p.add_argument("--L", type=int, default=None, help="default: the sweep config's")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(simdir=args.simdir, name=args.name, n_total=args.N, L=args.L, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
