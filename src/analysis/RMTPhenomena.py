#%%
"""
Data for the three-panel RMT phenomena figure, from the two-population split simulations.

N = 500 samples (250 per population, phi = 0.5), L_input = 2e4 SNP columns.

    A  one spectrum at a representative Fst: the bulk eigenvalues, for the histogram and
       the two Marchenko-Pastur curves (raw L against L_eff; gamma_eff is cross-checked
       against threshold_and_overlap.eta_moments, which agrees with N/L_eff exactly at
       eps = 0). Computed once -- no replicates.
    B  the gap lambda_1 / lambda_2 across the Fst sweep, the top two eigenvalues computed
       once per Fst -- no replicates.
    C  the PC1 cosine similarity across the same Fst sweep, read directly from the
       existing results/analysis/SingleSplitSweeps/vary_N.csv at N = 500 (median_cos, the
       measured median over its own replicates, not a fitted curve) rather than
       recomputed here. Only the three PC1 scatters are recomputed: one spectrum each at a
       very small Fst, one just above the detection threshold, and one very large Fst.

The spectra are rescaled by 0.5 (SPECTRUM_SCALE) because compute_sample_covariance
normalizes by Delta_bar / 2, which places the floor at 2 / (1 + alpha); the RMT statements
are written for a floor at 1 / (1 + alpha).

Fst is always the theoretical value from the filename, never re-estimated.

Outputs (results/analysis/RMTPhenomena):
    spectrum.csv    the panel-A spectrum, with its L_raw / L_eff
    gap.csv         panel B: per Fst, lambda_1 / lambda_2, L_raw, L_eff
    pc1.csv         PC1 coordinates for the three panel-C scatters
    meta.csv        one row: N, L_input, the two Fst_c, the panel-A Fst
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from pc_recovery import (load_simulations, subsample, effective_L, SIMULATION_ROOT,
                         theoretical_eigenvectors)
from sweeps import ANALYSIS_ROOT, split_T
from utils import compute_sample_covariance
from threshold_and_overlap import eta_moments

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "split")
DEFAULT_NAME = "RMTPhenomena"
COS_SWEEP_PATH = os.path.join(ANALYSIS_ROOT, "SingleSplitSweeps", "vary_N.csv")

N = 500
L_INPUT = int(2e4)
SEED = 0
SPECTRUM_SCALE = 0.5

# Panel A wants a clean, well-populated bulk, so it uses a spectrum from a simulation whose
# spike has clearly detached: below threshold the leading eigenvalue is still inside the
# bulk and excluding it would bias the histogram's upper edge.
PANEL_A_INDEX = 70

# Panel C insets: very small Fst (deep sub-threshold), just above the detection threshold,
# and very large Fst -- read off the existing SingleSplitSweeps/vary_N.csv sweep at N=500,
# not re-derived here.
INSET_KEY = ("small", "threshold", "large")


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


def _pick_inset_rows(cos_sweep):
    """
    The three Fst rows for panel C's insets: very small, just above threshold, very large.

    "Just above threshold" is the first row where nu_axis0 > sqrt(gamma), the BBP
    detectability condition -- not where analytical_cos turns positive, which is never
    exactly zero in this sweep (analytical_cosine floors its prediction at the finite-N
    random-overlap value sqrt(2/(pi*N)) rather than 0, so that test fires on the very first
    row and collapses "small" and "threshold" onto the same point).
    """
    d = cos_sweep.sort_values("Fst").reset_index(drop=True)
    above = d.index[d.nu_axis0 > np.sqrt(d.gamma)]
    j_thresh = int(above[0]) if len(above) else len(d) // 2
    return {"small": d.iloc[0], "threshold": d.iloc[j_thresh], "large": d.iloc[-1]}


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, n_total=N, L=L_INPUT,
        cos_sweep_path=COS_SWEEP_PATH, seed=SEED, verbose=True):
    """Compute everything the figure needs and write the four tables."""
    params, arrays = load_simulations(simdir)
    n = [n_total // 2, n_total - n_total // 2]
    outdir = os.path.join(ANALYSIS_ROOT, name)

    # ---- panel B: gap lambda_1 / lambda_2 across the Fst sweep, one pass per Fst.
    gap_rows = []
    for i, (_, prow) in enumerate(tqdm(list(params.iterrows()), total=len(params),
                                       disable=not verbose, desc="Fst sweep (panel B)")):
        Fst = float(prow["Fst"])
        rng = np.random.default_rng([seed, i])
        sub = subsample_first_L(arrays[i], n, L, rng)
        vals, _ = top2_spectrum(sub)
        gap_rows.append(dict(Fst=Fst, simID=prow["simID"],
                             gap=float(vals[0] / vals[1]),
                             L_raw=float(sub.shape[1]), L_eff=float(effective_L(sub))))
    gap = pd.DataFrame(gap_rows).sort_values("Fst").reset_index(drop=True)

    # ---- panel A: one spectrum, with the bulk isolated. Computed once.
    rng = np.random.default_rng([seed, PANEL_A_INDEX])
    sub_A = subsample_first_L(arrays[PANEL_A_INDEX], n, L, rng)
    vals_A, _ = spectrum(sub_A)
    # Drop the spike and the exact zero from mean-centering: neither is an MP draw.
    bulk_A = vals_A[1:-1]
    spectrum_df = pd.DataFrame(dict(eigenvalue=bulk_A))
    spectrum_df["Fst"] = float(params.Fst.iloc[PANEL_A_INDEX])
    spectrum_df["L_raw"] = sub_A.shape[1]
    spectrum_df["L_eff"] = effective_L(sub_A)
    spectrum_df["N"] = n_total

    # gamma_eff via the Ding & Yang locus-side moments (threshold_and_overlap.eta_moments),
    # the project's canonical route rather than the L_eff participation ratio: at eps = 0
    # (equal within-population coalescent times, as here) gamma_eff = gamma_raw * mu2 is
    # exactly gamma_raw * L/L_eff = N/L_eff, so the two routes agree numerically and this is
    # a change of derivation, not of value.
    p_A = np.asarray(sub_A, dtype=np.float64).mean(axis=0)
    h_A = p_A * (1 - p_A)
    mu2_A, _, _ = eta_moments(h_A)
    panelA_gamma_raw = n_total / sub_A.shape[1]
    panelA_gamma_eff = panelA_gamma_raw * mu2_A

    # ---- panel C: cosine similarity read from the existing SingleSplitSweeps sweep at
    # N = 1000, phi = 0.5 -- not recomputed here.
    cos_full = pd.read_csv(cos_sweep_path)
    cos_sweep = cos_full[cos_full.N == n_total].sort_values("Fst").reset_index(drop=True)
    if cos_sweep.empty:
        raise ValueError(f"{cos_sweep_path} has no rows with N == {n_total}")

    # ---- panel C insets: PC1 at very small / just-above-threshold / very large Fst.
    inset_rows = _pick_inset_rows(cos_sweep)
    pc_rows = []
    for key in INSET_KEY:
        row = inset_rows[key]
        Fst = float(row.Fst)
        i = int(params.index[np.isclose(params.Fst.values, Fst, rtol=1e-9)][0])
        rng = np.random.default_rng([seed, i])
        sub = subsample_first_L(arrays[i], n, L, rng)
        V, _ = theoretical_eigenvectors(split_T(Fst), n, indices=[0])
        vals, vecs = top2_spectrum(sub)
        pc1 = vecs[:, 0]
        pc1 = pc1 if pc1 @ V[:, 0] >= 0 else -pc1        # fix the arbitrary sign
        pc_rows.append(pd.DataFrame(dict(
            inset=key, Fst=Fst, cos_median=float(row.median_cos),
            population=np.repeat([0, 1], n), pc1=pc1)))
    pc1_df = pd.concat(pc_rows, ignore_index=True)

    # ---- metadata: the two critical Fst, from the median L_raw and L_eff of panel B's sweep.
    L_raw_med = float(gap.L_raw.median())
    L_eff_med = float(gap.L_eff.median())
    meta = pd.DataFrame([dict(
        N=n_total, L_input=L, seed=seed,
        L_raw=L_raw_med, L_eff=L_eff_med,
        gamma_raw=n_total / L_raw_med, gamma_eff=n_total / L_eff_med,
        Fst_c_raw=2.0 / np.sqrt(n_total * L_raw_med),
        Fst_c_eff=2.0 / np.sqrt(n_total * L_eff_med),
        panelA_Fst=float(params.Fst.iloc[PANEL_A_INDEX]),
        panelA_L_raw=float(spectrum_df.L_raw.iloc[0]),
        panelA_L_eff=float(spectrum_df.L_eff.iloc[0]),
        panelA_gamma_raw=panelA_gamma_raw,
        panelA_gamma_eff=panelA_gamma_eff)])

    os.makedirs(outdir, exist_ok=True)
    gap.to_csv(os.path.join(outdir, "gap.csv"), index=False)
    spectrum_df.to_csv(os.path.join(outdir, "spectrum.csv"), index=False)
    pc1_df.to_csv(os.path.join(outdir, "pc1.csv"), index=False)
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)
    cos_sweep.to_csv(os.path.join(outdir, "cos_sweep.csv"), index=False)

    if verbose:
        m = meta.iloc[0]
        print(f"\nN={n_total}  L_input={L:g}  L_raw={m.L_raw:.0f}  L_eff={m.L_eff:.0f}")
        print(f"Fst_c: raw L = {m.Fst_c_raw:.4e},  L_eff = {m.Fst_c_eff:.4e}")
        print(f"panel A: Fst={m.panelA_Fst:.4e}, {len(bulk_A)} bulk eigenvalues, "
              f"support [{bulk_A.min():.4f}, {bulk_A.max():.4f}]")
        print("panel C insets: " + ", ".join(
            f"{key}: Fst={g.Fst.iloc[0]:.3e} (cos={g.cos_median.iloc[0]:.2f})"
            for key, g in pc1_df.groupby("inset", sort=False)))
        print(f"-> {outdir}")
    return gap, spectrum_df, pc1_df, meta, cos_sweep


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--N", type=int, default=N)
    p.add_argument("--L", type=int, default=L_INPUT)
    p.add_argument("--cos-sweep-path", default=COS_SWEEP_PATH)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(simdir=args.simdir, name=args.name, n_total=args.N, L=args.L,
        cos_sweep_path=args.cos_sweep_path, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
