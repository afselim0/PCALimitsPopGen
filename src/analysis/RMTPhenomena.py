#%%
"""
Data for the three-panel RMT phenomena figure, from the two-population split simulations.

All three panels use N = 2000 samples (1000 per population, phi = 0.5) and every one of the
L = 2e5 SNP columns in results/simulations/split -- no column subsampling and no replicates.
Each simulation file is (20000, 200000) bool, block-ordered by population (10000 rows each),
so a plain row slice ([:1000] and [10000:11000]) reads two contiguous blocks; drawing a
random L-column subset instead (as earlier versions of this script did) forces ~1e5
scattered reads per replicate across a 4 GB memory-mapped file and was the actual cost, not
the eigendecomposition -- at this N a full eigh is fast regardless. With every column kept,
one pass over the data already is the population, so there is nothing left to replicate.

    A  one spectrum at a representative Fst: the bulk eigenvalues, for the histogram
       and the two Marchenko-Pastur curves (raw L against L_eff; gamma_eff is cross-checked
       against threshold_and_overlap.eta_moments, which agrees with N/L_eff exactly at
       eps = 0).
    B  the gap lambda_1 / lambda_2 across the Fst sweep.
    C  the PC1 cosine similarity across the same sweep, plus the PC1 coordinates at three
       representative Fst values.

The spectra are rescaled by 0.5 (SPECTRUM_SCALE) because compute_sample_covariance
normalizes by Delta_bar / 2, which places the floor at 2 / (1 + alpha); the RMT statements
are written for a floor at 1 / (1 + alpha).

Fst is always the theoretical value from the filename, never re-estimated.

Outputs (results/analysis/RMTPhenomena):
    sweep.csv       per Fst: gap and cosine similarity, L_raw, L_eff
    spectrum.csv    the panel-A spectrum, with its L_raw / L_eff
    pc1.csv         PC1 coordinates for the three panel-C scatters
    meta.csv        one row: N, L_input, the two Fst_c, the panel-A Fst
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh as scipy_eigh
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

N = 2000
L_INPUT = None  # keep every column; no subsampling
NUM_REPLICATES = 1  # one pass, no column subsampling to replicate over
SEED = 0
SPECTRUM_SCALE = 0.5

# Panel A wants a clean, well-populated bulk, so it uses a spectrum from a simulation whose
# spike has clearly detached: below threshold the leading eigenvalue is still inside the
# bulk and excluding it would bias the histogram's upper edge.
PANEL_A_INDEX = 70

# Panel C insets: the target cosine similarities to illustrate, from no recovery through
# partial to essentially perfect. The Fst actually shown is whichever sweep point comes
# closest to each target, so the choice adapts to the data rather than being hardcoded.
COS_TARGETS = (0.15, 0.65, 0.99)


def spectrum(Z):
    """Descending eigenvalues and eigenvectors of the rescaled sample GRM."""
    G = compute_sample_covariance(np.asarray(Z, dtype=np.float64)) * SPECTRUM_SCALE
    vals, vecs = np.linalg.eigh(G)
    return vals[::-1], vecs[:, ::-1]


def top2_spectrum(Z):
    """
    The two largest eigenvalues/vectors of the rescaled sample GRM, descending.

    A full eigh at N=1e4 takes ~26 minutes (LAPACK's dense O(N^3) solver has no way to
    skip the rest of the spectrum); scipy.linalg.eigh's subset_by_index asks LAPACK for
    only the top two, which panels B/C need (lambda_1/lambda_2 and PC1's cosine
    similarity) and is the only way 100 Fst x 20 replicates finishes in bounded time.
    """
    G = compute_sample_covariance(np.asarray(Z, dtype=np.float64)) * SPECTRUM_SCALE
    n = G.shape[0]
    vals, vecs = scipy_eigh(G, subset_by_index=[n - 2, n - 1])
    return vals[::-1], vecs[:, ::-1]


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, n_total=N, L=L_INPUT,
        num_replicates=NUM_REPLICATES, seed=SEED, verbose=True):
    """Compute everything the figure needs and write the four tables."""
    params, arrays = load_simulations(simdir)
    n = [n_total // 2, n_total - n_total // 2]
    outdir = os.path.join(ANALYSIS_ROOT, name)

    # ---- panels B and C: sweep over Fst. L=None (every column kept) makes subsample() a
    # plain contiguous row slice per population rather than a scattered column read, and
    # with no column subsampling there is nothing to replicate over -- one pass per Fst.
    rows = []
    for i, (_, prow) in enumerate(tqdm(list(params.iterrows()), total=len(params),
                                       disable=not verbose, desc="Fst sweep")):
        Fst = float(prow["Fst"])
        V, _ = theoretical_eigenvectors(split_T(Fst), n, indices=[0])
        v1 = V[:, 0]

        rng = np.random.default_rng([seed, i])
        sub = subsample(arrays[i], n, L, rng)
        vals, vecs = top2_spectrum(sub)

        rows.append(dict(Fst=Fst, simID=prow["simID"],
                         gap_median=float(vals[0] / vals[1]),
                         cos_median=abs(float(vecs[:, 0] @ v1)),
                         L_raw=float(sub.shape[1]),
                         L_eff=float(effective_L(sub))))
    sweep = pd.DataFrame(rows).sort_values("Fst").reset_index(drop=True)

    # ---- panel A: one spectrum, with the bulk isolated.
    rng = np.random.default_rng([seed, PANEL_A_INDEX, 0])
    sub_A = subsample(arrays[PANEL_A_INDEX], n, L, rng)
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

    # ---- panel C insets: the sweep points closest to each target cosine similarity.
    pc_rows = []
    for target in COS_TARGETS:
        j = int(np.argmin(np.abs(sweep.cos_median.values - target)))
        Fst = float(sweep.Fst.iloc[j])
        i = int(params.index[params.simID == sweep.simID.iloc[j]][0])
        rng = np.random.default_rng([seed, i, 0])
        sub = subsample(arrays[i], n, L, rng)
        V, _ = theoretical_eigenvectors(split_T(Fst), n, indices=[0])
        vals, vecs = top2_spectrum(sub)
        pc1 = vecs[:, 0]
        pc1 = pc1 if pc1 @ V[:, 0] >= 0 else -pc1        # fix the arbitrary sign
        pc_rows.append(pd.DataFrame(dict(
            target_cos=target, Fst=Fst, cos_median=float(sweep.cos_median.iloc[j]),
            population=np.repeat([0, 1], n), pc1=pc1)))
    pc1_df = pd.concat(pc_rows, ignore_index=True)

    # ---- metadata: the two critical Fst, from the median L_raw and L_eff of the sweep.
    L_raw_med = float(sweep.L_raw.median())
    L_eff_med = float(sweep.L_eff.median())
    meta = pd.DataFrame([dict(
        N=n_total, L_input=L, num_replicates=num_replicates, seed=seed,
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
    sweep.to_csv(os.path.join(outdir, "sweep.csv"), index=False)
    spectrum_df.to_csv(os.path.join(outdir, "spectrum.csv"), index=False)
    pc1_df.to_csv(os.path.join(outdir, "pc1.csv"), index=False)
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)

    if verbose:
        m = meta.iloc[0]
        L_input_str = f"{L:g}" if L is not None else "all"
        print(f"\nN={n_total}  L_input={L_input_str}  L_raw={m.L_raw:.0f}  L_eff={m.L_eff:.0f}")
        print(f"Fst_c: raw L = {m.Fst_c_raw:.4e},  L_eff = {m.Fst_c_eff:.4e}")
        print(f"panel A: Fst={m.panelA_Fst:.4e}, {len(bulk_A)} bulk eigenvalues, "
              f"support [{bulk_A.min():.4f}, {bulk_A.max():.4f}]")
        print("panel C insets: " + ", ".join(
            f"Fst={g.Fst.iloc[0]:.3e} (cos={g.cos_median.iloc[0]:.2f})"
            for _, g in pc1_df.groupby("target_cos")))
        print(f"-> {outdir}")
    return sweep, spectrum_df, pc1_df, meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=DEFAULT_SIMDIR)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--N", type=int, default=N)
    p.add_argument("--L", type=int, default=L_INPUT)
    p.add_argument("--num-replicates", type=int, default=NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(simdir=args.simdir, name=args.name, n_total=args.N, L=args.L,
        num_replicates=args.num_replicates, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
