#%%
"""
Data for the three-panel RMT phenomena figure, from the two-population split simulations.

All three panels use N = 400 samples (phi = 0.5) and L_input = 2e4 SNP columns drawn per
replicate from results/simulations/split.

    A  one spectrum at a representative Fst: the 398 bulk eigenvalues, for the histogram
       and the two Marchenko-Pastur curves (raw L against L_eff).
    B  the gap lambda_1 / lambda_2 across the Fst sweep, median over replicates.
    C  the PC1 cosine similarity across the same sweep, plus the PC1 coordinates at three
       representative Fst values.

The spectra are rescaled by 0.5 (SPECTRUM_SCALE) because compute_sample_covariance
normalizes by Delta_bar / 2, which places the floor at 2 / (1 + alpha); the RMT statements
are written for a floor at 1 / (1 + alpha).

Fst is always the theoretical value from the filename, never re-estimated.

Outputs (results/analysis/RMTPhenomena):
    sweep.csv       per Fst: median gap and cosine similarity, L_raw, L_eff
    spectrum.csv    the panel-A spectrum, with its L_raw / L_eff
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

DEFAULT_SIMDIR = os.path.join(SIMULATION_ROOT, "split")
DEFAULT_NAME = "RMTPhenomena"

N = 400
L_INPUT = int(2e4)
NUM_REPLICATES = 20
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


def run(simdir=DEFAULT_SIMDIR, name=DEFAULT_NAME, n_total=N, L=L_INPUT,
        num_replicates=NUM_REPLICATES, seed=SEED, verbose=True):
    """Compute everything the figure needs and write the four tables."""
    params, arrays = load_simulations(simdir)
    n = [n_total // 2, n_total - n_total // 2]
    outdir = os.path.join(ANALYSIS_ROOT, name)

    # ---- panels B and C: sweep over Fst.
    rows = []
    for i, (_, prow) in enumerate(tqdm(list(params.iterrows()), total=len(params),
                                       disable=not verbose, desc="Fst sweep")):
        Fst = float(prow["Fst"])
        V, _ = theoretical_eigenvectors(split_T(Fst), n, indices=[0])
        v1 = V[:, 0]

        gaps, coss, L_effs, L_raws = [], [], [], []
        for rep in range(num_replicates):
            rng = np.random.default_rng([seed, i, rep])
            sub = subsample(arrays[i], n, L, rng)
            vals, vecs = spectrum(sub)
            gaps.append(vals[0] / vals[1])
            coss.append(abs(float(vecs[:, 0] @ v1)))
            L_effs.append(effective_L(sub))
            L_raws.append(sub.shape[1])

        rows.append(dict(Fst=Fst, simID=prow["simID"],
                         gap_median=float(np.median(gaps)),
                         cos_median=float(np.median(coss)),
                         L_raw=float(np.median(L_raws)),
                         L_eff=float(np.median(L_effs))))
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

    # ---- panel C insets: the sweep points closest to each target cosine similarity.
    pc_rows = []
    for target in COS_TARGETS:
        j = int(np.argmin(np.abs(sweep.cos_median.values - target)))
        Fst = float(sweep.Fst.iloc[j])
        i = int(params.index[params.simID == sweep.simID.iloc[j]][0])
        rng = np.random.default_rng([seed, i, 0])
        sub = subsample(arrays[i], n, L, rng)
        V, _ = theoretical_eigenvectors(split_T(Fst), n, indices=[0])
        vals, vecs = spectrum(sub)
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
        panelA_L_eff=float(spectrum_df.L_eff.iloc[0]))])

    os.makedirs(outdir, exist_ok=True)
    sweep.to_csv(os.path.join(outdir, "sweep.csv"), index=False)
    spectrum_df.to_csv(os.path.join(outdir, "spectrum.csv"), index=False)
    pc1_df.to_csv(os.path.join(outdir, "pc1.csv"), index=False)
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)

    if verbose:
        m = meta.iloc[0]
        print(f"\nN={n_total}  L_input={L:g}  L_raw={m.L_raw:.0f}  L_eff={m.L_eff:.0f}")
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
