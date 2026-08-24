#%%
"""
PC recovery of the spatial axis vs M, for the recent-split demography (panmictic A + a
K x K grid B split off t_split generations ago). Figure 13 in CLAUDE.md.

Two sweeps, both subsampled from the single simulated K x K cell (only m is actually
swept by simulation; N-per-deme and phi are varied by subsampling):

    vary_n     phi = 0.5 fixed (n_A = n_per_deme_B * K^2, so B and A each hold half the
               sample), n_per_deme_B varied
    vary_phi   n_per_deme_B = 4 fixed, phi (B's share of N) varied by varying n_A

phi here is the *total* sample fraction from the structured demes B -- not the
fraction-of-demes-sampled phi in SteppingStoneRandomSampling, which is a different
sweep entirely (every deme is always sampled here; phi only trades off n_A vs the
B block's total).

The target axis is the (1,0)/(0,1) spatial pair from
split_stepping_stone_theory.within_B_F_hat -- population-space, restricted to B, then
lifted to sample space with a zero block for A. This pair is not reliably the leading
eigenvalue of the *full* A+B system once A carries a large sample share (checked
numerically: at phi=0.5, K=9 it is rank 0-1, but at small phi another axis outranks it),
so it is located by overlap (pc_recovery.pc_recovery(..., select_by_overlap=True)) as in
OOA3Demography.py, rather than assumed to be the leading PC.

Outputs (results/analysis/RecentSpatialOverlapVsM):
    overlap.csv   one row per (n_per_deme_B, phi) cell, long format, sweep column
        distinguishes vary_n / vary_phi
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from crossing_fit import crossings_table, replicate_rows
from pc_recovery import load_simulations, pc_recovery, SIMULATION_ROOT
from split_stepping_stone_theory import within_B_F_hat
from sweeps import ANALYSIS_ROOT

DEFAULT_NAME = "RecentSpatialOverlapVsM"
DEFAULT_NUM_REPLICATES = 20
DEFAULT_SEED = 0

N_PER_DEME_VALUES = [2, 4, 8, 16, 32, 64]
PHI_FIXED_FOR_N = 0.5
PHI_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_PER_DEME_FIXED_FOR_PHI = 4


def spatial_axis(K, m, t_split, Ne_ancestral, n_per_deme_B, n_A):
    """
    The theoretical (1,0)/(0,1) spatial pair, lifted to full [A, B_00, ..., B_(K-1)(K-1)]
    sample space: zero on A's block (A carries none of the spatial signal), the
    population-space pair spread per-deme on B's block.
    """
    nu, V_pop = within_B_F_hat(K, m, t_split, Ne_ancestral, n_per_deme_B)
    leading = V_pop[:, :2]
    n_B = np.full(K * K, n_per_deme_B)
    lifted_B = np.repeat(leading / np.sqrt(n_B)[:, None], n_B, axis=0)
    V_theory = np.vstack([np.zeros((n_A, 2)), lifted_B])
    return V_theory, nu[:2]


def measure_cell(Z, K, m, t_split, Ne_ancestral, n_per_deme_B, n_A, n_per_pop_A_avail,
                 n_per_deme_B_avail, L, num_replicates, seed):
    """One (n_per_deme_B, n_A) cell at a given m: theoretical axis, PC recovery by overlap."""
    V_theory, nu = spatial_axis(K, m, t_split, Ne_ancestral, n_per_deme_B, n_A)
    n = np.concatenate([[n_A], np.full(K * K, n_per_deme_B)])
    n_available = np.concatenate([[n_per_pop_A_avail], np.full(K * K, n_per_deme_B_avail)])
    res = pc_recovery(Z, n, V_theory, indices=[0, 1], L=L, num_replicates=num_replicates,
                      seed=seed, nu=nu, select_by_overlap=True, n_available=n_available)
    return res, nu


def analyse_sweep(simdir, sweep_name, levels, level_key, fixed, K, t_split, Ne_ancestral,
                  L, num_replicates, seed, verbose=True):
    """One sweep (vary_n or vary_phi) across every simulated m."""
    params, arrays = load_simulations(simdir)
    K_demes = K * K

    rows, rep_rows = [], []
    cells = [(r, lvl) for r in params.itertuples() for lvl in levels]
    bar = tqdm(cells, disable=not verbose, desc=f"{sweep_name}")
    for r, lvl in bar:
        design = dict(fixed)
        design[level_key] = lvl
        n_per_deme_B = design["n_per_deme_B"]
        phi = design["phi"]
        # phi = B's total share of N; n_A solved from phi and n_per_deme_B.
        N_B_total = n_per_deme_B * K_demes
        n_A = int(round(N_B_total * (1 - phi) / phi))
        if n_A < 1 or n_A > int(r.n_per_pop_A) or n_per_deme_B > int(r.n_per_deme_B):
            continue  # outside what this simulation can subsample

        L_cell = L if L is not None else int(r.L)
        res, nu = measure_cell(arrays[r.Index], K, r.m, t_split, Ne_ancestral,
                               n_per_deme_B, n_A, int(r.n_per_pop_A), int(r.n_per_deme_B),
                               L_cell, num_replicates, seed)
        M = 4 * Ne_ancestral / K_demes * r.m
        row = dict(
            simID=r.simID, sweep=sweep_name, K=K, m=r.m, M=M,
            L=L_cell, n_per_deme_B=n_per_deme_B, n_A=n_A, phi=phi,
            N=n_A + N_B_total, nu_1=float(nu[0]),
            median_cos=res["median_cos"], se_cos=res["se_cos"],
            analytical_cos=res["analytical_cos"], procrustes_d=res["procrustes_d"],
            L_eff=res["L_eff"], gamma=res["gamma"], num_replicates=num_replicates)
        rows.append(row)
        # M is the swept axis for this figure, so it is the column the fit regresses against.
        base = {k: v for k, v in row.items() if k != "M"}
        rep_rows.extend(replicate_rows(base, M, "M", res["cos_replicates"], seed=seed))
    return pd.DataFrame(rows), pd.DataFrame(rep_rows)


def run(simulation_root=None, K=None, t_split=None, Ne_ancestral=None, L=None,
        num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, name=DEFAULT_NAME,
        verbose=True):
    simulation_root = simulation_root or os.path.join(SIMULATION_ROOT, "split_stepping_stone")
    cells = sorted(d for d in os.listdir(simulation_root)
                   if d.startswith("K") and os.path.isdir(os.path.join(simulation_root, d)))
    if not cells:
        raise FileNotFoundError(f"no K* subdirectories in {simulation_root}")
    simdir = os.path.join(simulation_root, cells[0])

    probe, _ = load_simulations(simdir, mmap=True)
    K = K if K is not None else int(probe["K"].iloc[0])
    t_split = t_split if t_split is not None else float(probe["t_split"].iloc[0])
    Ne_ancestral = Ne_ancestral if Ne_ancestral is not None else float(probe["Ne_ancestral"].iloc[0])

    df_n, rep_n = analyse_sweep(simdir, "vary_n", N_PER_DEME_VALUES, "n_per_deme_B",
                                dict(phi=PHI_FIXED_FOR_N), K, t_split, Ne_ancestral, L,
                                num_replicates, seed, verbose=verbose)
    df_phi, rep_phi = analyse_sweep(simdir, "vary_phi", PHI_VALUES, "phi",
                                    dict(n_per_deme_B=N_PER_DEME_FIXED_FOR_PHI), K, t_split,
                                    Ne_ancestral, L, num_replicates, seed, verbose=verbose)
    df = pd.concat([df_n, df_phi], ignore_index=True)
    df_rep = pd.concat([rep_n, rep_phi], ignore_index=True)

    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "overlap.csv")
    df.to_csv(path, index=False)
    rep_path = os.path.join(outdir, "overlap_replicates.csv")
    df_rep.to_csv(rep_path, index=False)

    # 1/M is the axis, so the fitted curve is increasing; a curve is one level of one sweep.
    df_rep = df_rep.assign(inv_M=1.0 / df_rep["M"])
    df_cross = pd.concat([
        crossings_table(df_rep[df_rep.sweep == "vary_n"], ["sweep", "n_per_deme_B"],
                        "inv_M", extra_keys=("K", "phi", "n_A", "N", "L", "L_eff",
                                             "gamma", "nu_1", "num_replicates")),
        crossings_table(df_rep[df_rep.sweep == "vary_phi"], ["sweep", "phi"], "inv_M",
                        extra_keys=("K", "n_per_deme_B", "n_A", "N", "L", "L_eff",
                                    "gamma", "nu_1", "num_replicates")),
    ], ignore_index=True)
    cross_path = os.path.join(outdir, "overlap_crossings.csv")
    df_cross.to_csv(cross_path, index=False)

    if verbose:
        print(f"wrote {len(df)} rows -> {path}")
        print(f"wrote {len(df_rep)} replicate rows -> {rep_path}")
        print(f"wrote {len(df_cross)} crossing rows -> {cross_path}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simulation-root", default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args()
    run(simulation_root=args.simulation_root, L=args.L,
        num_replicates=args.num_replicates, seed=args.seed, name=args.name)


if __name__ == "__main__":
    main()

# %%
