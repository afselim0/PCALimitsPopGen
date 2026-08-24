#%%
"""
Recent-split demography (panmictic A + a K x K grid B) and its spatial PC pair, at three
representative migration rates. Figure 12 in CLAUDE.md.

For each of three m (no / intermediate / strong spatial structure, chosen by the
theoretical nu of the leading spatial pair relative to sqrt(gamma)), the sample PC pair
best matching the theoretical (1,0)/(0,1) spatial axis (by overlap -- see
RecentSpatialOverlapVsM, which established this axis is not reliably the leading PC of
the full A+B system) is recovered from a genotype matrix subsampled to N_total split
phi/(1-phi) between B (evenly over K^2 demes) and A, per CLAUDE.md's phi convention
(B's total share of N) -- not the simulation's raw full sample sizes, which only
coincidentally give a clean phi (the previous version plotted those directly, so
phi=0.5 silently meant whatever aggregate split the simulation config happened to use,
with each deme carrying the full simulated n_per_deme_B rather than its N_total*phi/K^2
share). Each B sample is labeled by its deme's grid position for the plotting script to
color by.

Outputs (results/analysis/RecentSpatialDemography):
    pc.csv     one row per sample: m_label, m, deme_row, deme_col (NaN for A), pc2, pc3
    meta.csv   one row per m_label: m, M, nu_1 (theoretical leading spatial eigenvalue)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import load_simulations, eigenvectors_by_overlap, subsample, SIMULATION_ROOT
from split_stepping_stone_theory import within_B_F_hat
from sweeps import ANALYSIS_ROOT

DEFAULT_NAME = "RecentSpatialDemography"
M_LABELS = ("weak", "intermediate", "strong")
DEFAULT_PHI = 0.5          # B's total share of N (CLAUDE.md's convention, matches figure 13)
DEFAULT_N_TOTAL = 10000


def spatial_axis(K, m, t_split, Ne_ancestral, n_per_deme_B, n_A):
    """Theoretical spatial pair lifted to [A, B_00, ..., B_(K-1)(K-1)] sample space."""
    nu, V_pop = within_B_F_hat(K, m, t_split, Ne_ancestral, n_per_deme_B)
    leading = V_pop[:, :2]
    n_B = np.full(K * K, n_per_deme_B)
    lifted_B = np.repeat(leading / np.sqrt(n_B)[:, None], n_B, axis=0)
    return np.vstack([np.zeros((n_A, 2)), lifted_B]), nu[:2]


def pick_m_rows(params, nu_1_per_row):
    """Weak/intermediate/strong: the rows with the largest, median, and smallest
    theoretical nu_1 (migration ~ inversely related to how well-separated the spatial
    axis is -- weak migration keeps demes distinct, so nu_1 is largest there)."""
    order = np.argsort(nu_1_per_row)[::-1]  # descending nu_1 == weak -> strong migration
    n = len(order)
    return {"weak": int(order[0]), "intermediate": int(order[n // 2]),
           "strong": int(order[-1])}


def sample_design(phi, N_total, K_demes, n_per_pop_A_avail, n_per_deme_B_avail):
    """
    n_A and n_per_deme_B for a total sample of N_total split phi/(1-phi) between B
    (spread evenly over K_demes) and A, per CLAUDE.md's convention: phi is B's total
    share of N, so each deme gets N_total*phi/K_demes samples, not the full simulated
    per-deme count -- plotting the raw simulated sizes instead (as this script used to)
    silently uses whatever aggregate split the simulation config happened to have, which
    only coincidentally matches a clean phi.
    """
    N_B_total = phi * N_total
    n_per_deme_B = max(int(round(N_B_total / K_demes)), 1)
    n_A = max(int(round(N_total - n_per_deme_B * K_demes)), 1)
    if n_A > n_per_pop_A_avail or n_per_deme_B > n_per_deme_B_avail:
        raise ValueError(
            f"phi={phi:g} at N_total={N_total} needs n_A={n_A}, n_per_deme_B={n_per_deme_B}, "
            f"but only {n_per_pop_A_avail} (A) / {n_per_deme_B_avail} (per B deme) are "
            f"available -- lower N_total or adjust phi")
    return n_A, n_per_deme_B


def run(simdir=None, t_split=None, Ne_ancestral=None, phi=DEFAULT_PHI,
        N_total=DEFAULT_N_TOTAL, seed=0, name=DEFAULT_NAME, verbose=True):
    simulation_root = simdir or os.path.join(SIMULATION_ROOT, "split_stepping_stone")
    if os.path.isdir(simulation_root) and any(
            d.startswith("K") for d in os.listdir(simulation_root)):
        cells = sorted(d for d in os.listdir(simulation_root) if d.startswith("K"))
        simulation_root = os.path.join(simulation_root, cells[0])

    params, arrays = load_simulations(simulation_root)
    K = int(params["K"].iloc[0])
    t_split = t_split if t_split is not None else float(params["t_split"].iloc[0])
    Ne_ancestral = Ne_ancestral if Ne_ancestral is not None else float(params["Ne_ancestral"].iloc[0])
    n_A, n_per_deme_B = sample_design(phi, N_total, K * K,
                                      int(params["n_per_pop_A"].iloc[0]),
                                      int(params["n_per_deme_B"].iloc[0]))
    if verbose:
        print(f"phi={phi:g}, N_total={N_total}: n_A={n_A}, n_per_deme_B={n_per_deme_B} "
             f"(x{K*K} demes = {n_per_deme_B * K * K})")

    nu_1_per_row = np.array([
        within_B_F_hat(K, float(r.m), t_split, Ne_ancestral, n_per_deme_B)[0][0]
        for r in params.itertuples()])
    rows_idx = pick_m_rows(params, nu_1_per_row)

    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    meta_rows, pc_rows = [], []
    K_demes = K * K
    for label, i in rows_idx.items():
        m = float(params.m.iloc[i])
        M = 4.0 * (Ne_ancestral / K_demes) * m
        V_theory, nu = spatial_axis(K, m, t_split, Ne_ancestral, n_per_deme_B, n_A)
        meta_rows.append(dict(label=label, m=m, M=M, nu_1=float(nu[0]),
                              simID=params.simID.iloc[i]))
        if verbose:
            print(f"{label}: m={m:.4e}, M={M:.3f}, nu_1={nu[0]:.4f}")

        rng = np.random.default_rng([seed, i])
        n = np.concatenate([[n_A], np.full(K_demes, n_per_deme_B)])
        n_available = np.concatenate(
            [[int(params.n_per_pop_A.iloc[i])], np.full(K_demes, int(params.n_per_deme_B.iloc[i]))])
        sub = subsample(arrays[i], n, None, rng, n_available=n_available)
        pc = eigenvectors_by_overlap(sub, V_theory)

        deme_row = np.concatenate([[np.nan] * n_A,
                                   np.repeat(np.repeat(np.arange(K), K), n_per_deme_B)])
        deme_col = np.concatenate([[np.nan] * n_A,
                                   np.repeat(np.tile(np.arange(K), K), n_per_deme_B)])
        pc_rows.append(pd.DataFrame(dict(
            label=label, m=m, deme_row=deme_row, deme_col=deme_col,
            pc2=pc[:, 0], pc3=pc[:, 1])))

    pc_df = pd.concat(pc_rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    pc_df.to_csv(os.path.join(outdir, "pc.csv"), index=False)
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)
    if verbose:
        print(f"-> {outdir}")
    return pc_df, meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simdir", default=None)
    p.add_argument("--phi", type=float, default=DEFAULT_PHI,
                   help="B's total share of N (default: %(default)s)")
    p.add_argument("--N-total", type=int, default=DEFAULT_N_TOTAL, dest="N_total")
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(simdir=args.simdir, phi=args.phi, N_total=args.N_total, name=args.name,
        seed=args.seed)


if __name__ == "__main__":
    main()

# %%
