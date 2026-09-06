#%%
"""
Recent-split demography (panmictic A + a K x K grid B) and its spatial PC pair, at three
representative migration rates. Figure 12 in CLAUDE.md.

For each of three m (weak / intermediate / strong spatial structure: the smallest and
largest M in the grid for the extremes, plus the row of median theoretical nu of the
leading spatial pair in between), the leading three sample eigenvectors PC1, PC2 and
PC3 are taken by eigenvalue rank -- literally, not located by overlap with the
theoretical spatial axis: PC1 is the A/B split and PC2/PC3 carry the grid's spatial
pair, and the figure shows all three as they fall out of the spectrum. Their overlap
with the theoretical (1,0)/(0,1) spatial pair is still reported in meta.csv as a
diagnostic. They are computed from EVERY sample the simulation wrote -- no subsampling of
rows; only the L SNP columns are drawn. phi is therefore descriptive, not a design
point: it is whatever n_A against K^2 * n_per_deme_B gives (0.509 on the K9 cell, from
n_A=5000 and 81 x 64 = 5184), and the run prints the realised N and phi. An earlier
version subsampled to N_total split phi/(1-phi), which made phi exact at the cost of
using ~10% of the samples; `sample_design` below still implements that and is kept for
callers that want it. Each B sample is labeled by its deme's grid position for the plotting script to
color by.

Outputs (results/analysis/RecentSpatialDemography):
    pc.csv     one row per sample: m_label, m, deme_row, deme_col (NaN for A),
               pc1, pc2, pc3 -- the literal leading three sample eigenvectors by
               eigenvalue rank, not axes located by overlap. PC2/PC3 are rotated onto
               the grid's (col, row) axes by procrustes, so the scatter's orientation
               matches panel A's colors
    meta.csv   one row per m_label: m, M, nu_1 (theoretical leading spatial eigenvalue),
               t_split (split time in generations, used by the schematic panel),
               cos (mean principal cosine between the PC2/PC3 pair and the
               theoretical spatial pair -- the same subspace-overlap convention the
               RecentSpatialOverlapVsM sweep reports)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from scipy.linalg import orthogonal_procrustes

from pc_recovery import (load_simulations, principal_cosines, sample_eigenvectors,
                         subsample, SIMULATION_ROOT)
from split_stepping_stone_theory import equilibrium_B_axes, within_B_F_hat
from sweeps import ANALYSIS_ROOT

DEFAULT_NAME = "RecentSpatialDemography"
M_LABELS = ("weak", "intermediate", "strong")
DEFAULT_PHI = 0.5          # B's total share of N (CLAUDE.md's convention, matches figure 13)
DEFAULT_N_TOTAL = 1000
DEFAULT_L = 10000        # ~11.6k loci still segregate at N=500; 20k were simulated


def spatial_axis(K, m, t_split, Ne_ancestral, n_per_deme_B, n_A):
    """
    Theoretical spatial pair lifted to [A, B_00, ..., B_(K-1)(K-1)] sample space.

    The pair is the equilibrium 2D stepping-stone one; only nu stays on the transient
    phase-type solve. See split_stepping_stone_theory.equilibrium_B_axes.
    """
    nu, _ = within_B_F_hat(K, m, t_split, Ne_ancestral, n_per_deme_B)
    leading, _ = equilibrium_B_axes(K, n_per_deme_B)
    n_B = np.full(K * K, n_per_deme_B)
    lifted_B = np.repeat(leading / np.sqrt(n_B)[:, None], n_B, axis=0)
    return np.vstack([np.zeros((n_A, 2)), lifted_B]), nu[:2]


def grid_axes(K, n_A, n_per_deme_B):
    """
    The (col, row) grid coordinates of every sample, as a 2-column orthonormal basis --
    the procrustes target that orients PC2/PC3 to the schematic's colors.

    spatial_axis()'s V_theory cannot serve as this target: its two columns come out of
    an eigendecomposition (equilibrium_B_axes'), whose order and sign within the
    degenerate pair are arbitrary, so aligning to it leaves the scatter rotated or mirrored relative to
    the grid. Column varies along x and row along y with row increasing upwards, matching
    panel A, which draws row r at y=r -- so row 0 (blue/red) is the BOTTOM edge and row
    K-1 (green/yellow) the top. Both axes are centred so A's zero rows do not bias the fit.
    """
    idx = np.arange(K * K)
    col = np.repeat(idx % K, n_per_deme_B).astype(float)
    row = np.repeat(idx // K, n_per_deme_B).astype(float)
    B = np.column_stack([col - col.mean(), row - row.mean()])
    V = np.vstack([np.zeros((n_A, 2)), B])
    return V / np.linalg.norm(V, axis=0, keepdims=True)


def pick_m_rows(params, nu_1_per_row):
    """Weak/intermediate/strong: the smallest and largest M in the grid for the two
    extremes, and the row with the median theoretical nu_1 in between (nu_1 is largest
    at weak migration, which keeps demes distinct)."""
    order = np.argsort(nu_1_per_row)[::-1]  # descending nu_1 == weak -> strong migration
    M = params.M.to_numpy()
    return {"weak": int(np.argmin(M)), "intermediate": int(order[len(order) // 2]),
           "strong": int(np.argmax(M))}


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
        N_total=DEFAULT_N_TOTAL, L=DEFAULT_L, seed=0, name=DEFAULT_NAME, verbose=True):
    simulation_root = simdir or os.path.join(SIMULATION_ROOT, "split_stepping_stone")
    if os.path.isdir(simulation_root) and any(
            d.startswith("K") for d in os.listdir(simulation_root)):
        cells = sorted(d for d in os.listdir(simulation_root) if d.startswith("K"))
        simulation_root = os.path.join(simulation_root, cells[0])

    params, arrays = load_simulations(simulation_root)
    K = int(params["K"].iloc[0])
    t_split = t_split if t_split is not None else float(params["t_split"].iloc[0])
    Ne_ancestral = Ne_ancestral if Ne_ancestral is not None else float(params["Ne_ancestral"].iloc[0])
    # ALL samples: the sizes the simulation actually wrote. phi and N_total are therefore
    # descriptive here, not design points -- the previous version subsampled to N_total
    # split phi/(1-phi), which is what made phi a controlled quantity. Reported below so
    # the realised split is visible rather than assumed.
    n_A = int(params["n_per_pop_A"].iloc[0])
    n_per_deme_B = int(params["n_per_deme_B"].iloc[0])
    N_realised = n_A + n_per_deme_B * K * K
    phi_realised = n_per_deme_B * K * K / N_realised
    if verbose:
        print(f"all samples: n_A={n_A}, n_per_deme_B={n_per_deme_B} "
             f"(x{K*K} demes = {n_per_deme_B * K * K}), N={N_realised}, "
             f"phi={phi_realised:.4f}")

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
                              t_split=t_split, simID=params.simID.iloc[i]))
        rng = np.random.default_rng([seed, i])
        # EVERY sample in the simulation, not a subsample: n == n_available, so subsample
        # only draws the L SNP columns. phi is then whatever the simulation's own sample
        # sizes give (n_A against K^2 * n_per_deme_B), NOT the phi argument -- see the note
        # on sample_design below.
        n_available = np.concatenate(
            [[int(params.n_per_pop_A.iloc[i])], np.full(K_demes, int(params.n_per_deme_B.iloc[i]))])
        n = n_available
        sub = subsample(arrays[i], n, L, rng, n_available=n_available,
                        L_polymorphic=True)
        pc = sample_eigenvectors(sub, [0, 1, 2])   # literal PC1, PC2, PC3 by eigenvalue
        cos = float(principal_cosines(V_theory, pc[:, 1:]).mean())
        # Rotate the PC2/PC3 pair onto the grid's (col, row) axes. The pair spans a
        # degenerate subspace, so any rotation of it is an equally valid pair of sample
        # eigenvectors -- this picks the one whose orientation matches the schematic,
        # leaving PC1 and the reported cos (rotation-invariant) untouched.
        R = orthogonal_procrustes(pc[:, 1:], grid_axes(K, n_A, n_per_deme_B))[0]
        pc = np.column_stack([pc[:, 0], pc[:, 1:] @ R])
        meta_rows[-1]["cos"] = cos
        if verbose:
            print(f"{label}: m={m:.4e}, M={M:.3f}, nu_1={nu[0]:.4f}, overlap={cos:.3f}")

        deme_row = np.concatenate([[np.nan] * n_A,
                                   np.repeat(np.repeat(np.arange(K), K), n_per_deme_B)])
        deme_col = np.concatenate([[np.nan] * n_A,
                                   np.repeat(np.tile(np.arange(K), K), n_per_deme_B)])
        pc_rows.append(pd.DataFrame(dict(
            label=label, m=m, deme_row=deme_row, deme_col=deme_col,
            pc1=pc[:, 0], pc2=pc[:, 1], pc3=pc[:, 2])))

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
    p.add_argument("--L", type=int, default=DEFAULT_L,
                   help="SNPs per panel, drawn from the polymorphic columns")
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(simdir=args.simdir, phi=args.phi, N_total=args.N_total, L=args.L,
        name=args.name, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
