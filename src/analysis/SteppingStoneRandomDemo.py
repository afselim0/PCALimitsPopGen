#%%
"""
A demonstration of random deme sampling and its effect on the recovered PCs, 1D and 2D.

A demonstration that, at a FIXED migration rate, how densely the demes are sampled decides
whether PCA recovers the spatial structure at all.

One M per dimension (M_TARGETS), chosen to sit in the regime where sparse sampling fails
and dense sampling succeeds, and three sampling fractions phi in [0.2, 0.4, 0.6] at a FIXED
TOTAL N (N_TOTAL: 200 in 1D, 600 in 2D), n-per-deme being derived as N_total / demes
sampled. Migration is held fixed across the whole figure so the only thing varying along a
row is the sampling scheme.

Total N is held fixed rather than n-per-deme so the figure varies deme coverage ALONE. With
n_per_deme = 8 fixed, as previously, N grew with phi (1D: 40 -> 120) and the panels
conflated sampling more demes with sampling more individuals. This matches sweep configs 06
and 08, which hold the same totals.

PCs are sign-fixed (1D) and procrustes-aligned (2D) to the closed-form eigenvectors
oriented so that PC1 increases with chain position / column and PC2 increases with row,
i.e. the scatter colors run the same way as the schematic above them.

For each (dimension, phi): which demes were sampled (for the schematic rows), and the PC1
(1D) / PC1+PC2 (2D) coordinates of the sampled individuals (for the PC rows). A single
fixed random deme draw per (dimension, phi), not a replicate sweep -- this figure
demonstrates the sampling scheme's effect, it does not measure detectability (that's
SteppingStoneRandomSampling.py / figure 10).

Outputs (results/analysis/SteppingStoneRandomDemo):
    sampled_demes.csv   one row per (dimension, phi, deme): whether it was sampled
    pcs.csv             one row per (dimension, phi, sample): deme, PC1 (, PC2), and
                        `cos`, the cosine similarity of that panel's recovered PC(s)
                        with the closed-form axis / axes (constant within a panel)
    meta.csv            one row per dimension: simID, m, M, M_c, K
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import (load_simulations, subsample, sample_eigenvectors,
                         principal_cosines, SIMULATION_ROOT)
import preprocess
from sweeps import ANALYSIS_ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import (closed_form_eigenvector_1d, closed_form_eigenvector_2d,
                                 critical_M, num_demes)

DEFAULT_NAME = "SteppingStoneRandomDemo"
SWEEP_NAME = "SteppingStoneRandomSampling"   # supplies the M grid this demo picks from
PHI_VALUES = [0.2, 0.4, 0.6]
# Total samples, split evenly over whichever demes are drawn; n-per-deme is DERIVED from
# this per phi (see n_per_deme_for). Matches configs 06/08 so the demo and the sweep show
# the same design. 1D: 5/10/15 demes -> 40/20/13 per deme. 2D: 16/32/49 -> 37/18/12.
N_TOTAL = {1: 200, 2: 600}
SEED = 0
# One target M per dimension. The two dimensions are chosen on OPPOSITE criteria, because
# holding total N fixed changes what each one has to show:
#   2D  M = 6.7, the largest separation across phi that survives fixed N -- cos^2 runs
#       0.84 / 0.89 / 0.95, so deme coverage genuinely helps here.
#   1D  M = 790, chosen to be FLAT (the nearest simulated M is 811; sweep medians there run
#       cos^2 0.742 / 0.736 / 0.798, spread 0.062) -- the partial-recovery regime, where a
#       real effect would be plainly visible if there were one. At fixed N, 1D deme coverage has no effect that stands out from draw-to-draw
#       noise: only 8 of 100 simulated M even give a monotone increase in phi (2D: 75 of
#       100), and single draws at the most-separated M scatter from 0.00 to 0.75 at one phi
#       (2 of 12 seeds monotone). A "separated" 1D panel can only be produced by picking a
#       flattering seed, so the panel reports the flat result instead. This is the same M
#       the figure used before, when n-per-deme was fixed at 8 and N grew 40 -> 120 across
#       the row -- the separation it showed then was sample size, not deme coverage.
# Medians over the sweep replicates (results/sweep_summaries/stepping_stone_random).
M_TARGETS = {1: 790.0, 2: 6.7}


def sample_demes(K_demes, phi, rng):
    n_sampled = max(int(round(phi * K_demes)), 1)
    return np.sort(rng.choice(K_demes, n_sampled, replace=False))



def n_per_deme_for(K_demes, phi, N_total):
    """n-per-deme spreading N_total over the demes a fraction phi retains.

    Floor, not round: the cell must never request more samples than were simulated per deme.
    """
    n_demes = max(int(round(phi * K_demes)), 1)
    if preprocess.DIPLOID:              # as models/stepping_stone.fraction_n_per_deme
        return preprocess.nearest_usable(N_total / n_demes)
    return max(int(N_total // n_demes), 1)


def pick_m_row(params, dimension, sweep_name=SWEEP_NAME):
    """The simulated M closest to this dimension's M_TARGETS, as (array index, sweep row).

    The simulation directory holds two sweeps under different naming conventions (an older
    invM* set, which owns the parameters CSV and the .npy files, and the m* set the overlap
    sweeps used, whose L survives only in the sweep table). The two simID spaces are
    DISJOINT, so the sweep row is matched to a simulation by M VALUE, not by simID -- an
    identity match only ever resolved for the handful of targets whose nearest sweep row
    happened to carry an invM id, and raised KeyError for every other target.
    """
    sweep = pd.read_csv(os.path.join(ANALYSIS_ROOT, sweep_name, f"random_{dimension}D.csv"))
    sweep = sweep.drop_duplicates("simID")
    row = sweep.iloc[int(np.argmin(np.abs(sweep.M.values - M_TARGETS[dimension])))]
    i = int(np.argmin(np.abs(params.M.values - row.M)))
    if not np.isclose(params.M.values[i], row.M, rtol=0.02):
        raise KeyError(f"{dimension}D: no simulation within 2% of sweep M={row.M:.1f}")
    return i, row


def demo_1d(Z, K, n_per_deme, phi, L, rng):
    demes = sample_demes(K, phi, rng)
    n = np.zeros(K, dtype=int)
    n[demes] = n_per_deme
    sub = subsample(Z, n, L, rng)
    pc1 = sample_eigenvectors(sub, [0])[:, 0]

    row_mask = np.zeros(K, dtype=bool)
    row_mask[demes] = True
    # cos(pi/K * (j+1/2)) DECREASES in j, so aligning to it as-is leaves PC1 running
    # backwards along the chain and the colors mirrored relative to the schematic.
    u_theory = -closed_form_eigenvector_1d(1, K, n_per_deme)
    u_sub = u_theory[np.repeat(row_mask, n_per_deme)]
    u_sub = u_sub / np.linalg.norm(u_sub)
    pc1 = pc1 if pc1 @ u_sub >= 0 else -pc1
    cos = float(principal_cosines(u_sub[:, None], pc1[:, None])[0])

    pc_df = pd.DataFrame(dict(deme=np.repeat(demes, n_per_deme), pc1=pc1))
    deme_df = pd.DataFrame(dict(deme=np.arange(K), sampled=row_mask))
    return deme_df, pc_df, cos


def demo_2d(Z, K, n_per_deme, phi, L, rng):
    K_demes = num_demes(K, 2)
    demes = sample_demes(K_demes, phi, rng)
    n = np.zeros(K_demes, dtype=int)
    n[demes] = n_per_deme
    sub = subsample(Z, n, L, rng)
    V_s = sample_eigenvectors(sub, [0, 1])

    row_mask = np.zeros(K_demes, dtype=bool)
    row_mask[demes] = True
    sample_mask = np.repeat(row_mask, n_per_deme)
    # Column first, then row: the schematic draws column along x and row along y, so
    # (0,1) -- which varies with the column index -- has to be the axis plotted as PC1,
    # otherwise the scatter comes out as the schematic's transpose. Both are negated
    # because cos(pi/K * (i+1/2)) decreases in i, which would otherwise mirror each axis.
    V_theory = np.column_stack([
        -closed_form_eigenvector_2d(0, 1, K, n_per_deme)[sample_mask],
        -closed_form_eigenvector_2d(1, 0, K, n_per_deme)[sample_mask]])
    # Restricting to a deme subset destroys the orthonormality the closed forms have on
    # the full lattice, so re-orthonormalize before comparing subspaces.
    V_theory = np.linalg.qr(V_theory)[0]
    R = orthogonal_procrustes(V_s, V_theory)[0]
    V_aligned = V_s @ R
    cos = float(principal_cosines(V_theory, V_s).mean())

    deme = np.repeat(demes, n_per_deme)
    pc_df = pd.DataFrame(dict(deme=deme, row=deme // K, col=deme % K,
                              pc1=V_aligned[:, 0], pc2=V_aligned[:, 1]))
    deme_df = pd.DataFrame(dict(deme=np.arange(K_demes), row=np.arange(K_demes) // K,
                                col=np.arange(K_demes) % K, sampled=row_mask))
    return deme_df, pc_df, cos


def run(name=DEFAULT_NAME, n_total=None, phi_values=PHI_VALUES, L=None,
        seed=SEED, verbose=True):
    outdir = os.path.join(ANALYSIS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)

    deme_rows, pc_rows, meta_rows = [], [], []
    for dimension, demo_func in [(1, demo_1d), (2, demo_2d)]:
        simroot = os.path.join(SIMULATION_ROOT, f"stepping_stone_{dimension}d")
        cells = sorted(d for d in os.listdir(simroot) if d.startswith("K"))
        simdir = os.path.join(simroot, cells[0])
        params, arrays = load_simulations(simdir)
        K = int(params["K"].iloc[0])

        i, row = pick_m_row(params, dimension)
        L_cell = L if L is not None else int(row.L)
        N_tot = (n_total or N_TOTAL)[dimension]
        K_demes = num_demes(K, dimension)
        # n-per-deme now varies with phi, so the critical M does too; it is reported per
        # phi rather than once for the dimension.
        if verbose:
            print(f"{dimension}D: simID={row.simID}, M={float(row.M):.0f}, N_total={N_tot}")

        for phi in phi_values:
            n_per_deme = n_per_deme_for(K_demes, phi, N_tot)
            n_demes = max(int(round(phi * K_demes)), 1)
            M_c = critical_M(K, n_per_deme, L_cell, dimension)
            rng = np.random.default_rng([seed, dimension, int(round(phi * 100))])
            deme_df, pc_df, cos = demo_func(arrays[i], K, n_per_deme, phi, L_cell, rng)
            deme_df["dimension"], deme_df["phi"] = dimension, phi
            pc_df["dimension"], pc_df["phi"] = dimension, phi
            pc_df["cos"] = cos
            deme_rows.append(deme_df)
            pc_rows.append(pc_df)
            meta_rows.append(dict(dimension=dimension, phi=phi, simID=row.simID,
                                  m=float(row.m), M=float(row.M), M_c=M_c,
                                  M_over_Mc=float(row.M) / M_c, K=K,
                                  n_per_deme=n_per_deme, n_demes=n_demes,
                                  N=n_demes * n_per_deme, N_total=N_tot))
            if verbose:
                print(f"  phi={phi:g}: n_per_deme={n_per_deme}, N={n_demes * n_per_deme}, "
                      f"M/M_c={float(row.M) / M_c:.3f}, cos={cos:.3f}")

    pd.concat(deme_rows, ignore_index=True).to_csv(
        os.path.join(outdir, "sampled_demes.csv"), index=False)
    preprocess.individuals(pd.concat(pc_rows, ignore_index=True)).to_csv(
        os.path.join(outdir, "pcs.csv"), index=False)
    pd.DataFrame(meta_rows).to_csv(os.path.join(outdir, "meta.csv"), index=False)
    if verbose:
        print(f"-> {outdir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(name=args.name, L=args.L, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
