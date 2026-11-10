#%%
"""
How the spatial signal is distributed across PCs in the recent-split demography, against 1/M.

Feeds panel C of results/figures/PhaseTransitionRecentSS. Panel D shows the theoretical
modes crowding together at weak migration; this asks what that does to the MEASURED PCs on
the simulated genotypes -- specifically whether the spatial signal is conserved and merely
redistributed, or partly lost.

Quantities. With U_inf the orthonormal basis of the equilibrium spatial pair (lifted to
sample space, zero on A's rows) and u_k the observed PCs in EIGENVALUE ORDER, the per-rank
leakage is

    p_k = || U_inf^T u_k ||^2 = <u1_inf, u_k>^2 + <u2_inf, u_k>^2

which is invariant to rotation within the target plane, so nothing is Procrustes-aligned.
From it, per replicate:

    S     = sum_{k=2..k_max} p_k            total retained signal, ceiling 2
    r_eff = S^2 / sum_{k=2..k_max} p_k^2    participation width (PCs carrying the signal)

Three things this measure can get wrong, each guarded explicitly:

  * **Averaging order.** S and r_eff are formed WITHIN each replicate and only then
    reduced across replicates. Building them from replicate-averaged p_k would smear the
    unstable basis choice of a near-degenerate set into a spuriously wide r_eff, and would
    break the S <= 2 ceiling.
  * **Rank order.** PCs are taken in eigenvalue order, never selected by overlap with the
    target. Overlap-selection would search the spectrum and make the measure report what it
    was looking for. (Panel C identifies the theoretical pair by overlap, which is a
    different question -- there the modes are known exactly and only need labelling.)
  * **Window width.** S is recomputed at k_max in {10, 20, 40}; if it keeps growing the
    degenerate set is wider than the window and r_eff is understated. The figure's k_max is
    the smallest value at which S has plateaued, which is reported rather than assumed.

Sums start at k=2 (1-based), i.e. PC1 is excluded: rank 0 is the A-vs-B mode, which carries
no spatial signal. Its p_1 is measured and reported across the sweep rather than asserted.

Noise floor: under no spatial structure E[p_k] ~ 2/(N-1), so S accumulates ~2*k_max/(N-1)
from noise alone. Reported for the chosen k_max so that "S sits just under 2" can be read
against it.

Replicates reuse the sweep's own draws: rng = default_rng([seed, rep]) with seed=0, the
convention in pc_recovery, on the same (phi=0.5, n_B=4, L=1e4) cell as panels A and D.

Outputs (results/analysis/RecentSpatialParticipation/):
    participation.csv   one row per (inv_M, replicate): S and r_eff at each k_max, p_1
    per_rank.csv        one row per (inv_M, k): median p_k over replicates, k = 1..21
    per_rank_replicates.csv  one row per (inv_M, replicate, k): the individual p_k, which
                        the cumulative-leakage panel needs for its windows and its spread
    summary.csv         one row per inv_M: median/IQR/std of S and r_eff at the chosen k_max
    checks.txt          the three checks, the p_k table, and p_1 across the sweep
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # src/analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from pc_recovery import load_simulations, subsample, sample_eigenvectors
from split_stepping_stone_theory import equilibrium_B_axes

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMDIR = os.path.join(_PROJECT_ROOT, "results", "simulations",
                      "split_stepping_stone", "K9")
OUTDIR = os.path.join(_PROJECT_ROOT, "results", "analysis",
                      "RecentSpatialParticipation")

K = 9
K_DEMES = K * K
N_PER_DEME = 4
PHI = 0.5
L = 10000
N_REPLICATES = 8
SEED = 0
K_MAX_VALUES = [10, 20, 40]
FIRST_K = 2          # 1-based: skip PC1, the A-vs-B mode
N_RANKS_REPORT = 21  # per-rank values saved for k = 1..21 (the widest cumulative window)


def theory_pair(n_A):
    """
    U_inf: the equilibrium spatial pair in sample space, zero on A's rows, orthonormal.

    equilibrium_B_axes returns the population-space (1,0)/(0,1) modes; spreading them over
    each deme's samples and zero-padding A gives the sample-space axes the sweep measures
    against. Re-orthonormalized by QR because the lift does not preserve orthonormality
    exactly (see the restricted-axis note in the project memory).
    """
    leading, _ = equilibrium_B_axes(K, N_PER_DEME)
    n_B = np.full(K_DEMES, N_PER_DEME)
    lifted = np.repeat(leading / np.sqrt(n_B)[:, None], n_B, axis=0)
    V = np.vstack([np.zeros((n_A, 2)), lifted])
    Q, _ = np.linalg.qr(V)
    return Q


def leakage(U_inf, V_pcs):
    """p_k for each column of V_pcs: the squared norm of its projection onto U_inf."""
    return np.sum((U_inf.T @ V_pcs) ** 2, axis=0)


def participation(p, k_max, first_k=FIRST_K):
    """
    (S, r_eff) from a per-rank leakage vector, using 1-based ranks [first_k, k_max].

    p is 1-based conceptually; p[0] is PC1. Returns NaN r_eff if S is 0.
    """
    seg = p[first_k - 1:k_max]
    S = float(seg.sum())
    denom = float((seg ** 2).sum())
    return S, (S * S / denom if denom > 0 else np.nan)


def run(simdir=SIMDIR, outdir=OUTDIR, targets=None, verbose=True):
    os.makedirs(outdir, exist_ok=True)
    params, arrays = load_simulations(simdir)

    if targets is None:
        # A spread across the sweep; the full 100-point grid is unnecessary for a smooth
        # curve and each point costs an eigendecomposition per replicate.
        targets = [0.0405, 0.06, 0.09, 0.13, 0.19, 0.28, 0.40, 0.58, 0.85,
                   1.2, 1.8, 2.6, 3.3, 4.05]
    pick = sorted({int(np.argmin(np.abs(params.inv_M.values - t))) for t in targets})

    N_B = N_PER_DEME * K_DEMES
    n_A = int(round(N_B * (1.0 - PHI) / PHI))
    n = np.concatenate([[n_A], np.full(K_DEMES, N_PER_DEME)])
    N = int(n.sum())
    U_inf = theory_pair(n_A)
    n_pcs = max(K_MAX_VALUES)

    rows, rank_rows, rep_rank_rows = [], [], []
    for i in pick:
        row = params.iloc[i]
        n_available = np.concatenate([[int(row.n_per_pop_A)],
                                      np.full(K_DEMES, int(row.n_per_deme_B))])
        per_rank = []
        for rep in range(N_REPLICATES):
            rng = np.random.default_rng([SEED, rep])      # the sweep's own convention
            Z = subsample(arrays[i], n, L=L, rng=rng, n_available=n_available,
                          L_polymorphic=True)
            V_pcs = sample_eigenvectors(Z, list(range(n_pcs)))
            p = leakage(U_inf, V_pcs)
            per_rank.append(p)

            rec = dict(inv_M=float(row.inv_M), M=float(row.M), replicate=rep,
                       p_1=float(p[0]), N=N)
            for km in K_MAX_VALUES:
                S, r_eff = participation(p, km)
                rec[f"S_k{km}"] = S
                rec[f"r_eff_k{km}"] = r_eff
            rows.append(rec)

        per_rank = np.array(per_rank)
        for k in range(1, N_RANKS_REPORT + 1):
            rank_rows.append(dict(inv_M=float(row.inv_M), k=k,
                                  p_k=float(np.median(per_rank[:, k - 1]))))
            for rep in range(N_REPLICATES):
                rep_rank_rows.append(dict(inv_M=float(row.inv_M), replicate=rep, k=k,
                                          p_k=float(per_rank[rep, k - 1])))
        if verbose:
            print(f"  1/M={row.inv_M:.4g} done", flush=True)

    part = pd.DataFrame(rows)
    per_rank_df = pd.DataFrame(rank_rows)
    per_rank_rep = pd.DataFrame(rep_rank_rows)

    lines = []
    k_max = _checks(part, per_rank_df, N, lines)

    summary = _summarise(part, k_max)
    part.to_csv(os.path.join(outdir, "participation.csv"), index=False)
    per_rank_df.to_csv(os.path.join(outdir, "per_rank.csv"), index=False)
    per_rank_rep.to_csv(os.path.join(outdir, "per_rank_replicates.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    with open(os.path.join(outdir, "checks.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if verbose:
        print("\n".join(lines))
        print(f"\nwrote {outdir}")
    return part, per_rank_df, summary


def _summarise(part, k_max):
    """Median, IQR and std of S and r_eff per 1/M, at the chosen k_max."""
    S, R = f"S_k{k_max}", f"r_eff_k{k_max}"
    g = part.groupby("inv_M")
    out = pd.DataFrame({
        "inv_M": g.inv_M.first(),
        "M": g.M.first(),
        "S_med": g[S].median(),
        "S_lo": g[S].quantile(0.25), "S_hi": g[S].quantile(0.75),
        "S_std": g[S].std(),
        "r_eff_med": g[R].median(),
        "r_eff_lo": g[R].quantile(0.25), "r_eff_hi": g[R].quantile(0.75),
        "r_eff_std": g[R].std(),
        "p_1_med": g.p_1.median(),
        "k_max": k_max,
    }).reset_index(drop=True)
    return out


def _checks(part, per_rank, N, lines):
    """The three guards, printed. Returns the k_max chosen for the figure."""
    fmt = lambda v: f"{v:.4g}"
    lines.append(f"cell: phi={PHI}, n_B={N_PER_DEME}, N={N}, L={L}, "
                 f"{N_REPLICATES} replicates, PC window starts at k={FIRST_K}")
    lines.append("")

    # --- 1. cap sensitivity -------------------------------------------------
    lines.append("=== check 1: cap sensitivity (median S and r_eff by k_max) ===")
    med = part.groupby("inv_M")[[f"S_k{k}" for k in K_MAX_VALUES]
                                + [f"r_eff_k{k}" for k in K_MAX_VALUES]].median()
    lines.append(med.to_string(float_format=fmt))

    tail = med.iloc[-1]
    grow = [(k, float(tail[f"S_k{k}"])) for k in K_MAX_VALUES]
    lines.append("")
    lines.append("  at the weak-migration tail, S by k_max: " +
                 ", ".join(f"k={k}: {v:.4f}" for k, v in grow))
    # Plateau = the smallest k_max whose S is within 2% of the largest window's.
    biggest = grow[-1][1]
    k_max = K_MAX_VALUES[-1]
    for k, v in grow:
        if biggest > 0 and abs(v - biggest) / biggest <= 0.02:
            k_max = k
            break
    lines.append(f"  -> S plateaus by k_max={k_max} (within 2% of k_max="
                 f"{K_MAX_VALUES[-1]}); figure uses k_max={k_max}")
    if k_max == K_MAX_VALUES[-1] and abs(grow[0][1] - biggest) / max(biggest, 1e-12) > 0.02:
        lines.append("  WARNING: S still growing at the largest window -- the degenerate "
                     "set is wider than the window and r_eff is UNDERSTATED.")

    # --- 2. noise floor -----------------------------------------------------
    floor_per_rank = 2.0 / (N - 1)
    floor_S = 2.0 * k_max / (N - 1)
    lines.append("")
    lines.append("=== check 2: noise floor ===")
    lines.append(f"  E[p_k] under no structure ~ 2/(N-1) = {floor_per_rank:.5f}")
    lines.append(f"  S picks up ~2*k_max/(N-1) = {floor_S:.4f} from noise at k_max={k_max}")
    lines.append(f"  (S ceiling is 2, so the floor is {100*floor_S/2:.1f}% of full signal)")

    # --- 3. replicate spread ------------------------------------------------
    S, R = f"S_k{k_max}", f"r_eff_k{k_max}"
    med_S = part.groupby("inv_M")[S].median()
    peak_x = float(med_S.idxmax())
    tail_x = float(part.inv_M.max())
    lines.append("")
    lines.append("=== check 3: replicate spread ===")
    for label, x in (("peak", peak_x), ("tail", tail_x)):
        d = part[part.inv_M == x]
        lines.append(f"  {label} (1/M={x:.4g}): "
                     f"S {d[S].median():.4f} +- {d[S].std():.4f}, "
                     f"r_eff {d[R].median():.3f} +- {d[R].std():.3f} "
                     f"[min {d[R].min():.3f}, max {d[R].max():.3f}]")

    # --- p_k table and p_1 --------------------------------------------------
    lines.append("")
    lines.append("=== per-rank median p_k (k = 1..8) at three representative 1/M ===")
    xs = sorted(per_rank.inv_M.unique())
    show = [xs[0], xs[len(xs) // 2], xs[-1]]
    tab = per_rank[per_rank.inv_M.isin(show) & (per_rank.k <= 8)]
    lines.append(tab.pivot(index="inv_M", columns="k", values="p_k")
                 .to_string(float_format=fmt))

    lines.append("")
    lines.append("=== p_1 (the A-vs-B mode) across the sweep ===")
    p1 = part.groupby("inv_M").p_1.median()
    lines.append("  " + "  ".join(f"{x:.3g}:{v:.3f}" for x, v in p1.items()))
    lines.append(f"  range [{p1.min():.4f}, {p1.max():.4f}] -- expected small (~0.03-0.12)")

    # --- the headline -------------------------------------------------------
    lines.append("")
    lines.append("=== result ===")
    d_peak = part[part.inv_M == peak_x]
    d_tail = part[part.inv_M == tail_x]
    lines.append(f"  S     {d_peak[S].median():.4f} -> {d_tail[S].median():.4f} "
                 f"(ceiling 2; change {d_tail[S].median()-d_peak[S].median():+.4f})")
    lines.append(f"  r_eff {d_peak[R].median():.3f} -> {d_tail[R].median():.3f} "
                 f"(change {d_tail[R].median()-d_peak[R].median():+.3f})")
    drop = (d_peak[S].median() - d_tail[S].median()) / max(d_peak[S].median(), 1e-12)
    if drop > 0.05:
        lines.append(f"  NOTE: S falls by {100*drop:.1f}% -- the signal is NOT purely "
                     f"redistributed; part of it is genuinely lost.")
    else:
        lines.append(f"  S is conserved to within {100*abs(drop):.1f}% -- consistent with "
                     f"pure redistribution across PCs.")
    return k_max


if __name__ == "__main__":
    run()

# %%
