#%%
"""
Sweep a sampling parameter across a simulation set and record how well PCA recovers the
theoretical structure axes.

One driver for every demographic scenario: the caller supplies the simulation directory, a
function mapping each simulation's parameters to its theoretical coalescent time matrix,
and the sampling schemes to sweep. Everything else -- subsampling, L_eff, the PCs, the
principal-angle comparison -- is pc_recovery.

Three sweeps are provided, matching the ways the sampling design can vary:

    sweep_N       total sample size, holding the per-subpopulation ratios fixed
    sweep_ratios  the ratios between subpopulations, holding N fixed
    sweep_L       the number of SNPs, holding the sample design fixed

Output is one tidy row per (simulation, cell) in long format, so plotting scripts filter
rather than reshape. Each row carries the measured median_cos next to analytical_cos, the
BBP prediction built from the measured L_eff and the theoretical nu -- no free parameters,
so the two columns can be compared directly.

A second file holds one row per (simulation, cell, replicate): the plotting scripts fit a
transition curve through that whole cloud and report where it crosses a target overlap,
rather than drawing the medians. The summary file is kept because other figures read it.

Outputs (results/analysis/<name>), for a sweep written as sweep.csv:
    sweep.csv              one row per (simulation, cell), the median summary
    sweep_replicates.csv   one row per (simulation, cell, replicate)
    sweep_crossings.csv    one row per cell: the swept value at which the measured overlap
                           crosses the target, beside the value at which theory does
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossing_fit import crossings_table, replicate_rows
from pc_recovery import (load_simulations, pc_recovery, theoretical_eigenvectors,
                         SIMULATION_ROOT)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_ROOT = os.path.join(_PROJECT_ROOT, "results", "analysis")

DEFAULT_NUM_REPLICATES = 20
DEFAULT_SEED = 0


def split_T(Fst):
    """Coalescent time matrix for a symmetric two-population split, in units of 2*Ne."""
    return np.array([[1.0, 1.0 + Fst], [1.0 + Fst, 1.0]])


def run_sweep(simdir, T_from_params, schemes, indices, name, outdir=None, L=None,
              num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, param_col=None,
              verbose=True, filename="sweep.csv", desc=None):
    """
    Measure PC recovery for every (simulation, sampling scheme) pair.

    Args:
        simdir: directory of .npy simulations (see pc_recovery.load_simulations).
        T_from_params: row of the parameter table -> its (K, K) coalescent time matrix.
        schemes: list of per-subpopulation sample size vectors, one per cell of the sweep.
        indices: which structure axes to compare, 0 = largest. Passed to both
            theoretical_eigenvectors and the sample PCs, so they always correspond.
        name: subdirectory of results/analysis to write to.
        L: SNP columns per replicate, or None to keep all polymorphic loci.
        param_col: the swept simulation parameter to record (e.g. "Fst", "m"). Inferred
            from the parameter table when not given.
        filename: output file within outdir, so several sweeps can share a directory.
        desc: label for the progress bar.

    Returns:
        The long-format DataFrame, which is also written to <outdir>/<filename>.
    """
    params, arrays = load_simulations(simdir)
    outdir = os.path.join(ANALYSIS_ROOT, name) if outdir is None else outdir

    if param_col is None:
        candidates = [c for c in ("Fst", "m", "t_split", "t") if c in params.columns]
        param_col = candidates[0] if candidates else None

    rows, rep_rows = [], []
    total = len(arrays) * len(schemes)
    bar = tqdm(total=total, disable=not verbose, desc=desc)
    for i, (_, prow) in enumerate(params.iterrows()):
        T = T_from_params(prow)
        for n in schemes:
            n = np.asarray(n, dtype=int)
            V, nu = theoretical_eigenvectors(T, n, indices=indices)
            res = pc_recovery(arrays[i], n, V, indices=indices, L=L,
                              num_replicates=num_replicates, seed=seed, nu=nu)
            row = dict(simID=prow["simID"], scheme="-".join(map(str, n)),
                       n_min=int(n.min()), n_max=int(n.max()),
                       phi_min=float(n.min() / n.sum()), L_requested=L,
                       nu_min=float(np.min(nu)), nu_max=float(np.max(nu)))
            if param_col is not None:
                row[param_col] = prow[param_col]
            # mu2/delta ride along because the plotting scripts rebuild the theoretical
            # overlap from them (the two-moment route) without reopening the genotypes.
            for key in ("N", "K", "k", "L_raw", "L_eff", "gamma", "gamma_raw", "mu2",
                        "delta", "median_cos", "se_cos",
                        "analytical_cos", "procrustes_d", "se_procrustes",
                        "num_replicates", "seed"):
                row[key] = res[key]
            # Per-axis columns, named by the requested PC index so a sweep over different
            # index sets still lines up column-wise.
            for j, axis in enumerate(np.atleast_1d(indices)):
                row[f"cos_axis{axis}"] = float(res["cos_per_axis"][j])
                row[f"se_cos_axis{axis}"] = float(res["se_cos_per_axis"][j])
                row[f"analytical_cos_axis{axis}"] = float(res["analytical_cos_per_axis"][j])
                row[f"nu_axis{axis}"] = float(nu[j])
            rows.append(row)

            # One row per replicate of this cell, for the curve fit. The per-axis cosines
            # are averaged into the subspace similarity, matching what median_cos reduces.
            base = {k: v for k, v in row.items() if k != param_col}
            rep_rows.extend(replicate_rows(base, prow[param_col] if param_col else np.nan,
                                           param_col or "x", res["cos_replicates"],
                                           seed=seed))
            bar.update(1)
    bar.close()

    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    df.to_csv(out_path, index=False)

    df_rep = pd.DataFrame(rep_rows)
    rep_path = os.path.join(outdir, filename.replace(".csv", "_replicates.csv"))
    df_rep.to_csv(rep_path, index=False)

    # Where each curve crosses the target overlap, measured vs theory, as a table -- so the
    # comparison exists as a number without having to draw the figure. The curve here is
    # one sampling scheme; the swept simulation parameter is the x axis.
    df_cross = crossings_table(df_rep, ["scheme"], param_col, extra_keys=(
        "N", "n_min", "n_max", "phi_min", "L_eff", "L_raw", "gamma", "num_replicates"))
    cross_path = os.path.join(outdir, filename.replace(".csv", "_crossings.csv"))
    df_cross.to_csv(cross_path, index=False)

    if verbose:
        print(f"wrote {len(df)} rows -> {out_path}")
        print(f"wrote {len(df_rep)} replicate rows -> {rep_path}")
        print(f"wrote {len(df_cross)} crossing rows -> {cross_path}")
    return df


def sweep_N(simdir, T_from_params, N_values, indices, name, ratios=(1, 1), **kw):
    """Vary total sample size N, holding the per-subpopulation ratios fixed."""
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()
    schemes = [np.maximum(np.round(ratios * N).astype(int), 1) for N in N_values]
    return run_sweep(simdir, T_from_params, schemes, indices, name, **kw)


def sweep_ratios(simdir, T_from_params, N, ratio_list, indices, name, **kw):
    """Vary the sampling ratios between subpopulations, holding N fixed."""
    schemes = []
    for r in ratio_list:
        r = np.asarray(r, dtype=float)
        n = np.maximum(np.round(r / r.sum() * N).astype(int), 1)
        n[-1] += N - n.sum()          # absorb the rounding residual so sum(n) == N exactly
        schemes.append(n)
    return run_sweep(simdir, T_from_params, schemes, indices, name, **kw)


def sweep_L(simdir, T_from_params, L_values, n, indices, name, **kw):
    """Vary the number of SNPs, holding the sample design fixed."""
    filename = kw.pop("filename", "sweep.csv")
    frames = []
    for L in L_values:
        # Each L writes an intermediate file; the concatenation below is the real output.
        frames.append(run_sweep(simdir, T_from_params, [n], indices, name, L=L,
                                filename=f"_partial_L{L}.csv", **kw))
    df = pd.concat(frames, ignore_index=True)
    outdir = kw.get("outdir") or os.path.join(ANALYSIS_ROOT, name)
    df.to_csv(os.path.join(outdir, filename), index=False)
    return df

# %%
