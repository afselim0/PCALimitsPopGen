#%%
"""
Cosine similarity against migration rate, along the two axes that separate K from N.

Two sweeps per dimension, each holding one of K and N fixed while the other varies:

    vary_K_fixed_N   K varies, N held (as close as possible to) constant
    vary_N_fixed_K   N varies via the per-deme sample size, K held constant

Separating them matters because M_c ~ sqrt(N) * K^(3/2) in 1D and sqrt(N) * K in 2D, so
a sweep that lets both move confounds two different dependences. Holding one fixed is
what makes each panel a test of a single exponent.

Even sampling ties them together: N = n * K_demes, so "constant N while K varies" forces
n = N / K_demes, which must be a whole number no larger than the simulated n_per_deme.
select_fixed_N_cells() picks the N admitting the most K values under those constraints
and reports what it had to drop. In 2D the divisibility is severe -- N must be a common
multiple of several K^2 -- and no N gives four usable K within reach of the simulated
sample size, so the selector falls back to nearest-n, which holds N approximately rather
than exactly constant. The realized N is recorded per row either way, so a panel never
claims a constancy it does not have.

The N sweep costs no new simulations: cost scales with the number of demes, not samples
per deme, so the matrices carry n_per_deme = 10 and every smaller n is a subsample.

Outputs (results/analysis/stepping_stone_sweeps):
    sweeps_<dim>D.csv   one row per (simulation, sweep, cell), long format
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/analysis/
from pc_recovery import load_simulations, pc_recovery, theoretical_eigenvectors, SIMULATION_ROOT
from stepping_stone_detection import leading_multiplet, stepping_stone_T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "simulations"))  # src/simulations/
from stepping_stone_grid import critical_M, critical_M_approx, num_demes

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_ROOT = os.path.join(_PROJECT_ROOT, "results", "analysis")

DEFAULT_NUM_REPLICATES = 20
DEFAULT_SEED = 0
DEFAULT_N_VALUES = [2, 4, 6, 10]
MIN_N_PER_DEME = 2


def discover_cells(simulation_root, K_values=None):
    """
    The K* simulation directories under a root, as {K: path}.

    Cells written by an L sweep carry a _L<L> suffix; those are kept out here so the two
    sweeps compare a single L, which is the axis they are not varying.
    """
    if not os.path.isdir(simulation_root):
        raise FileNotFoundError(f"no such simulation root: {simulation_root}")
    cells = {}
    for name in sorted(os.listdir(simulation_root)):
        path = os.path.join(simulation_root, name)
        if not (name.startswith("K") and os.path.isdir(path)) or "_L" in name:
            continue
        try:
            K = int(name[1:])
        except ValueError:
            continue
        if K_values is None or K in K_values:
            cells[K] = path
    if not cells:
        raise FileNotFoundError(f"no K* subdirectories in {simulation_root}")
    return cells


def select_fixed_N_cells(K_values, dimension, n_available, exact=True,
                         min_cells=4, N_target=None):
    """
    Per-K sample sizes holding N as close to constant as the geometry allows.

    Exact mode requires n = N / K_demes to be a whole number in [MIN_N_PER_DEME,
    n_available], and returns the N satisfying that for the most K. When no N reaches
    min_cells -- the usual outcome in 2D, where N must divide several K^2 at once -- the
    caller can fall back to nearest-n, which keeps every K at the cost of N varying a
    little.

    Args:
        K_values: candidate K.
        dimension: 1 or 2.
        n_available: n_per_deme actually simulated, the upper bound on any subsample.
        exact: require exactly constant N; False selects by nearest n.
        min_cells: how many K an exact solution must cover to be preferred.
        N_target: aim for this N. In exact mode it is a tie-break among equally large
            solutions; in fallback mode it is the N that nearest-n rounds towards.

    Returns:
        (chosen, dropped, mode): {K: n}, the K left out with why, and "exact" or
        "nearest".
    """
    demes = {K: num_demes(K, dimension) for K in K_values}

    if exact:
        candidate_N = sorted({n * demes[K] for K in K_values
                              for n in range(MIN_N_PER_DEME, n_available + 1)})
        best = None
        for N in candidate_N:
            fit = {K: N // demes[K] for K in K_values
                   if N % demes[K] == 0 and MIN_N_PER_DEME <= N // demes[K] <= n_available}
            if not fit:
                continue
            # Prefer covering more K; among equally wide solutions prefer the larger N,
            # which leaves more samples per deme and so a less noisy covariance estimate.
            key = (len(fit), -abs(N - N_target) if N_target else N)
            if best is None or key > best[0]:
                best = (key, N, fit)
        if best is not None and len(best[2]) >= min_cells:
            _, N, fit = best
            dropped = {K: f"N={N} not divisible by {demes[K]} demes, or n out of range"
                       for K in K_values if K not in fit}
            return fit, dropped, "exact"

    # Fallback: nearest n to a target N, which keeps every K but lets N drift. The
    # target is chosen to minimise that drift rather than guessed: n is clamped into
    # [MIN_N_PER_DEME, n_available], so a target that is too large saturates the big
    # grids and a target that is too small saturates the little ones, and either way the
    # realised N spread widens. Scanning candidate targets picks the flattest.
    def realise(target):
        return {K: max(MIN_N_PER_DEME, min(n_available, int(round(target / demes[K]))))
                for K in K_values}

    def spread(fit):
        Ns = [n * demes[K] for K, n in fit.items()]
        return max(Ns) / min(Ns)

    if N_target is not None:
        chosen = realise(N_target)
    else:
        candidates = sorted({n * demes[K] for K in K_values
                             for n in range(MIN_N_PER_DEME, n_available + 1)})
        chosen = min((realise(t) for t in candidates), key=spread)
    return chosen, {}, "nearest"


def _measure(Z, K, dimension, n_per, row, L, num_replicates, seed):
    """One (simulation, n) cell: theoretical axes, PC recovery, and the threshold."""
    K_demes = num_demes(K, dimension)
    n = np.full(K_demes, n_per, dtype=int)
    indices = leading_multiplet(K, dimension)

    T = stepping_stone_T(K, row.m, row.Ne, dimension)
    V_theory, nu = theoretical_eigenvectors(T, n, indices=indices)
    res = pc_recovery(Z, n, V_theory, indices, L=L, num_replicates=num_replicates,
                      seed=seed, nu=nu)

    M = 4.0 * row.Ne * row.m
    M_c = critical_M(K, n_per, res["L_eff"], dimension)
    return dict(
        simID=row.simID, dimension=dimension, K=K, K_demes=K_demes,
        m=row.m, Ne=row.Ne, M=M, n_per_deme=n_per, N=int(n.sum()),
        L=L if L is not None else int(row.L),
        L_eff=res["L_eff"], gamma=res["gamma"],
        M_c=M_c, M_c_approx=critical_M_approx(K, n_per, res["L_eff"], dimension),
        M_over_Mc=M / M_c if np.isfinite(M_c) else np.nan,
        nu_1=float(nu[0]), multiplicity=len(indices),
        median_cos=res["median_cos"], se_cos=res["se_cos"],
        analytical_cos=res["analytical_cos"], procrustes_d=res["procrustes_d"],
        num_replicates=num_replicates)


def vary_K_fixed_N(cells, dimension, n_available, exact=True, N_target=None,
                   L=None, num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED,
                   verbose=True):
    """Recovery vs M for each K, at (as near as possible) a single N."""
    chosen, dropped, mode = select_fixed_N_cells(
        sorted(cells), dimension, n_available, exact=exact, N_target=N_target)
    if verbose:
        detail = ", ".join(f"K={K}: n={n} (N={n * num_demes(K, dimension)})"
                           for K, n in sorted(chosen.items()))
        print(f"[vary_K_fixed_N, {dimension}D, {mode}] {detail}")
        for K, why in dropped.items():
            print(f"   dropped K={K}: {why}")

    rows = []
    for K, n_per in tqdm(sorted(chosen.items()), disable=not verbose, desc="vary K"):
        params, arrays = load_simulations(cells[K])
        for r in params.itertuples():
            rows.append(dict(_measure(arrays[r.Index], K, dimension, n_per, r, L,
                                      num_replicates, seed),
                             sweep="vary_K_fixed_N", constancy=mode))
    return pd.DataFrame(rows)


def vary_N_fixed_K(cells, dimension, K, n_values=DEFAULT_N_VALUES, L=None,
                   num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED,
                   verbose=True):
    """Recovery vs M at a single K, subsampling to each n in n_values."""
    if K not in cells:
        raise KeyError(f"K={K} not among the simulated cells {sorted(cells)}")
    params, arrays = load_simulations(cells[K])
    n_available = int(params["n_per_deme"].iloc[0])
    usable = [n for n in n_values if MIN_N_PER_DEME <= n <= n_available]
    if verbose:
        skipped = [n for n in n_values if n not in usable]
        print(f"[vary_N_fixed_K, {dimension}D] K={K}, n={usable}"
              + (f"; skipped {skipped} (only n<={n_available} simulated)" if skipped else ""))

    rows = []
    for n_per in tqdm(usable, disable=not verbose, desc="vary N"):
        for r in params.itertuples():
            rows.append(dict(_measure(arrays[r.Index], K, dimension, n_per, r, L,
                                      num_replicates, seed),
                             sweep="vary_N_fixed_K", constancy="exact"))
    return pd.DataFrame(rows)


def run(dimension, simulation_root=None, K_values=None, K_fixed=None,
        n_values=DEFAULT_N_VALUES, exact_N=True, N_target=None, L=None,
        num_replicates=DEFAULT_NUM_REPLICATES, seed=DEFAULT_SEED, outdir=None,
        verbose=True):
    """Both sweeps for one dimension, concatenated into a single long-format table."""
    simulation_root = simulation_root or os.path.join(SIMULATION_ROOT, f"SS_{dimension}D")
    cells = discover_cells(simulation_root, K_values)

    probe, _ = load_simulations(next(iter(cells.values())), mmap=True)
    n_available = int(probe["n_per_deme"].iloc[0])

    # Default the fixed-K panel to the largest K available, where the asymptotics the
    # closed forms rely on are most nearly met.
    K_fixed = K_fixed if K_fixed is not None else max(cells)

    frames = [
        vary_K_fixed_N(cells, dimension, n_available, exact=exact_N, N_target=N_target,
                       L=L, num_replicates=num_replicates, seed=seed, verbose=verbose),
        vary_N_fixed_K(cells, dimension, K_fixed, n_values=n_values, L=L,
                       num_replicates=num_replicates, seed=seed, verbose=verbose),
    ]
    out = pd.concat(frames, ignore_index=True)

    outdir = outdir or os.path.join(ANALYSIS_ROOT, "stepping_stone_sweeps")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"sweeps_{dimension}D.csv")
    out.to_csv(path, index=False)
    if verbose:
        print(f"wrote {len(out)} rows -> {path}")
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dimension", type=int, choices=(1, 2), default=1, help="1D chain or 2D grid (default: 1)")
    p.add_argument("--simulation-root", default=None, help="directory of K* cells (default: results/simulations/SS_<dim>D)")
    p.add_argument("--K", type=int, nargs="+", default=None, dest="K_values", help="restrict the vary-K panel to these K (default: every cell found)")
    p.add_argument("--K-fixed", type=int, default=None, help="K for the vary-N panel (default: the largest available)")
    p.add_argument("--n", type=int, nargs="+", default=DEFAULT_N_VALUES, dest="n_values", help=f"per-deme sample sizes for the vary-N panel (default: {DEFAULT_N_VALUES})")
    p.add_argument("--N-target", type=int, default=None, help="N to hold constant in the vary-K panel (default: chosen automatically)")
    p.add_argument("--approx-N", action="store_false", dest="exact_N", help="allow N to vary slightly so every K is kept, instead of requiring exactly constant N")
    p.add_argument("--L", type=int, default=None, help="SNP columns per replicate; default keeps each simulation's full L")
    p.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    run(args.dimension, simulation_root=args.simulation_root, K_values=args.K_values,
        K_fixed=args.K_fixed, n_values=args.n_values, exact_N=args.exact_N,
        N_target=args.N_target, L=args.L, num_replicates=args.num_replicates,
        seed=args.seed, outdir=args.outdir)


if __name__ == "__main__":
    main()

# %%
