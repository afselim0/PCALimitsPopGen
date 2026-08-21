"""
Exact theoretical sister eigenvalue from the OOA3 coalescent-time matrix.

nu_1 = N*H*Fst was derived under eps_a = 0 and tau_a/tau_bar = 1. On this demography those
are false -- the measured floor sits at 1/(1+alpha) with alpha ~ 0.10, which is direct
evidence of heterogeneous within-population coalescent times. So N*H*Fst is an
approximation, and testing a measurement against it conflates two questions.

This module supplies the exact reference. Given the K x K population coalescent-time matrix
T and a sample-size vector n, it builds the sample-level G = -H T H / T_bar, diagonalizes
it, and returns the eigenvalue whose eigenvector best matches the predicted sister axis.
Nothing is assumed about eps_a. That splits the test in two:

    nu_meas  vs  nu_exact    does simulation + measurement recover the theory
    nu_exact vs  N*H*Fst     what the eps_a = 0 approximation costs

Coalescent times are expensive (msprime integrates numerically, and the default tolerance
does not converge on this model) but depend only on t_split, and the analysis uses 8
distinct values, so results are cached to disk keyed by t_split.

Outputs: results/analysis/SisterEigenvalue/coal_times.npz (cache)
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POPS = ["CEU", "CHB", "YRI", "YRI2"]
SISTER = (2, 3)
RTOL = 1e-3          # the msprime default fails to converge on this model
MAX_ITER = 20


def _demography(t_split):
    """OOA3 with the YRI2 split, loaded from the simulation kind module by path."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "simulations", "demographies", "african_split.py")
    spec = importlib.util.spec_from_file_location("_ooa3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.demography(t_split)


# Splitting YRI2 off YRI at t_split leaves the CEU/CHB/YRI history untouched, so the six
# pairs among those three are independent of t_split and are integrated once. Only the four
# pairs involving YRI2 move. They are *not* recoverable from their YRI counterparts by
# symmetry: at t_split = 1000, T[YRI2,YRI2] differs from T[YRI,YRI] by 1.9e-3 relative and
# T[CEU,YRI2] from T[CEU,YRI] by 8.0e-3, both growing with t_split. Those are small but
# comparable to the effect under test, so the four are computed exactly rather than
# approximated.
BACKBONE = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def _pair(dbg, pops, i, j):
    lineages = {pops[i]: 2} if i == j else {pops[i]: 1, pops[j]: 1}
    return dbg.mean_coalescence_time(lineages=lineages, rtol=RTOL, max_iter=MAX_ITER)


def pop_coal_times(t_split, pops=POPS, backbone=None):
    """
    K x K mean pairwise coalescent times, including the within-population diagonal.

    `backbone` supplies the six t_split-independent entries so they are not recomputed.
    """
    dbg = _demography(t_split).debug()
    K = len(pops)
    T = np.zeros((K, K))

    for i, j in BACKBONE:
        T[i, j] = T[j, i] = backbone[i, j] if backbone is not None else _pair(dbg, pops, i, j)
    for i in range(K):                       # every pair involving YRI2 (index 3)
        T[i, 3] = T[3, i] = _pair(dbg, pops, i, 3)
    return T


def compute_backbone(pops=POPS, t_split=1000.0):
    """The six CEU/CHB/YRI entries. Any t_split gives the same values."""
    dbg = _demography(t_split).debug()
    B = np.zeros((len(pops), len(pops)))
    for i, j in BACKBONE:
        B[i, j] = B[j, i] = _pair(dbg, pops, i, j)
    return B


def cached_coal_times(t_splits, cachefile, verbose=True):
    """Coalescent times for each t_split, computed once and cached by value."""
    cache = {}
    if os.path.exists(cachefile):
        with np.load(cachefile) as z:
            cache = {k: z[k] for k in z.files}

    missing = [t for t in t_splits if f"{t:.4e}" not in cache]
    if missing:
        if "backbone" not in cache:
            if verbose:
                print("  backbone (6 t_split-independent pairs)", flush=True)
            cache["backbone"] = compute_backbone(pops=POPS)
        backbone = cache["backbone"]
        for i, t in enumerate(missing):
            if verbose:
                print(f"  YRI2 pairs {i+1}/{len(missing)}  t_split={t:.1f}", flush=True)
            cache[f"{t:.4e}"] = pop_coal_times(t, backbone=backbone)

        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        np.savez(cachefile, **cache)
    return {t: cache[f"{t:.4e}"] for t in t_splits}


def expand(T, n):
    """Population-level T to sample level; the diagonal is 0 (a sample coalesces with itself
    at time 0) while two distinct samples from population a coalesce at tau_a."""
    Tn = np.repeat(np.repeat(T, n, axis=0), n, axis=1)
    np.fill_diagonal(Tn, 0.0)
    return Tn


def exact_spectrum(T, n):
    """
    Eigenvalues and eigenvectors of G = -H T_sample H / T_bar, descending.

    T_bar is the mean over distinct sample pairs (the zero diagonal excluded), matching the
    definition used for the sample covariance.
    """
    n = np.asarray(n, dtype=int)
    N = int(n.sum())
    Tn = expand(T, n)
    T_bar = Tn.sum() / (N * (N - 1))

    H = np.eye(N) - np.ones((N, N)) / N
    G = -(H @ Tn @ H) / T_bar
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def exact_sister_nu(T, n, axis):
    """
    The exact sister structure eigenvalue, located the same way as in the measurement.

    G's structure eigenvalues are (1+nu)/(1+alpha) and its floor is 1/(1+alpha), so
    nu = lambda/floor - 1 recovers nu without needing alpha separately -- the same
    normalization the measurement uses, which is what makes the two comparable.
    """
    n = np.asarray(n, dtype=int)
    K = len(n)
    vals, vecs = exact_spectrum(T, n)

    floor = float(vals[K - 1:-1].mean())
    overlaps = np.abs(vecs.T @ axis)
    k = int(np.argmax(overlaps))
    return dict(nu_exact=float(vals[k] / floor - 1.0),
                lam_exact=float(vals[k]), floor_exact=floor,
                alpha_exact=float(1.0 / floor - 1.0),
                overlap_exact=float(overlaps[k]), rank_exact=k)
