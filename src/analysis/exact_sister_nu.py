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

POPS = ["CEU", "CHB", "YRI1", "YRI2"]
SISTER = (2, 3)
RTOL = 1e-3          # the msprime default fails to converge on this model
MAX_ITER = 20


def _demography(t_split):
    """OOA3 with the symmetric YRI1/YRI2 split, loaded from the simulation kind module by
    path."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "simulations", "demographies", "african_split.py")
    spec = importlib.util.spec_from_file_location("_ooa3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.demography(t_split)


# YRI1 and YRI2 are both new populations descending from the unsampled ancestral YRI, so
# every pair -- not just those touching YRI2 -- depends on t_split: a YRI1 or YRI2 lineage
# spends time waiting to reach t_split before it behaves like the old single-YRI backbone,
# and that wait is itself t_split-dependent. Nothing here is cacheable across t_split.
# Exact symmetry between the two siblings still halves the work: T[CEU,YRI1]=T[CEU,YRI2],
# T[CHB,YRI1]=T[CHB,YRI2], T[YRI1,YRI1]=T[YRI2,YRI2], so only 7 of the 10 pairs are unique.
UNIQUE_PAIRS = [(0, 0), (1, 1), (0, 1), (2, 2), (0, 2), (1, 2), (2, 3)]
MIRROR = {(0, 3): (0, 2), (1, 3): (1, 2), (3, 3): (2, 2)}


def _pair(dbg, pops, i, j):
    lineages = {pops[i]: 2} if i == j else {pops[i]: 1, pops[j]: 1}
    return dbg.mean_coalescence_time(lineages=lineages, rtol=RTOL, max_iter=MAX_ITER)


def pop_coal_times(t_split, pops=POPS):
    """K x K mean pairwise coalescent times, including the within-population diagonal."""
    dbg = _demography(t_split).debug()
    K = len(pops)
    T = np.zeros((K, K))

    for i, j in UNIQUE_PAIRS:
        T[i, j] = T[j, i] = _pair(dbg, pops, i, j)
    for (i, j), (si, sj) in MIRROR.items():
        T[i, j] = T[j, i] = T[si, sj]
    return T


def cached_coal_times(t_splits, cachefile, verbose=True):
    """Coalescent times for each t_split, computed once and cached by value."""
    cache = {}
    if os.path.exists(cachefile):
        with np.load(cachefile) as z:
            cache = {k: z[k] for k in z.files}

    missing = [t for t in t_splits if f"{t:.4e}" not in cache]
    if missing:
        for i, t in enumerate(missing):
            if verbose:
                print(f"  coal times {i+1}/{len(missing)}  t_split={t:.1f}", flush=True)
            cache[f"{t:.4e}"] = pop_coal_times(t)

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


def analytical_nu(T, n, Fst):
    """
    The closed-form sister eigenvalue nu = eps_0 + (2*N*phi_1*phi_2/s)*(tau/tau_bar)*Fst
    (CLAUDE.md's OOA3 section), computed directly from the population-level coalescent-time
    matrix T -- no numerical diagonalization.

    This is exact for the sister pair's own within-population asymmetry (tau/tau_bar is not
    assumed to be 1, unlike SisterEigenvalue.predicted_nu's N*H*Fst), but still an
    approximation relative to exact_sister_nu: it does not capture the sister pair's
    coupling to the other populations' structure, which only the full F_hat/G
    diagonalization sees.

    Args:
        T: (K, K) population-level mean coalescent times, diag = tau.
        n: per-population sample sizes, length K. SISTER indexes the sister pair.
        Fst: theoretical Fst(sister_1, sister_2).
    """
    n = np.asarray(n, dtype=float)
    N = n.sum()
    phi = n / N
    a, b = SISTER

    tau = np.diag(T)
    tau_bar = float(phi @ tau)
    tau_sister = tau[a]  # tau[a] == tau[b] by construction (YRI1/YRI2 symmetry)
    eps_0 = tau_sister / tau_bar - 1.0

    s = phi[a] + phi[b]
    return float(eps_0 + (2.0 * N * phi[a] * phi[b] / s) * (tau_sister / tau_bar) * Fst)


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
