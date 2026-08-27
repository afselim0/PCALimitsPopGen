"""
Population-level mean coalescent times for split_stepping_stone (panmictic A + a K x K
grid B of demes, split from a common ancestor ANC at t_split), computed by solving the
transient two-lineage structured coalescent directly -- not via msprime's
DemographyDebugger.mean_coalescence_time, which is exact but far too slow to scale here
(measured ~14s/pair at K=3's 10 populations with default tolerance, and a single pair at
K=9's 82 populations did not return within 115s even at a loosened tolerance; K=9 needs
3321 pairs).

Model. Going backward in time, for 0 <= t < t_split two lineages move under B's
migration process (rate m between grid neighbours) and coalesce within a shared deme at
rate 1/(2*Ne_deme); A has no migration and its own lineages coalesce at rate
1/(2*Ne_ancestral). At t_split every surviving lineage (A and B alike) merges into ANC
(size Ne_ancestral), after which the residual expected coalescence time is exactly
2*Ne_ancestral (memoryless). So

    E[T] = E[T * 1(T < t_split)] + P(T >= t_split) * (t_split + 2*Ne_ancestral)

which is a standard continuous-time phase-type calculation given the sub-generator Q
over unordered pairs of lineage locations (a state space of size K^2(K^2+1)/2 for B-B
pairs -- 3321 states at K=9). Q is extremely sparse (each state has at most 8 migration
neighbours plus its own coalescence rate, ~0.24% dense at K=9), so it is built and solved
with scipy.sparse: expm_multiply for the survival vector and spsolve for the phase-type
mean, rather than dense expm/solve. Dense was tried first and did not return within
several minutes at K=9; sparse takes ~3s.

A-A and A-B pairs. Two lineages both in A can still coalesce before t_split (A is an
ordinary panmictic population), so A-A goes through the same phase-type formula as a
1-state chain. A-B pairs can never coalesce before t_split (A has no migration into B,
so the two lineages can never share a deme), which makes their sub-block of Q singular
(no true absorption) and destabilizes the general matrix formula; but exactly because
coalescence before t_split is impossible, T[A, B_pq] = t_split + 2*Ne_ancestral is exact
in closed form and does not depend on (p, q) -- A cannot distinguish B's demes.

Validated against exact msprime.DemographyDebugger.mean_coalescence_time at K=2 (5
populations, tractable in seconds even at msprime's default tolerance) across
t_split in {10, 50, 200} and m in {0.005, 0.05}: every entry (within-deme, adjacent-deme,
diagonal-deme, A-B, A-A) agrees with msprime to a consistent ~0.15-0.16% relative error,
the signature of msprime's discrete-generation Wright-Fisher coalescent vs. this module's
continuous-time approximation (confirmed on a single-population toy case: a Ne=2500
population's E[T] from msprime is 4996.3, not exactly 2*Ne=5000, the same offset). This
is not a modeling error -- CLAUDE.md's own F_hat/resistance-distance theory throughout is
built on the continuous-time coalescent, and B-B's *relative* structure (which is what
determines eigenvectors) matches essentially exactly.
"""
import os
import sys
from functools import lru_cache
from itertools import product

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, spsolve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from utils import compute_F_hat

A_LOC = "A"


def _neighbors(p, q, K):
    for dp, dq in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        np_, nq_ = p + dp, q + dq
        if 0 <= np_ < K and 0 <= nq_ < K:
            yield np_, nq_


def deme_locs(K):
    """Grid cell locations (p, q), row-major, matching split_stepping_stone's B_<row>_<col>."""
    return list(product(range(K), repeat=2))


def _within_B_generator(K, m, Ne_deme):
    """
    Sparse sub-generator over unordered pairs of B-grid locations: migration between
    grid neighbours at rate m, coalescence within a shared deme at rate 1/(2*Ne_deme).

    Returns (locs, states, idx, Q) with Q a scipy.sparse.csr_matrix.
    """
    locs = deme_locs(K)
    states = [(a, b) for i, a in enumerate(locs) for b in locs[i:]]
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)

    rows, cols, vals = [], [], []
    coal_rate = np.zeros(n)
    off_rate = np.zeros(n)
    for i, (a, b) in enumerate(states):
        if a == b:
            coal_rate[i] = 1.0 / (2.0 * Ne_deme)
        for pos, other in ((a, b), (b, a)):
            for nb in _neighbors(*pos, K):
                j = idx[tuple(sorted([nb, other]))]
                rows.append(i)
                cols.append(j)
                vals.append(m)
                off_rate[i] += m
    Q = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    Q = Q + sp.diags(-(off_rate + coal_rate))
    return locs, states, idx, Q


def _phase_type_mean(Q, t_split, Ne_ancestral):
    """
    E[T] for every state of a sparse sub-generator Q, per the decomposition in the module
    docstring: E[T] = E[T*1(T<t)] + P(T>=t)*(t + 2*Ne_ancestral).

    E[min(T,t)] = -Q^{-1}(I - expm(Qt)) 1 is the standard phase-type formula; subtracting
    t*P(T>=t) isolates E[T*1(T<t)]. Computed via expm_multiply (expm(Qt)@v without
    forming the dense matrix exponential) and a single sparse solve, rather than dense
    expm/inv -- dense did not return within minutes at K=9's 3321 states; sparse takes
    seconds, since Q has only O(1) nonzeros per row.
    """
    n = Q.shape[0]
    ones = np.ones(n)
    expQt_ones = expm_multiply(Q * t_split, ones)  # = expm(Qt) @ 1 = row sums = survival
    survival = expQt_ones
    E_min = spsolve(-Q, ones - expQt_ones)
    E_before = E_min - t_split * survival
    return E_before + survival * (t_split + 2.0 * Ne_ancestral)


def population_coal_times(K, m, t_split, Ne_ancestral):
    """
    The (1+K^2) x (1+K^2) population-level mean coalescent time matrix T for
    split_stepping_stone(K, Ne_ancestral, t_split, m), population order
    [A, (0,0), (0,1), ..., (K-1,K-1)] (row-major, matching split_stepping_stone.py's
    B_<row>_<col> naming).

    A-A and every A-B entry are the closed form t_split + 2*Ne_ancestral (A cannot
    coalesce with anything before t_split: no migration links it to B, and two A
    lineages that haven't coalesced by t_split simply carry on in ANC). B-B entries
    (including the within-deme diagonal) come from _phase_type_mean on B's own
    migration+coalescence generator.
    """
    Ne_deme = Ne_ancestral / (K * K)
    locs, states, idx, Q = _within_B_generator(K, m, Ne_deme)
    E_T_B = _phase_type_mean(Q, t_split, Ne_ancestral)

    K_demes = K * K
    T = np.full((1 + K_demes, 1 + K_demes), t_split + 2.0 * Ne_ancestral)
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            T[1 + i, 1 + j] = E_T_B[idx[tuple(sorted([a, b]))]]
    return T


def sample_order(K):
    """Population names in T's row/column order: 'A' then row-major B_<row>_<col>."""
    return [A_LOC] + [f"B_{p}_{q}" for p, q in deme_locs(K)]


@lru_cache(maxsize=None)
def _T_B_matrix(K, m, t_split, Ne_ancestral):
    """
    B's K^2 x K^2 mean coalescent time matrix -- the sparse phase-type solve, which
    depends only on (K, m, t_split, Ne_ancestral), not on how B is sampled. Cached because
    callers sweeping a sampling design (n_per_deme_B, phi, ...) at fixed demography
    parameters would otherwise re-solve the same generator once per sweep cell; the solve
    itself (~seconds at K=9's 3321 states, per the module docstring) dominates
    within_B_F_hat's cost, while the eigendecomposition it feeds is cheap.
    """
    Ne_deme = Ne_ancestral / (K * K)
    locs, states, idx, Q = _within_B_generator(K, m, Ne_deme)
    E_T_B = _phase_type_mean(Q, t_split, Ne_ancestral)

    K_demes = K * K
    T_B = np.empty((K_demes, K_demes))
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            T_B[i, j] = E_T_B[idx[tuple(sorted([a, b]))]]
    return T_B


def within_B_F_hat(K, m, t_split, Ne_ancestral, n_per_deme_B):
    """
    F_hat restricted to B's K^2 demes alone (even sampling n_per_deme_B), the structure
    that carries the spatial axes figures 12/13 target -- A does not enter, since its
    coupling to every B deme is identical and therefore washes out of F_hat's
    P_hat-centred structure among B's demes.

    Returns (nu, V): eigenvalues and eigenvectors of compute_F_hat(T_B, n), descending,
    with the trivial mean-direction eigenvalue already dropped (K^2 - 1 remain).
    """
    T_B = _T_B_matrix(K, m, t_split, Ne_ancestral)
    K_demes = K * K

    n = np.full(K_demes, n_per_deme_B)
    F_hat = compute_F_hat(T_B, n)
    w, V = np.linalg.eigh(F_hat)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    psi = np.sqrt(n / n.sum())
    trivial = int(np.argmax(np.abs(psi @ V)))
    keep = [j for j in range(K_demes) if j != trivial]
    return w[keep], V[:, keep]


@lru_cache(maxsize=None)
def equilibrium_B_axes(K, n_per_deme_B):
    """
    B's leading spatial pair from the EQUILIBRIUM 2D stepping-stone model, in
    population space -- the (1,0)/(0,1) cosine modes of the K x K grid.

    This is the axis the overlap is measured against, in place of within_B_F_hat's
    transient pair. The equilibrium grid's eigenvectors do not depend on m (m enters
    stepping_stone_T only as an overall 1/m scale on F_2, which cancels out of F_hat's
    eigenvectors), nor on t_split, so the target is one fixed pair for the whole M
    sweep rather than a per-M solve.

    Using it does not throw away the recent split: the transient and equilibrium pairs
    span nearly the same subspace at K=9 (principal cosines 0.9999 at the strong-migration
    end, falling only to 0.979 at the weakest M simulated), so the measured overlap is
    barely affected. What differs sharply is the *eigenvalue*: the equilibrium closed form
    has no t_split ceiling on how much spatial structure can accumulate, so it runs away at
    small M (nu = 134 against the transient 10.3 at M = 0.247). Eigenvalues therefore stay
    on within_B_F_hat's phase-type solve; only the eigenvectors come from here.

    Cached: the result depends only on (K, n_per_deme_B), so a sweep over M reuses one
    eigendecomposition rather than rebuilding T and re-solving per point. Under even
    sampling the eigenvectors do not actually move with n_per_deme_B either (checked: cos
    = 1 between n = 4 and n = 17); it is kept as an argument because it sets nu_eq's scale
    and keeps the signature parallel to within_B_F_hat.

    Returns (V, nu_eq): the (K^2, 2) population-space pair and its equilibrium
    eigenvalues, the latter for reference only. Treat both as read-only, as the cache
    hands back the same arrays to every caller.
    """
    # Imported here rather than at module scope: stepping_stone_detection pulls in
    # pc_recovery and tqdm, which this module otherwise does not need.
    from stepping_stone_detection import stepping_stone_T

    n = np.full(K * K, n_per_deme_B)
    # m and Ne only scale T here; the eigenvectors are invariant to both.
    T_eq = stepping_stone_T(K, 1.0, 1.0, 2)
    F_hat = compute_F_hat(T_eq, n)
    w, V = np.linalg.eigh(F_hat)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    psi = np.sqrt(n / n.sum())
    trivial = int(np.argmax(np.abs(psi @ V)))
    keep = [j for j in range(K * K) if j != trivial]
    return V[:, keep][:, :2], w[keep][:2]
