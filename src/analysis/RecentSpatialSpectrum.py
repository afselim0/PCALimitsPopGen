#%%
"""
Theoretical structural spectrum of F_hat for the recent-split demography (panmictic A +
a K x K grid B split off t_split generations ago), as a function of 1/M.

Feeds panels C and D of results/figures/PhaseTransitionRecentSS. Panels A and B of that
figure measure how well PCA recovers B's spatial axis; these panels show the spectrum that
recovery is reading from, so the reader can see the gap between the spatial pair and the
next modes close at the same 1/M where the cosine-similarity curves turn over.

Why a new phase-type routine rather than split_stepping_stone_theory's. That module solves
the B-B block only, which is all its callers need (F_hat among B's demes, where A washes
out of the P_hat-centring). Here the full (K^2 + 1)-population T is needed: A enters
tau_bar, eps and hence F_hat once A is part of the sample, and phi = 0.5 means A carries
half of it. So the state space is extended to A-B and A-A pairs, and the two routines are
cross-checked against each other on the B-B block (validation 5 below).

Model. Backward in time, for 0 <= s < t_split two lineages move on B's grid (symmetric
nearest-neighbour rate m per edge, 4-neighbour, non-periodic) and coalesce within a shared
deme at rate 1/(2 Ne_deme); A's lineages do not move and coalesce at rate 1/(2 Ne_A); an
A lineage and a B lineage can never share a deme, so they cannot coalesce before the
split. At t_split everything merges into ANC, after which the residual expected
coalescence time is exactly 2 * Ne_ancestral (memoryless). Hence

    T_s = E[tau; tau < t] + P(tau > t) * (t + 2 Ne_anc) = E[min(tau, t)] + 2 Ne_anc P(tau > t)

the second form because E[tau; tau<t] = E[min(tau,t)] - t P(tau>t), which cancels the t in
the second term. Writing it as E[min] + P(tau>t)(t + 2 Ne_anc) double-counts the pre-split
time and is wrong by t P(tau>t); the A-B block, which cannot coalesce before the split and
must come out at exactly t + 2 Ne_anc, catches that immediately.

E[min(tau,t)] = int_0^t P(tau > s) ds. The survival curve for ALL start states at
once is surv(s) = exp(Q s) 1 -- the ROW sums, indexed by start state, NOT exp(Q^T s) 1,
which is a column sum over end states. Q is not symmetric (coalescence enters the diagonal
only, and the reflecting grid gives edge and corner demes degree 3 and 2 against the
interior's 4), so the two differ; the A-B block cannot detect the confusion because it has
no coalescence and both routes return 1 there. Obtained on a uniform grid in a single
expm_multiply call and integrated by Simpson's rule -- the quadrature route, which unlike the closed form
-Q^{-1}(I - e^{Qt})1 needs no sparse solve and stays well conditioned when the A-B block
is included (that block has no absorption at all, so Q is singular on it and the closed
form is not usable here).

Diploid convention throughout, matching msprime: pairwise coalescence rate 1/(2N) per
generation, expected pairwise coalescence time 2N in a panmictic population of N diploids.

Outputs (results/analysis/RecentSpatialSpectrum/):
    spectrum.csv     one row per (M, phi, n_B, mode): nu_k, its equilibrium counterpart,
                     the identified spatial-pair / A-vs-B indices, the relative gap, nu_c
    gaps.csv         one row per (M, phi, n_B): the relative gap, the crossing summary, and
                     pair_p2 -- how much of the equilibrium spatial plane the NON-EQUILIBRIUM
                     theoretical pair spans, summed squared projection on a 0-2 scale. That
                     is the ceiling panel C's measured p_r curves approach: it is a property
                     of the theoretical eigenvectors alone, with no finite-L estimation in
                     it, so the distance from 2 is structure the split has rotated out of
                     the equilibrium plane and no amount of data can recover.
    censoring.csv    per-mode nu_k / nu_k^inf against x_k = 2 m l_k t, the wavelength check
    validation.txt   the printed validation block
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, spsolve
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from utils import compute_F_hat
from threshold_and_overlap import spectrum_psi

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIM_CONFIG = os.path.join(_PROJECT_ROOT, "src", "simulations", "configs",
                          "split_stepping_stone.yaml")
OUTDIR = os.path.join(_PROJECT_ROOT, "results", "analysis", "RecentSpatialSpectrum")

N_STEPS = 400          # quadrature intervals over [0, t_split]; convergence is checked
N_MODES = 8            # structural modes recorded per cell
A_INDEX = -1           # A is the last row/column of T_pop


# --- demography ------------------------------------------------------------

def load_config(path=SIM_CONFIG):
    """
    (K, Ne_ancestral, t_split, Ne_deme, Ne_A) from the simulation config.

    Ne_deme and Ne_A are not written in the config; they are set by
    simulations/demographies/split_stepping_stone.demography, which gives A
    initial_size=Ne_ancestral and each deme Ne_ancestral / K^2. Read from the config where
    stated and derived exactly as the simulation derives them otherwise, so the two can
    never drift apart.
    """
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    f = cfg["fixed"]
    K, Ne_anc = int(f["K"]), float(f["Ne_ancestral"])
    return dict(K=K, Ne_ancestral=Ne_anc, t_split=float(f["t_split"]),
                Ne_deme=Ne_anc / (K * K), Ne_A=Ne_anc,
                inv_M_min=float(cfg["sweep"]["inv_M_min"]),
                inv_M_max=float(cfg["sweep"]["inv_M_max"]))


def m_from_M(M, Ne_deme):
    """M = 4 Ne_deme * mbar with mbar = m, matching the simulation's parameter table."""
    return M / (4.0 * Ne_deme)


def deme_locs(K):
    """Grid cells (row, col), row-major -- the simulation's B_<row>_<col> order."""
    return [(p, q) for p in range(K) for q in range(K)]


def _neighbors(p, q, K):
    for dp, dq in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        a, b = p + dp, q + dq
        if 0 <= a < K and 0 <= b < K:
            yield a, b


# --- the pair-of-lineages chain -------------------------------------------

def build_states(K):
    """
    All unordered pair states, in one indexable list.

    Order: B-B pairs {a,b} with a <= b, then A-B pairs {a, A}, then {A, A}.
    Returns (states, index) with each state a tuple; A is the string 'A'.
    """
    locs = deme_locs(K)
    states = [(a, b) for i, a in enumerate(locs) for b in locs[i:]]
    states += [(a, "A") for a in locs]
    states += [("A", "A")]
    return states, {s: i for i, s in enumerate(states)}


def _key(x, y):
    """Canonical unordered-pair key, keeping A last."""
    if x == "A":
        return (y, "A") if y != "A" else ("A", "A")
    if y == "A":
        return (x, "A")
    return tuple(sorted([x, y]))


def build_generator(K, m, Ne_deme, Ne_A):
    """
    Sparse sub-generator Q over the pair states of the structured phase.

    Migration moves ONE lineage at a time between grid neighbours at rate m (A's lineage
    never moves); coalescence removes the pair, so it enters only the diagonal. The
    diagonal is minus the total outflow, migration plus coalescence.
    """
    states, idx = build_states(K)
    n = len(states)
    rows, cols, vals = [], [], []
    coal = np.zeros(n)
    out = np.zeros(n)

    for i, (a, b) in enumerate(states):
        if a == "A" and b == "A":
            coal[i] = 1.0 / (2.0 * Ne_A)
            continue
        if b == "A":                      # A-B: only the B lineage moves, no coalescence
            for nb in _neighbors(*a, K):
                rows.append(i); cols.append(idx[_key(nb, "A")]); vals.append(m)
                out[i] += m
            continue
        if a == b:                        # both lineages in the same deme
            coal[i] = 1.0 / (2.0 * Ne_deme)
        for pos, other in ((a, b), (b, a)):
            for nb in _neighbors(*pos, K):
                rows.append(i); cols.append(idx[_key(nb, other)]); vals.append(m)
                out[i] += m

    Q = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    Q = (Q + sp.diags(-(out + coal))).tocsc()
    return states, idx, Q, coal, out


def _check_generator(Q, coal, out, tol=1e-9):
    """Row sums of Q equal -(coalescence rate): migration conserves probability."""
    rs = np.asarray(Q.sum(axis=1)).ravel()
    return float(np.max(np.abs(rs + coal)))


def expected_coal_times(K, m, t_split, Ne_deme, Ne_A, Ne_anc, n_steps=N_STEPS):
    """
    E[T] for every pair state, by quadrature on the survival curve.

    surv(s) = exp(Q^T s) 1 is evaluated on a uniform grid over [0, t_split] in ONE
    expm_multiply call, then E[min(tau,t)] = int_0^t surv ds by Simpson's rule. Returns
    (states, idx, E_T, surv_t, resid) where resid is the generator sanity check.
    """
    states, idx, Q, coal, out = build_generator(K, m, Ne_deme, Ne_A)
    resid = _check_generator(Q, coal, out)

    ones = np.ones(Q.shape[0])
    # exp(Q s) @ 1 -- the ROW sums, i.e. P(tau > s) indexed by START state. NOT Q.T:
    # transposing gives a column sum (the distribution over END states), a different
    # quantity, and Q is genuinely non-symmetric here because coalescence sits on the
    # diagonal and the reflecting grid gives edge and corner demes unequal degrees. The
    # A-B block cannot catch this -- it has no coalescence, so both routes return 1 there.
    surv = expm_multiply(Q, ones, start=0.0, stop=t_split,
                         num=n_steps + 1, endpoint=True)
    s_grid = np.linspace(0.0, t_split, n_steps + 1)
    E_min = simpson(surv, x=s_grid, axis=0)
    surv_t = surv[-1]

    # E[T] = E[T; T<t] + P(T>t)(t + 2 Ne_anc), and E[T; T<t] = E[min(T,t)] - t P(T>t),
    # so the two t P(T>t) terms cancel and E[T] = E[min(T,t)] + 2 Ne_anc P(T>t). Writing
    # it as E[min] + P(t + 2 Ne_anc) would DOUBLE-COUNT the pre-split time for surviving
    # pairs: an A-B pair, which cannot coalesce before the split, has surv == 1 and must
    # come out at exactly t + 2 Ne_anc.
    return states, idx, E_min + 2.0 * Ne_anc * surv_t, surv_t, resid


def T_pop_matrix(K, m, t_split, Ne_deme, Ne_A, Ne_anc, n_steps=N_STEPS):
    """
    The (K^2 + 1) x (K^2 + 1) population-level T, order [B_0_0, ..., B_K-1_K-1, A].

    Diagonal entries are the within-population mean coalescent times tau_a.
    """
    states, idx, E_T, surv_t, resid = expected_coal_times(
        K, m, t_split, Ne_deme, Ne_A, Ne_anc, n_steps)
    locs = deme_locs(K)
    n_pop = K * K + 1
    T = np.empty((n_pop, n_pop))
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            T[i, j] = E_T[idx[_key(a, b)]]
        T[i, -1] = T[-1, i] = E_T[idx[_key(a, "A")]]
    T[-1, -1] = E_T[idx[("A", "A")]]
    return T, resid


def T_equilibrium_B(K, m, Ne_deme, periodic=False):
    """
    B's equilibrium (no split) within-B coalescent time matrix, from the B-only chain.

    With no split the B-B sub-generator is genuinely absorbing, so E[T] = -Q_B^{-T} 1 is
    a single sparse solve rather than a quadrature.

    periodic=True wraps the grid onto a torus. The simulations are non-periodic, so that
    is only for validation: a reflecting grid has unequal degrees at its edges and corners
    and so is NOT conservative, which is exactly the hypothesis Strobeck's uniform-tau
    result needs.
    """
    locs = deme_locs(K)
    states = [(a, b) for i, a in enumerate(locs) for b in locs[i:]]
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    rows, cols, vals = [], [], []
    coal = np.zeros(n)
    out = np.zeros(n)
    nbrs = (lambda p, q: [((p + d) % K, q) for d in (-1, 1)]
                         + [(p, (q + d) % K) for d in (-1, 1)]) if periodic \
        else (lambda p, q: list(_neighbors(p, q, K)))
    for i, (a, b) in enumerate(states):
        if a == b:
            coal[i] = 1.0 / (2.0 * Ne_deme)
        for pos, other in ((a, b), (b, a)):
            for nb in nbrs(*pos):
                rows.append(i); cols.append(idx[tuple(sorted([nb, other]))]); vals.append(m)
                out[i] += m
    Q = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    Q = (Q + sp.diags(-(out + coal))).tocsc()
    # Q E[T] = -1 indexed by START state, so no transpose (see expected_coal_times).
    E_T = spsolve(Q, -np.ones(n))

    T = np.empty((K * K, K * K))
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            T[i, j] = E_T[idx[tuple(sorted([a, b]))]]
    return T


# --- lattice reference quantities -----------------------------------------

def grid_laplacian(K):
    """Unscaled Laplacian of the K x K reflecting grid, row-major."""
    n = K * K
    A = np.zeros((n, n))
    for p in range(K):
        for q in range(K):
            i = p * K + q
            if q + 1 < K:
                A[i, i + 1] = A[i + 1, i] = 1.0
            if p + 1 < K:
                A[i, i + K] = A[i + K, i] = 1.0
    return np.diag(A.sum(1)) - A


def laplacian_eigs(K):
    """(eigenvalues ascending, eigenvectors) of the unscaled grid Laplacian."""
    w, V = np.linalg.eigh(grid_laplacian(K))
    return w, V


def dct_pair(K):
    """
    The (1,0)/(0,1) cosine modes of the K x K grid, as (K^2, 2), orthonormal.

    u_kl(p,q) = cos(k pi (p+1/2)/K) cos(l pi (q+1/2)/K) -- CLAUDE.md's closed form.
    """
    p = np.arange(K)
    c1 = np.cos(np.pi * (p + 0.5) / K)
    o = np.ones(K)
    u10 = np.outer(c1, o).ravel()
    u01 = np.outer(o, c1).ravel()
    V = np.column_stack([u10, u01])
    return V / np.linalg.norm(V, axis=0)


def resistance_distance(K):
    """R_ij from the unscaled grid Laplacian's pseudo-inverse."""
    Lp = np.linalg.pinv(grid_laplacian(K))
    d = np.diag(Lp)
    return d[:, None] + d[None, :] - 2 * Lp


# --- spectrum of F_hat -----------------------------------------------------

def _sample_sizes(K, n_B, phi):
    """n over [B demes..., A]; A's size solved from phi = B's share of the total."""
    K_demes = K * K
    N_B = n_B * K_demes
    n_A = int(round(N_B * (1.0 - phi) / phi)) if phi < 1 else 0
    return np.concatenate([np.full(K_demes, float(n_B)), [float(n_A)]]), n_A


def structural_spectrum(T, n):
    """
    (nu, V) of F_hat with the trivial psi-direction dropped, descending.

    The trivial mode is identified by its overlap with psi rather than by position, as
    elsewhere in the repo.
    """
    F = compute_F_hat(T, n)
    w, V = np.linalg.eigh(F)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    psi = np.sqrt(n / n.sum())
    trivial = int(np.argmax(np.abs(psi @ V)))
    keep = [j for j in range(len(w)) if j != trivial]
    return w[keep], V[:, keep]


def identify_modes(V, K, has_A):
    """
    (spatial_pair_indices, mean principal cosine, summed squared projection, A_vs_B_index).

    The spatial pair is the 2D subspace of the top modes with the largest mean principal
    cosine against the equilibrium DCT pair (padded with 0 on A) -- searched over adjacent
    index pairs among the leading modes rather than assumed to be (0,1). The A-vs-B mode
    is the leading mode whose B-block is closest to constant, i.e. that contrasts A
    against B as a whole; None when A carries no samples.
    """
    K_demes = K * K
    target = dct_pair(K)
    if has_A:
        target = np.vstack([target, np.zeros((1, 2))])
        target = target / np.linalg.norm(target, axis=0)

    n_search = min(V.shape[1], 10)
    best, best_c, best_p2 = (0, 1), -1.0, np.nan
    for i in range(n_search):
        for j in range(i + 1, n_search):
            s = np.linalg.svd(target.T @ V[:, [i, j]], compute_uv=False)
            c = np.clip(s, 0, 1).mean()
            if c > best_c:
                best_c, best = c, (i, j)
                # p_2: the squared projection of the THEORETICAL pair onto the equilibrium
                # spatial plane, summed over the two axes -- the same 0-2 scale as panel
                # C's measured p_r, and its ceiling. best_c is a MEAN of cosines, a
                # different normalisation, so it cannot be reused here.
                best_p2 = float((np.clip(s, 0, 1) ** 2).sum())

    a_idx = None
    if has_A:
        # A-vs-B: large |A component|, B block nearly constant.
        score = [abs(V[-1, k]) for k in range(n_search)]
        a_idx = int(np.argmax(score))
    return best, best_c, best_p2, a_idx


def cell_spectrum(T, T_eq, K, n_B, phi, L_eff):
    """
    Everything recorded for one (M, phi, n_B) cell.

    The transient spectrum uses the FULL sample (B demes + A); the equilibrium reference
    is B-only, since the equilibrium model has no A -- stated in the figure caption.
    nu_c is the spectrum-based finite-K criterion (CLAUDE.md: Phi_2 does not apply when
    K ~ N), reported as the Psi of the spatial pair; Psi < 1 means detectable.
    """
    n, n_A = _sample_sizes(K, n_B, phi)
    nu, V = structural_spectrum(T, n)
    (i, j), pair_cos, pair_p2, a_idx = identify_modes(V, K, has_A=n_A > 0)

    nu_spatial = 0.5 * (nu[i] + nu[j])
    rest = [k for k in range(len(nu)) if k not in (i, j)
            and (a_idx is None or k != a_idx)]
    nu_next = nu[rest[0]] if rest else np.nan
    gap = (nu_spatial - nu_next) / nu_spatial

    n_eq = np.full(K * K, float(n_B))
    nu_eq, _ = structural_spectrum(T_eq, n_eq)

    N = int(n.sum())
    spectrum = np.concatenate([1.0 + nu, np.ones(max(N - len(nu), 1))])
    psi_pair = spectrum_psi(i, spectrum, L_eff, degenerate_subspace=True)
    nu_c = _nu_c_spectrum(nu, i, j, N, L_eff)

    return dict(nu=nu, nu_eq=nu_eq, pair=(i, j), pair_cos=pair_cos,
                pair_p2=pair_p2, a_idx=a_idx,
                nu_spatial=nu_spatial, nu_next=nu_next, gap=gap, psi_pair=psi_pair,
                nu_c=nu_c, N=N, n_A=n_A)


def _nu_c_spectrum(nu, i, j, N, L_eff, lo=1e-3, hi=1e4, n_scan=2000, tol=1e-8):
    """
    The nu the spatial pair would need for Psi_spec = 1, holding the rest of the spectrum
    fixed -- the detection threshold on this demography, as an actual level in nu.

    Psi is not a level in nu, so the threshold is found by moving the pair's own eigenvalue
    and re-evaluating Psi against the OTHER eigenvalues, which stay put.

    Psi(v) - 1 is NOT monotone and generally has SEVERAL roots: spectrum_psi sums
    ((1+nu_j)/(v-nu_j))^2 over the other eigenvalues, so it has a pole at every one of them
    and dips back below 1 in the gaps between neighbouring modes. A plain bisection on
    [lo, hi] therefore returns whichever root its initial bracket happened to contain, which
    is what made nu_c a sawtooth across a smooth M grid (1.28 -> 3.23 -> 1.11 at adjacent
    grid points) while nu_spatial and nu_next stayed smooth.

    Nor is the largest root right: Psi also has a pole at the A-vs-B mode, which on this
    demography sits ABOVE the spatial pair (nu ~ 8.2 against the pair's ~4.9), so the
    topmost crossing is the level at which the pair would separate from the A MODE -- a
    different question, and it lands above the pair's actual value, implying the pair is
    never detectable when direct evaluation gives Psi = 0.04 there.

    The threshold wanted is the bottom edge of the contiguous detectable window the pair
    ACTUALLY occupies: start at the pair's own nu (where Psi < 1), walk DOWN, and take the
    first crossing back up through 1. Below it the pair merges with the modes beneath it,
    which is the mechanism this figure is about.

    Returns nan if the pair's own position is not detectable (Psi >= 1 there), which is
    itself meaningful -- it is what happens at phi = 0.1 at the strong-migration end.
    """
    others = np.array([nu[k] for k in range(len(nu)) if k not in (i, j)])
    pad = np.ones(max(N - len(nu), 1))
    nu_pair = 0.5 * (nu[i] + nu[j])

    def psi_at(v):
        spec = np.concatenate([[1.0 + v, 1.0 + v], 1.0 + others, pad])
        return spectrum_psi(0, spec, L_eff, degenerate_subspace=True) - 1.0

    if not np.isfinite(psi_at(nu_pair)) or psi_at(nu_pair) >= 0:
        return np.nan

    # Walk down from the pair's own position to the first point where Psi >= 1 again.
    grid = np.geomspace(lo, nu_pair, n_scan)
    vals = np.array([psi_at(v) for v in grid])
    ok = np.isfinite(vals)
    k = None
    for t in range(len(grid) - 1, 0, -1):
        if ok[t] and ok[t - 1] and vals[t] < 0 <= vals[t - 1]:
            k = t
            break
    if k is None:
        return np.nan

    a, b = grid[k - 1], grid[k]
    fa = vals[k - 1]
    for _ in range(200):
        mid = np.sqrt(a * b)
        fm = psi_at(mid)
        if not np.isfinite(fm):
            break
        if fa * fm <= 0:
            b = mid
        else:
            a, fa = mid, fm
        if b / a < 1.0 + tol:
            break
    return np.sqrt(a * b)


# --- validation ------------------------------------------------------------

def validate(cfg, lines):
    """The four checks the plan requires, plus a cross-check against the existing module."""
    K, t = cfg["K"], cfg["t_split"]
    Ne_d, Ne_A, Ne_anc = cfg["Ne_deme"], cfg["Ne_A"], cfg["Ne_ancestral"]

    lines.append("=== validation ===")

    # 1. m -> 0: an isolated deme of size Ne_deme, closed form.
    #
    # For an exponential(c) coalescent censored at t, E[tau; tau<t] = (1-e)/c - t e with
    # e = exp(-c t), so E[T] = (1-e)/c - t e + e (t + 2 Ne_anc) = (1-e)/c + 2 Ne_anc e,
    # the t e terms cancelling exactly as in expected_coal_times. With c = 1/(2 Ne_deme)
    # that is 2 Ne_deme (1-e) + 2 Ne_anc e (checked against quadrature: 2854.199372).
    m0 = 1e-12
    T0, resid0 = T_pop_matrix(K, m0, t, Ne_d, Ne_A, Ne_anc)
    e = np.exp(-t / (2.0 * Ne_d))
    expect = 2.0 * Ne_d * (1.0 - e) + 2.0 * Ne_anc * e
    got = T0[0, 0]
    lines.append(f"1. m->0 within-deme tau: got {got:.9f}, expected {expect:.9f}, "
                 f"rel err {abs(got - expect) / expect:.3e}")
    # A-B pairs cannot coalesce before the split, so this one is exact by construction.
    lines.append(f"   A-B (exact t + 2 Ne_anc = {t + 2 * Ne_anc:.1f}): got {T0[0, -1]:.9f}, "
                 f"rel err {abs(T0[0, -1] - (t + 2 * Ne_anc)) / (t + 2 * Ne_anc):.3e}")
    eA = np.exp(-t / (2.0 * Ne_A))
    expect_A = 2.0 * Ne_A * (1.0 - eA) + 2.0 * Ne_anc * eA
    lines.append(f"   A-A: got {T0[-1, -1]:.9f}, expected {expect_A:.9f}, "
                 f"rel err {abs(T0[-1, -1] - expect_A) / expect_A:.3e}")
    lines.append(f"   generator row-sum residual: {resid0:.3e}")

    # 2. Strobeck: under CONSERVATIVE migration every within-deme tau equals
    # 2 Ne_deme K^2 = 2 Ne_total and does not depend on m.
    #
    # The REFLECTING grid the simulations use is not conservative -- edge and corner demes
    # have degree 3 and 2 against the interior's 4 -- so Strobeck's hypothesis fails there
    # and tau genuinely varies between demes. The theorem is therefore checked on a torus,
    # where it must hold exactly, and the reflecting grid's spread is reported separately
    # as the size of the boundary effect rather than as an error.
    m_eq = m_from_M(1.0, Ne_d)
    D = K * K
    strobeck = 2.0 * Ne_d * D
    tau_tor = np.diag(T_equilibrium_B(K, m_eq, Ne_d, periodic=True))
    lines.append(f"2. Strobeck tau = 2 Ne_deme K^2 = {strobeck:.4f} (conservative case):")
    lines.append(f"   torus      tau in [{tau_tor.min():.4f}, {tau_tor.max():.4f}], "
                 f"max rel dev {np.max(np.abs(tau_tor - strobeck)) / strobeck:.3e}")
    T_eq = T_equilibrium_B(K, m_eq, Ne_d)
    tau_eq = np.diag(T_eq)
    lines.append(f"   reflecting tau in [{tau_eq.min():.4f}, {tau_eq.max():.4f}], "
                 f"spread {(tau_eq.max() - tau_eq.min()) / tau_eq.mean():.3e} "
                 f"(boundary effect, not an error)")

    # 3. F_2 vs (K^2/4) R / m, the resistance-distance approximation.
    R = resistance_distance(K)
    for M in (10.0, 1.0, 0.25):
        mm = m_from_M(M, Ne_d)
        Te = T_equilibrium_B(K, mm, Ne_d)
        tau = np.diag(Te)
        F2 = Te - (tau[:, None] + tau[None, :]) / 2.0
        F2_approx = (K * K / 4.0) * R / mm
        off = ~np.eye(K * K, dtype=bool)
        rel = np.abs(F2[off] - F2_approx[off]) / np.abs(F2[off])
        lines.append(f"3. M={M:>5g}: F_2 vs (K^2/4)R/m  median rel err {np.median(rel):.4f}, "
                     f"max {rel.max():.4f}")

    # 5. Cross-check the B-B block against the existing B-only module.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from split_stepping_stone_theory import _T_B_matrix
        mm = m_from_M(1.0, Ne_d)
        T_new, _ = T_pop_matrix(K, mm, t, Ne_d, Ne_A, Ne_anc)
        T_old = _T_B_matrix(K, mm, t, Ne_anc)
        rel = np.abs(T_new[:K * K, :K * K] - T_old) / T_old
        lines.append(f"5. B-B block vs split_stepping_stone_theory._T_B_matrix at M=1: "
                     f"max rel diff {rel.max():.3e}")
    except Exception as exc:                                  # pragma: no cover
        lines.append(f"5. cross-check skipped: {exc}")

    # Quadrature convergence.
    mm = m_from_M(1.0, Ne_d)
    Ta, _ = T_pop_matrix(K, mm, t, Ne_d, Ne_A, Ne_anc, n_steps=N_STEPS)
    Tb, _ = T_pop_matrix(K, mm, t, Ne_d, Ne_A, Ne_anc, n_steps=2 * N_STEPS)
    lines.append(f"6. quadrature: max rel change on doubling n_steps "
                 f"{np.max(np.abs(Ta - Tb) / Tb):.3e} (target < 1e-4)")
    return T_eq


def validate_msprime(cfg, lines, M=1.0, K=3):
    """
    Check 4: a handful of entries against msprime's exact mean_coalescence_time.

    Run at K=3 (10 populations), NOT the sweep's K=9. mean_coalescence_time is a dense
    solve over the whole pair state space and is hopeless at K=9's 82 populations -- the
    same limit split_stepping_stone_theory's docstring records (a single pair did not
    return within 115s even at a loosened tolerance), and a K=9 attempt here produced no
    output in half an hour. What is being validated is the CONVENTIONS -- diploid rates,
    the backward migration direction, the split time, the ancestral residual -- and none of
    them depends on K, so a small grid tests them just as well and finishes in seconds.

    ~0.15% disagreement is expected: msprime's discrete-generation Wright-Fisher coalescent
    against this module's continuous-time approximation, the offset documented in
    split_stepping_stone_theory.
    """
    try:
        import msprime
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "simulations", "demographies"))
        from split_stepping_stone import demography as build_dem
    except Exception as exc:                                  # pragma: no cover
        lines.append(f"4. msprime check skipped: {exc}")
        return

    # Ne_deme follows the simulation's own rule, Ne_ancestral / K^2, at this K.
    Ne_anc = cfg["Ne_ancestral"]
    Ne_deme = Ne_anc / (K * K)
    m = m_from_M(M, Ne_deme)
    T, _ = T_pop_matrix(K, m, cfg["t_split"], Ne_deme, cfg["Ne_A"], Ne_anc)
    dbg = build_dem(K, Ne_anc, cfg["t_split"], m).debug()

    pairs = [(("B_0_0", "B_0_0"), (0, 0)),
             (("B_0_0", "B_0_1"), (0, 1)),
             (("B_0_0", "B_2_2"), (0, 2 * K + 2)),
             (("B_0_0", "A"), (0, K * K)),
             (("A", "A"), (K * K, K * K))]
    lines.append(f"4. msprime mean_coalescence_time at M={M:g}, K={K} "
                 f"(conventions check; K-independent):")
    for (names, (i, j)) in pairs:
        try:
            lineages = {names[0]: 2} if names[0] == names[1] else {names[0]: 1, names[1]: 1}
            exact = dbg.mean_coalescence_time(lineages)
        except Exception as exc:
            lines.append(f"   {names[0]:>8}-{names[1]:<8} failed: {exc}")
            continue
        got = T[i, j]
        lines.append(f"   {names[0]:>8}-{names[1]:<8} phase-type {got:12.3f}  "
                     f"msprime {exact:12.3f}  rel {abs(got - exact) / exact:.4%}")


# --- censoring check -------------------------------------------------------

def censoring_table(cfg, inv_M_values, n_B=4):
    """
    nu_k / nu_k^inf against x_k = 2 m l_k t, for every mode and every M.

    Both spectra are B-only here so the modes correspond one-to-one; l_k are the unscaled
    grid-Laplacian eigenvalues, ascending with the trivial zero dropped, matched to the
    descending nu. The pairing is checked by the near-equilibrium end of the sweep, where
    every ratio must be 1: at 1/M = 0.0405 the largest |ratio - 1| over all 80 modes is
    2.4e-3, including across the degenerate pairs.

    The ratio is NOT bounded above by 1. Both spectra are normalised by their own tau_bar,
    which differs between the transient and equilibrium systems, so the short-wavelength
    modes can sit slightly above their equilibrium counterparts (up to ~1.04 at the weak-
    migration end) even while the long-wavelength ones are censored far below. The
    wavelength ordering, which is what the check is about, is unaffected.
    """
    K, t = cfg["K"], cfg["t_split"]
    Ne_d, Ne_A, Ne_anc = cfg["Ne_deme"], cfg["Ne_A"], cfg["Ne_ancestral"]
    lam = np.sort(laplacian_eigs(K)[0])[1:]        # drop the trivial 0
    n_eq = np.full(K * K, float(n_B))

    rows = []
    for x in inv_M_values:
        m = m_from_M(1.0 / x, Ne_d)
        T_tr, _ = T_pop_matrix(K, m, t, Ne_d, Ne_A, Ne_anc)
        nu_tr, _ = structural_spectrum(T_tr[:K * K, :K * K], n_eq)   # B-only, like-for-like
        nu_eq, _ = structural_spectrum(T_equilibrium_B(K, m, Ne_d), n_eq)
        for k in range(min(len(lam), len(nu_tr), len(nu_eq))):
            rows.append(dict(inv_M=x, M=1.0 / x, mode=k + 1, l_k=lam[k],
                             x_k=2.0 * m * lam[k] * t,
                             nu=nu_tr[k], nu_eq=nu_eq[k],
                             ratio=nu_tr[k] / nu_eq[k]))
    return pd.DataFrame(rows)


# --- driver ----------------------------------------------------------------

def run(outdir=OUTDIR, n_grid=60, L_eff=3462.0, verbose=True):
    """
    Build the spectrum tables over a log grid of 1/M spanning the simulated range.

    L_eff is the median effective SNP count measured for the n_B = 4 cell of
    split_ss_vary_N (results/sweeps/split_ss_vary_N/replicates.csv); it enters only nu_c.
    """
    os.makedirs(outdir, exist_ok=True)
    cfg = load_config()
    K, t = cfg["K"], cfg["t_split"]
    Ne_d, Ne_A, Ne_anc = cfg["Ne_deme"], cfg["Ne_A"], cfg["Ne_ancestral"]

    lines = [f"demography: K={K} ({K*K} demes), Ne_ancestral={Ne_anc:g}, "
             f"Ne_deme={Ne_d:.4f}, Ne_A={Ne_A:g}, t_split={t:g}",
             f"1/M range [{cfg['inv_M_min']:g}, {cfg['inv_M_max']:g}], L_eff={L_eff:g}", ""]
    validate(cfg, lines)
    validate_msprime(cfg, lines)
    if verbose:
        print("\n".join(lines), flush=True)

    inv_M = np.geomspace(cfg["inv_M_min"], cfg["inv_M_max"], n_grid)
    cells = [(4, 0.5), (4, 0.1)]            # panel A reference, and panel B's extreme

    spec_rows, gap_rows = [], []
    for x in inv_M:
        m = m_from_M(1.0 / x, Ne_d)
        T_pop, _ = T_pop_matrix(K, m, t, Ne_d, Ne_A, Ne_anc)
        T_eq = T_equilibrium_B(K, m, Ne_d)
        for n_B, phi in cells:
            r = cell_spectrum(T_pop, T_eq, K, n_B, phi, L_eff)
            # The equilibrium spectrum is B-only and so has no A-vs-B mode: its spatial
            # pair leads at 0, 1 while the transient one sits at r["pair"]. Storing the two
            # rank-for-rank would pair the transient A mode with the equilibrium spatial
            # mode and draw a dashed "equilibrium" curve belonging to a different mode.
            # The offset that aligns the two pairs is applied instead, and any transient
            # mode with no equilibrium counterpart (the A mode itself) gets NaN.
            shift = min(r["pair"])
            for k in range(min(N_MODES, len(r["nu"]))):
                eq_k = k - shift
                spec_rows.append(dict(
                    inv_M=x, M=1.0 / x, m=m, n_B=n_B, phi=phi, N=r["N"], n_A=r["n_A"],
                    mode=k, nu=r["nu"][k],
                    nu_eq=(r["nu_eq"][eq_k] if 0 <= eq_k < len(r["nu_eq"]) else np.nan),
                    is_spatial=k in r["pair"], is_A_mode=(k == r["a_idx"])))
            gap_rows.append(dict(
                inv_M=x, M=1.0 / x, m=m, n_B=n_B, phi=phi, N=r["N"], n_A=r["n_A"],
                nu_spatial=r["nu_spatial"], nu_next=r["nu_next"], gap=r["gap"],
                pair_i=r["pair"][0], pair_j=r["pair"][1], pair_cos=r["pair_cos"],
                pair_p2=r["pair_p2"],
                a_idx=r["a_idx"], psi_pair=r["psi_pair"], nu_c=r["nu_c"],
                gap_eq=_equilibrium_gap(r["nu_eq"])))
        if verbose and (len(gap_rows) % 20 == 0):
            print(f"  1/M={x:.4g} done", flush=True)

    spec = pd.DataFrame(spec_rows)
    gaps = pd.DataFrame(gap_rows)
    # Every 3rd grid point: the censoring check pools over (mode, M) and wants a spread of
    # M, but each point costs another phase-type solve. A stride that leaves only two or
    # three M values makes the pooled Spearman meaningless.
    cens = censoring_table(cfg, inv_M[::3])

    spec.to_csv(os.path.join(outdir, "spectrum.csv"), index=False)
    gaps.to_csv(os.path.join(outdir, "gaps.csv"), index=False)
    cens.to_csv(os.path.join(outdir, "censoring.csv"), index=False)

    _summarise(spec, gaps, cens, lines)
    with open(os.path.join(outdir, "validation.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if verbose:
        print("\n".join(lines[len(lines) - _SUMMARY_LINES:]), flush=True)
        print(f"\nwrote {outdir}")
    return spec, gaps, cens


def _equilibrium_gap(nu_eq):
    """
    The equilibrium relative gap between the spatial pair and the next mode.

    The equilibrium system is B-only, so it has no A-vs-B mode and its spatial pair leads:
    indices 0, 1 are the pair and index 2 is the next mode. That is NOT the transient
    indexing, where the A-vs-B mode leads and the pair sits at 1, 2 -- the two spectra must
    never be compared rank-for-rank.

    The result is ~0.51 at K=9 and drifts only slightly over the sweep (0.506 at M = 24.7
    to 0.554 at M = 0.247). It is NOT exactly M-invariant: strict proportionality to 1/M
    holds for the resistance-distance approximation, but this is the exact phase-type solve,
    where nu*M runs from 32.95 to 29.98 over the same range because coalescence within a
    deme competes with migration out of it. The point of the comparison is the order of
    magnitude anyway -- the transient gap falls by ~10x across the sweep while this one
    moves by under 10% and in the opposite direction.
    """
    if len(nu_eq) < 3:
        return np.nan
    pair = 0.5 * (nu_eq[0] + nu_eq[1])
    return (pair - nu_eq[2]) / pair


_SUMMARY_LINES = 0


def _summarise(spec, gaps, cens, lines):
    """The printed run-log summary the plan asks for."""
    global _SUMMARY_LINES
    start = len(lines)
    lines.append("")
    lines.append("=== summary ===")

    peaks = {0.5: 0.328968, 0.1: 0.461648}      # panel-A / panel-B fitted maxima
    for phi, g in gaps.groupby("phi"):
        g = g.sort_values("inv_M")
        lines.append(f"phi={phi:g} (N={int(g.N.iloc[0])}, n_A={int(g.n_A.iloc[0])}):")
        for lvl in (0.2, 0.1, 0.05):
            below = g[g.gap < lvl]
            x = below.inv_M.min() if len(below) else np.nan
            lines.append(f"   relative gap crosses {lvl:.2f} at 1/M = {x:.4g}")
        lines.append(f"   measured cos^2 maximum at 1/M = {peaks.get(phi, np.nan):.4g}")
        not_23 = (~(((g.pair_i == 0) & (g.pair_j == 1)) |
                    ((g.pair_i == 1) & (g.pair_j == 2)))).mean()
        lines.append(f"   spatial pair not at leading adjacent indices: {not_23:.1%} of M "
                     f"(pair index modes: {sorted(set(zip(g.pair_i, g.pair_j)))})")
        lines.append(f"   min over M of pair-vs-DCT cosine: {g.pair_cos.min():.4f}")
        lines.append(f"   max over M of Psi(spatial pair): {g.psi_pair.max():.4g} "
                     f"(detectable iff < 1)")
        ratio = g.nu_spatial / g.nu_c
        lines.append(f"   min over M of nu_spatial / nu_c: {ratio.min():.4g} "
                     f"(at 1/M = {g.inv_M.iloc[int(np.argmin(ratio.values))]:.4g}); "
                     f"nu_c in [{g.nu_c.min():.4g}, {g.nu_c.max():.4g}]")
        lines.append(f"   equilibrium gap: [{g.gap_eq.min():.4f}, {g.gap_eq.max():.4f}]")

    rho, p = spearmanr(cens.x_k, cens.ratio)
    lines.append(f"censoring: spearman(x_k, nu/nu_eq) = {rho:+.4f} (p={p:.2g}, "
                 f"n={len(cens)})")
    try:
        popt, _ = curve_fit(lambda x, a: 1.0 - np.exp(-a * x), cens.x_k, cens.ratio,
                            p0=[1.0], maxfev=10000)
        pred = 1.0 - np.exp(-popt[0] * cens.x_k)
        ss = 1.0 - np.sum((cens.ratio - pred) ** 2) / np.sum(
            (cens.ratio - cens.ratio.mean()) ** 2)
        lines.append(f"           fit nu/nu_eq = 1 - exp(-a x_k): a = {popt[0]:.4f}, "
                     f"R^2 = {ss:.4f}")
    except Exception as exc:                                  # pragma: no cover
        lines.append(f"           fit failed: {exc}")
    _SUMMARY_LINES = len(lines) - start


if __name__ == "__main__":
    run()

# %%
