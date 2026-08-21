"""
Where the stepping-stone detection threshold sits, and the migration-rate grids that
bracket it.

The sweeps in SS_1D.py / SS_2D.py only test the theory if the critical migration rate
falls inside the range they scan. M_c is not a fixed number: it grows like K^(3/2) in 1D
and like K in 2D, and like sqrt(N L) in both, so a single hard-coded [min_m, max_m]
cannot bracket the transition across a K sweep. This module computes M_c per cell and
builds the grid around it, which is what keeps every simulation informative.

    laplacian_eigenvalues   non-zero mu_k of the scaled Laplacian, 1D chain or 2D grid
    structure_eigenvalues   nu_k = N / (M K mu_k), the spatial structure eigenvalues
    critical_M              largest M at which PC1 is still detectable (Psi criterion)
    critical_M_approx       the closed form, for reference
    migration_grid          num_m rates log-spaced around m_c = M_c / (4 Ne)
    Ne_for_target_m         Ne placing m_c at a target rate, keeping m physical

Ne is a free dial. Only M = 4 Ne mbar enters the theory, so raising Ne is how a large-K
1D cell keeps its critical migration rate well below 1 -- at Ne = 500, K = 50 would need
m_c = 0.80, which is near-total replacement per generation and no longer a stepping
stone.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from utils import detection_psi


def laplacian_eigenvalues(K, dimension=1):
    """
    Non-zero eigenvalues of the scaled migration Laplacian, ascending.

    Reflecting (Neumann) boundaries make the 1D chain a second-difference matrix whose
    eigenvalues are known in closed form, mu_k = 4 sin^2(pi k / 2K). The 2D grid is the
    Kronecker sum I (x) Lambda_1D + Lambda_1D (x) I, so its eigenvalues are all pairwise
    sums; the single zero (k = l = 0) is dropped and the rest returned.

    Args:
        K: number of demes (1D) or grid side length, i.e. K x K demes (2D).
        dimension: 1 or 2.
    """
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}")
    mu_1d = 4 * np.sin(np.pi * np.arange(1, K) / (2 * K)) ** 2
    if dimension == 1:
        return mu_1d
    if dimension == 2:
        with_zero = np.concatenate([[0.0], mu_1d])
        return np.sort((with_zero[:, None] + with_zero[None, :]).ravel())[1:]
    raise ValueError(f"dimension must be 1 or 2, got {dimension}")


def num_demes(K, dimension=1):
    """Total demes: K for a chain, K^2 for a grid."""
    return K if dimension == 1 else K * K


def structure_eigenvalues(K, M, n_per_deme, dimension=1):
    """
    The K_demes - 1 structure eigenvalues nu_k = N / (M K_demes mu_k), descending.

    Under even sampling and homogeneous migration the resistance-distance form of F_hat
    collapses to this: the eigenvectors of R restricted to 1^perp are those of the
    Laplacian, so only the mu_k are needed.
    """
    mu = laplacian_eigenvalues(K, dimension)
    K_demes = num_demes(K, dimension)
    N = n_per_deme * K_demes
    return np.sort(N / (M * K_demes * mu))[::-1]


def closed_form_eigenvector_1d(k, K, n_per_deme):
    """
    u_k(j) = cos(k*pi/K * (j + 1/2)), j = 0..K-1, lifted to sample space and normalized.

    k = 1..K-1 indexes the modes in the same ascending-mu / descending-nu order as
    laplacian_eigenvalues; k = 1 is the leading structure axis. Verified numerically
    against theoretical_eigenvectors(stepping_stone_T(...)): cosine similarity 1.0 at
    K=25, n_per_deme=64.
    """
    j = np.arange(K)
    u = np.cos(k * np.pi / K * (j + 0.5))
    lifted = np.repeat(u, n_per_deme)
    return lifted / np.linalg.norm(lifted)


def closed_form_eigenvector_2d(k, l, K, n_per_deme):
    """
    u_kl(p, q) = cos(k*pi/K*(p+1/2)) * cos(l*pi/K*(q+1/2)), a K x K grid of demes
    ordered row-major (p*K + q, matching demographies.stepping_stone_2d), lifted to
    sample space and normalized.

    (k, l) = (1, 0) and (0, 1) are the degenerate leading pair. Verified numerically:
    the (1,0)/(0,1) pair spans the same subspace as the leading multiplet of
    theoretical_eigenvectors(stepping_stone_T(...)) to within ~1e-4 (numerical
    resistance-distance asymmetry), at K=9, n_per_deme=64.
    """
    p, q = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    u = (np.cos(k * np.pi / K * (p + 0.5)) * np.cos(l * np.pi / K * (q + 0.5))).ravel()
    lifted = np.repeat(u, n_per_deme)
    return lifted / np.linalg.norm(lifted)


def closed_form_spectrum(K, M, n_per_deme, dimension=1):
    """
    Closed-form (mode, eigenvalue, eigenvector) triples, descending eigenvalue.

    In 1D modes are k = 1..K-1. In 2D modes are (k, l) with 0 <= k, l <= K-1, excluding
    (0, 0) (the mean-centering direction); the eigenvalue depends on (k, l) only through
    mu_k + mu_l, so nu_{kl} = nu_{lk} and (1, 0)/(0, 1) are always the leading degenerate
    pair.

    Returns:
        (modes, nu, V): modes is a list of int (1D) or (int, int) tuples (2D); nu is the
        matching eigenvalue array; V is the (N, K_demes - 1) matrix of lifted, normalized
        eigenvectors, columns in the same order as modes and nu.
    """
    K_demes = num_demes(K, dimension)
    N = n_per_deme * K_demes

    if dimension == 1:
        modes = list(range(1, K))
        mu = 4 * np.sin(np.pi * np.arange(1, K) / (2 * K)) ** 2
        nu = N / (M * K_demes * mu)
        V = np.column_stack([closed_form_eigenvector_1d(k, K, n_per_deme) for k in modes])
    elif dimension == 2:
        mu_1d = 4 * np.sin(np.pi * np.arange(K) / (2 * K)) ** 2  # mu_1d[0] = 0
        modes = [(k, l) for k in range(K) for l in range(K) if not (k == 0 and l == 0)]
        mu = np.array([mu_1d[k] + mu_1d[l] for k, l in modes])
        nu = N / (M * K_demes * mu)
        V = np.column_stack([closed_form_eigenvector_2d(k, l, K, n_per_deme)
                             for k, l in modes])
    else:
        raise ValueError(f"dimension must be 1 or 2, got {dimension}")

    order = np.argsort(nu)[::-1]
    modes = [modes[i] for i in order]
    return modes, nu[order], V[:, order]


def full_spectrum(K, M, n_per_deme, dimension=1):
    """
    Structure eigenvalues followed by the N - K_demes floor levels.

    Homogeneous migration and equal deme sizes make the within-deme coalescent times
    equal, so every eps_a vanishes and the whole floor sits at zero.
    """
    nu = structure_eigenvalues(K, M, n_per_deme, dimension)
    K_demes = num_demes(K, dimension)
    N = n_per_deme * K_demes
    return np.concatenate([nu, np.zeros(N - K_demes)])


def is_pc1_detectable(K, M, n_per_deme, L, dimension=1):
    """
    Whether the leading structure eigenvalue clears the Psi < 1 criterion.

    N ~ K_demes here rather than N >> K_demes, so the floor is not an overwhelming
    majority and the spatial criterion applies instead of g(). In 2D the leading
    eigenvalue is doubly degenerate (mu_10 = mu_01) and is judged as the subspace it
    spans, per the structure-eigenvalue convention in detection_psi.
    """
    spectrum = full_spectrum(K, M, n_per_deme, dimension)
    return detection_psi(0, spectrum, L, degenerate_subspace=True) < 1.0


def critical_M(K, n_per_deme, L, dimension=1, lo=1e-2, hi=1e9, iterations=200):
    """
    The largest population-scaled migration rate M at which PC1 is still detectable.

    Detectability is monotone in M -- larger M means more mixing, smaller nu_k -- so a
    bisection on the Psi criterion is well posed. This is the exact threshold for the
    given K; critical_M_approx is its large-K closed form.

    Returns nan if PC1 is undetectable even at lo (nothing to bracket).
    """
    if not is_pc1_detectable(K, lo, n_per_deme, L, dimension):
        return np.nan
    for _ in range(iterations):
        mid = np.sqrt(lo * hi)
        if is_pc1_detectable(K, mid, n_per_deme, L, dimension):
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def critical_M_approx(K, n_per_deme, L, dimension=1):
    """
    The closed-form threshold, from mu_1 -> pi^2 / K^2 in the large-K limit.

        1D:  M_c = sqrt(n K^3 L) / pi^2
        2D:  M_c = sqrt(n K^2 L) / pi^2

    Within ~0.5% of critical_M down to K = 10 (1D) and K = 5 (2D); kept as the reference
    the simulations are ultimately testing.
    """
    K_demes = num_demes(K, dimension)
    N = n_per_deme * K_demes
    scale = K if dimension == 1 else 1.0
    return scale * np.sqrt(N * L) / np.pi ** 2


PANMICTIC_M_THRESHOLD = 0.1
PANMICTIC_TOLERANCE = 1e-6
_MAX_SPLIT_DERIVED = 99


def mixing_time(K, m, dimension=1, tolerance=PANMICTIC_TOLERANCE):
    """
    Generations until a lineage is essentially equally likely to be in any deme.

    A single lineage's deme occupancy p(t) follows dp/dt = -m Lambda^T p, so in the
    Laplacian eigenbasis the component along mode k decays as exp(-m mu_k t). The mu_0 = 0
    mode is the uniform stationary distribution; every deviation from uniform decays, and
    the slowest is the one along mu_1. Requiring that deviation to fall below `tolerance`
    gives

        t_mix = log(1 / tolerance) / (m * mu_1).

    Checked against the exact total-variation distance (worst-case start, via matrix
    exponential): the coefficient log(1/tol) is accurate to a few percent -- 8.76 measured
    against log(1e4) = 9.21 -- and the residual dependence on deme count is weak, moving
    only 8.76 -> 8.98 from 10 to 100 demes. This is a single-lineage argument; coalescence
    depends on *pair* occupancy, which relaxes on a comparable but not identical timescale,
    so the tolerance carries a margin rather than being a tight bound.
    """
    if not 0 < tolerance < 1:
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    mu_1 = laplacian_eigenvalues(K, dimension)[0]
    return float(np.log(1.0 / tolerance) / (m * mu_1))


def add_panmictic_phase(demography, K, m, Ne, dimension=1,
                        tolerance=PANMICTIC_TOLERANCE):
    """
    Collapse a stepping-stone demography into one panmictic deme past the mixing time.

    Once lineages are uniformly spread the structured model is indistinguishable from a
    single population of the pooled size Ne * K_demes (Wakeley's collecting phase), but
    msprime still pays to track migration among every deme. Splitting the demes into one
    ancestral population at t_mix keeps the part of the history that carries structure and
    drops the part that does not.

    Measured on a 10x10 grid this is 5.9x faster at m = 0.05 and 45x at m = 0.40 -- the
    speedup grows with m because faster mixing means an earlier switch -- while mean
    coalescent times stay within ~1-2% of the exact model, inside Monte Carlo error.

    Mutates and returns `demography`.
    """
    K_demes = num_demes(K, dimension)
    t_switch = mixing_time(K, m, dimension, tolerance)

    names = [pop.name for pop in demography.populations]
    if len(names) != K_demes:
        raise ValueError(f"expected {K_demes} demes, found {len(names)}")

    demography.add_population(name="pooled", initial_size=Ne * K_demes)
    # msprime allows at most 100 populations per split event, counting the ancestral one,
    # so at most 99 derived. A 10x10 grid has exactly 100 demes and just exceeds that;
    # the remainder joins in a second event an instant later.
    demography.add_population_split(
        time=t_switch, derived=names[:_MAX_SPLIT_DERIVED], ancestral="pooled")
    if len(names) > _MAX_SPLIT_DERIVED:
        demography.add_population_split(
            time=t_switch * (1 + 1e-9), derived=names[_MAX_SPLIT_DERIVED:],
            ancestral="pooled")
    return demography


def maybe_pool(demography, K, m, Ne, dimension=1,
               threshold=PANMICTIC_M_THRESHOLD, tolerance=PANMICTIC_TOLERANCE):
    """
    Apply add_panmictic_phase only when m exceeds `threshold`.

    Below the threshold the timescales are not well separated enough for the switch to be
    clearly safe, and there is little to gain anyway: the saving comes from skipping a long
    well-mixed phase, which barely exists when migration is slow. Returns
    (demography, t_switch or None).
    """
    if m <= threshold:
        return demography, None
    t_switch = mixing_time(K, m, dimension, tolerance)
    return add_panmictic_phase(demography, K, m, Ne, dimension, tolerance), t_switch


def Ne_for_target_m(M_c, target_m):
    """
    Deme size placing the critical migration rate at target_m, since M = 4 Ne mbar.

    Rounded to a whole individual. Used to keep m_c ~ 0.05 across a 1D K sweep instead of
    letting it climb towards 1.
    """
    return int(round(M_c / (4.0 * target_m)))


def migration_grid(m_c, num_m, span=8.0):
    """
    num_m migration rates geometrically spaced over [m_c/span, m_c*span].

    Centring on m_c is the point: it guarantees the transition is bracketed with equal
    resolution on both sides, whatever K and L do to the threshold. Geometric spacing is
    what makes "both sides" true -- the grid is symmetric in log m about m_c, so half the
    simulations land on each side of the transition.

    The earlier sweeps used a mirrored geomspace, min + max - geomspace(min, max), to
    bunch points towards high m. That reflection is linear while the spacing is
    logarithmic, so it does not preserve the log grid: for span = 8 it puts 97% of the
    rates above m_c and leaves the detectable side with almost no points. Plain
    geomspace instead.

    This rejection is specific to centring symmetrically around a known m_c -- it is not
    a blanket objection to mirrored spacing. When the sweep instead uses an explicit
    [min, max] range and the transition is expected near one end (e.g. m_max), denser
    coverage there is exactly the point; see sweep.axis_values(..., spacing=
    "log_dense_high"), used by the stepping-stone kind configs for that case.
    """
    if num_m < 1:
        raise ValueError(f"num_m must be at least 1, got {num_m}")
    if not np.isfinite(m_c) or m_c <= 0:
        raise ValueError(f"m_c must be finite and positive, got {m_c}")
    if span <= 1:
        raise ValueError(f"span must exceed 1, got {span}")
    if num_m == 1:
        return np.array([m_c])
    return np.geomspace(m_c / span, m_c * span, num_m)


def sweep_plan(K_values, n_per_deme, L, dimension=1, num_m=60, span=8.0,
               Ne=None, target_m=None):
    """
    One row per (K, m) cell: the migration grid for each K, centred on that K's own M_c.

    Exactly one of Ne or target_m must be given. A fixed Ne keeps the demography constant
    and lets m_c move with K; target_m instead solves for Ne per K so that m_c lands on a
    chosen rate, which is what keeps m physical when M_c grows like K^(3/2).

    L here is the raw SNP count, so the M_c used to centre the grid is an overestimate:
    the aspect ratio that governs the bulk is N / L_eff, and the variance-effective count
    runs well under half of L (~470 of 1000 in the existing 2D runs). Since M_c ~ sqrt(L)
    that is roughly a factor 1.5, comfortably inside the default span of 8, so the grid
    still brackets the true threshold -- but it is why the grid is centred generously
    rather than tightly, and why the analysis recomputes M_c from the measured L_eff
    instead of reusing the M_c recorded here.

    Returns:
        list of dicts, one per K, with keys K, K_demes, N, Ne, M_c, M_c_approx, m_c,
        and the m_values array.
    """
    if (Ne is None) == (target_m is None):
        raise ValueError("give exactly one of Ne or target_m")

    plan = []
    for K in K_values:
        K_demes = num_demes(K, dimension)
        M_c = critical_M(K, n_per_deme, L, dimension)
        Ne_K = Ne if Ne is not None else Ne_for_target_m(M_c, target_m)
        m_c = M_c / (4.0 * Ne_K)
        plan.append(dict(
            K=K, K_demes=K_demes, N=n_per_deme * K_demes, Ne=Ne_K,
            M_c=M_c, M_c_approx=critical_M_approx(K, n_per_deme, L, dimension),
            m_c=m_c, m_values=migration_grid(m_c, num_m, span=span)))
    return plan
