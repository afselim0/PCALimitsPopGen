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
