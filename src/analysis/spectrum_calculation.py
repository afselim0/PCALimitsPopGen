#%%
import numpy as np
from scipy.optimize import brentq
#%%
def center_and_normalize(X):
    """
    Double-center and normalize: G = -(H @ X @ H) / X_bar, with H = I - (1/n) * 11^T
    and X_bar the mean of the off-diagonal entries of X.

    X_bar is taken from the raw input X (not from the centered, negated matrix), per
    G = -(H T H)/T_bar and G_hat = (H Delta H)/(L * Delta_bar); normalizing by the
    off-diagonal mean of -(H X H) instead would flip the sign of the whole spectrum.
    """
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    off_diag_mean = X[~np.eye(n, dtype=bool)].mean()
    return -(H @ X @ H) / off_diag_mean

def compute_sample_covariance(Z):
    """
    Sample GRM from a binary genotype matrix Z, shape (n, L).

    Delta_ij = mean squared Hamming distance per locus between samples i, j.
    G = -H @ Delta @ H, rescaled by the off-diagonal mean distance.
    """
    n, L = Z.shape
    sq_norms = np.sum(Z ** 2, axis=1)
    Delta = (sq_norms[:, None] + sq_norms[None, :] - 2 * Z @ Z.T) / L
    return center_and_normalize(Delta)

def expand_coal_times(T, sample_sizes):
    """
    Expand a population-level (K x K) mean pairwise coalescent time matrix T to the
    sample-level (N x N) matrix. Each entry T_ij fills a block of size (n_i, n_j),
    where n_i and n_j are the sample sizes for populations i and j.

    The sample-level diagonal is zero: a sample's coalescent time with itself is 0,
    while two *distinct* samples from population a coalesce at tau_a = T[a, a]. Leaving
    tau_a on the diagonal would make each diagonal block constant, collapsing the
    N - K "floor" eigenvalues of the resulting covariance matrix.
    """
    num_pops = len(sample_sizes)
    expanded_T = np.zeros((sum(sample_sizes), sum(sample_sizes)))

    start_i = 0
    for i in range(num_pops):
        end_i = start_i + sample_sizes[i]
        start_j = 0
        for j in range(num_pops):
            end_j = start_j + sample_sizes[j]
            expanded_T[start_i:end_i, start_j:end_j] = T[i, j]
            start_j = end_j
        start_i = end_i

    np.fill_diagonal(expanded_T, 0.0)
    return expanded_T

def compute_expected_covariance(T, sample_sizes):
    """
    E[GRM] from a population-level mean pairwise coalescent time matrix T, that is exanded according to sample sizes
    """
    expanded_T = expand_coal_times(T, sample_sizes)
    return center_and_normalize(expanded_T)

def compute_F_hat(T, sample_sizes):
    """
    Compute the F_hat matrix from a population-level mean pairwise coalescent time
    matrix T (K x K, with diag(T) = tau the within-population mean coalescent times)
    and the sample sizes n_a for each population.

    phi_a   = n_a / N,                          psi_a = sqrt(phi_a)
    eps_a   = tau_a / tau_bar - 1,              tau_bar = sum_a phi_a * tau_a
    (F_2)_ab = T_ab - (tau_a + tau_b) / 2
    F_ab    = -N / tau_bar * sqrt(phi_a * phi_b) * (F_2)_ab   (zero diagonal)
    P_hat   = I_K - psi psi^T
    F_hat   = P_hat @ (diag(eps) + F) @ P_hat
    """
    T = np.asarray(T, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    K = T.shape[0]
    N = n.sum()

    phi = n / N
    psi = np.sqrt(phi)

    tau = np.diag(T)
    tau_bar = phi @ tau
    eps = tau / tau_bar - 1

    F_2 = T - (tau[:, None] + tau[None, :]) / 2
    F = -N / tau_bar * np.outer(psi, psi) * F_2
    np.fill_diagonal(F, 0.0)

    P_hat = np.eye(K) - np.outer(psi, psi)

    return P_hat @ (np.diag(eps) + F) @ P_hat

def K2_structure_eigenvalue(sample_sizes, T_within, T_between):
    """
    Compute the single eigenvalue of a K = 2 demogrpahy with two populations and a single split time, given the sample sizes and mean coalescent times.

    Args:
        sample_sizes: list or array of length 2, the sample sizes for the two populations
        T_within: list or array of length 2, the mean coalescent times within each population
        T_between: float, the mean coalescent time between the two populations
    """
    assert len(sample_sizes) == 2, "sample_sizes must have length 2"
    assert len(T_within) == 2, "T_within must have length 2"
    n1, n2 = sample_sizes
    tau_1, tau_2 = T_within
    N = n1 + n2
    phi_1 = n1 / N
    phi_2 = 1 - phi_1
    tau_bar = phi_1 * tau_1 + phi_2 * tau_2
    F2 = T_between - (tau_1 + tau_2) / 2
    return 2*N*phi_1*phi_2*F2/tau_bar - (phi_1 - phi_2)*(tau_1 - tau_2)/tau_bar

def critical_eigenvalue(phi, eps, gamma, xtol=1e-14, max_expand=200):
    """
    The detection threshold nu_c: the largest root of

        g(nu) = gamma * sum_a phi_a * ((1 + eps_a) / (nu - eps_a))^2 = 1,

    where gamma = N / L is the aspect ratio, phi_a = n_a / N the sampling
    proportions, and eps_a = tau_a / tau_bar - 1 the per-population deviations of
    the within-population coalescent times. A structure eigenvalue nu_k is
    predicted to be detectable only if nu_k > nu_c.

    The largest root always lies in (eps_max, inf), which makes the solve
    unconditionally robust rather than dependent on a starting guess: on that
    interval every term of g is positive and strictly decreasing in nu, so g is
    strictly decreasing, with g(nu) -> +inf as nu -> eps_max+ (the nearest pole)
    and g(nu) -> 0 as nu -> inf. Hence there is exactly one crossing of 1 there,
    and any root below eps_max is smaller by construction. Bisection on a bracket
    grown outward from eps_max therefore converges to the root wanted, and cannot
    be trapped between poles the way a Newton iteration can.

    Populations with phi_a == 0 contribute nothing and are dropped, so that an
    empty population's eps_a cannot plant a spurious pole.

    Args:
        phi: sampling proportions phi_a (length K). Need not be normalized.
        eps: within-population deviations eps_a (length K).
        gamma: aspect ratio N / L (positive).

    Returns:
        float: nu_c. For eps_a == 0 for all a this is exactly sqrt(gamma).
    """
    phi = np.asarray(phi, dtype=float)
    eps = np.asarray(eps, dtype=float)

    if phi.shape != eps.shape:
        raise ValueError(f"phi and eps must have the same shape, got "
                         f"{phi.shape} and {eps.shape}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma:g}")
    if np.any(phi < 0):
        raise ValueError("phi must be non-negative")

    keep = phi > 0
    if not keep.any():
        raise ValueError("phi must have at least one positive entry")
    phi, eps = phi[keep], eps[keep]
    phi = phi / phi.sum()

    def g(nu):
        return gamma * np.sum(phi * ((1.0 + eps) / (nu - eps)) ** 2)

    eps_max = eps.max()

    # Step off the pole by an amount scaled to the problem, then walk in until
    # g > 1. Starting exactly at eps_max would divide by zero.
    scale = max(abs(eps_max), np.sqrt(gamma), 1.0)
    lo = eps_max + scale * 1e-12
    for _ in range(max_expand):
        if g(lo) > 1.0:
            break
        lo = eps_max + (lo - eps_max) / 2.0
    else:
        raise RuntimeError("could not find a point with g > 1 next to the pole")

    # Grow the upper end until g < 1; g -> 0 at infinity so this always terminates.
    hi = max(lo + scale, eps_max + 2.0 * scale)
    for _ in range(max_expand):
        if g(hi) < 1.0:
            break
        hi = eps_max + 2.0 * (hi - eps_max)
    else:
        raise RuntimeError("could not bracket the root from above")

    return float(brentq(lambda nu: g(nu) - 1.0, lo, hi, xtol=xtol, rtol=8.9e-16))

def critical_eigenvalue_from_T(T, sample_sizes, L):
    """
    nu_c computed directly from a population-level coalescent time matrix T
    (K x K, diag = tau) and the per-population sample sizes, with L SNPs.

    Convenience wrapper: derives phi, eps, and gamma = N/L the same way
    compute_F_hat does, so the threshold and the structure eigenvalues it is
    compared against are always built from a consistent set of quantities.
    """
    T = np.asarray(T, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    N = n.sum()

    phi = n / N
    tau = np.diag(T)
    tau_bar = phi @ tau
    eps = tau / tau_bar - 1.0

    return critical_eigenvalue(phi, eps, gamma=N / L)

import numpy as np

def K3_structure_eigenvalues(T, n):
    """Analytically computes the two structure eigenvalues of a K=3 demography given a generic matrix of pairwise coalescent times and a generic number of samples 

    Args:
        T (np.ndarray): A 3x3 matrix of pairwise coalescent times.
        n (list): A list or array of length 3, the sample sizes for the three populations.

    Returns:
        np.ndarray: An array of length 2, the two structure eigenvalues.
    """
    T = np.asarray(T, dtype=float)
    n = np.asarray(n, dtype=float)
    assert T.shape == (3, 3), "T must be a 3x3 matrix"
    assert len(n) == 3, "n must have length 3"
    N = n.sum()
    phi = n / N
    phi1, phi2, phi3 = phi
    s = phi1 + phi2

    tau = np.diag(T)
    tau1, tau2, tau3 = tau
    tau_bar = phi @ tau
    eps = tau / tau_bar - 1.0
    eps1, eps2, eps3 = eps

    def F2(a, b):
        return T[a, b] - (tau[a] + tau[b]) / 2.0

    g12 = -N * np.sqrt(phi1 * phi2) * F2(0, 1) / tau_bar
    g13 = -N * np.sqrt(phi1 * phi3) * F2(0, 2) / tau_bar
    g23 = -N * np.sqrt(phi2 * phi3) * F2(1, 2) / tau_bar

    M11 = (phi2 * eps1 + phi1 * eps2 - 2 * np.sqrt(phi1 * phi2) * g12) / s
    M22 = (phi1 * phi3 * eps1 + phi2 * phi3 * eps2 + s**2 * eps3
           + 2 * np.sqrt(phi1 * phi3 * phi2 * phi3) * g12
           - 2 * np.sqrt(phi1 * phi3) * s * g13
           - 2 * np.sqrt(phi2 * phi3) * s * g23) / s
    M12 = (np.sqrt(phi1 * phi2 * phi3) * (eps1 - eps2)
           + np.sqrt(phi3) * (phi2 - phi1) * g12
           - np.sqrt(phi2) * s * g13
           + np.sqrt(phi1) * s * g23) / s

    trace = M11 + M22
    disc = np.sqrt(((M11 - M22) / 2.0) ** 2 + M12 ** 2)
    return np.array([trace / 2.0 + disc, trace / 2.0 - disc])