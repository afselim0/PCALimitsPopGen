"""
Detection thresholds and eigenvector overlaps for spiked separable covariance
matrices, following Ding & Yang (2021), "Spiked separable covariance matrices and
principal components," Annals of Statistics, DOI 10.1214/20-AOS1995.

This module is the reference for the theory as well as the implementation; the
full derivation lives here rather than in CLAUDE.md.

================================================================================
SETUP
================================================================================

K populations, a K x K matrix of mean pairwise coalescent times, sampling
proportions phi_a (a = 1..K), N total samples, L independent loci, gamma = N / L.

The theoretical (population) covariance matrix has:

  * spike eigenvalues  (1 + nu_k) / (1 + alpha),  k = 1..K-1   (the structure)
  * bulk/floor eigenvalues  (1 + eps_a) / (1 + alpha),  each with multiplicity
    N phi_a - 1,  where eps_a = tau_a / tau_bar - 1

alpha is a fixed normalizing constant (see NOTE ON ALPHA at the bottom); it
cancels out of every formula below and is ignored when computing nu_c and the
overlap. It matters only when converting a predicted eigenvalue back to the
observed scale of S = X X^T / L.

Two questions:

  1. How large must nu_k be, relative to the bulk, to produce a detectable
     outlier eigenvalue in S?  (critical value: nu_c)
  2. Given a detectable nu_k, how well does the sample eigenvector recover the
     true one?  (eigenvector overlap: <u_k, u_hat_k>^2)

Two sources of heteroskedasticity complicate both:

  1. eps_a is generically non-zero -> the bulk is heteroskedastic (row/individual
     side).
  2. Loci have unequal variance h_l = p_l (1 - p_l) -> the noise is
     heteroskedastic on the column/locus side.

Ding & Yang handles exactly this two-sided setting.

================================================================================
DEFINITIONS
================================================================================

    h_l = p_l (1 - p_l),   h_bar = (1/L) sum_l h_l,   eta_l = h_l / h_bar

    Phi(nu)   = sum_a phi_a (1 + eps_a) / (nu - eps_a)
    Phi_2(nu) = sum_a phi_a ((1 + eps_a) / (nu - eps_a))^2

    Psi(w)    = (1/L) sum_l eta_l / (w - eta_l)
    Psi_2(w)  = (1/L) sum_l (eta_l / (w - eta_l))^2

Phi and Psi are strictly decreasing on their domains (nu > max_a eps_a,
w > max_l eta_l), so every root-find below is unique and can use plain bisection.

================================================================================
EXACT PROCEDURE
================================================================================

Step 1 -- coupling.  For a given nu, find w(nu) solving

    Psi(w) = gamma * Phi(nu)                              [bisection in w]

Step 2 -- critical value.  nu_c solves, jointly with step 1,

    (Phi_2(nu_c) / Phi(nu_c)) * (Psi_2(w_c) / Psi(w_c)) = 1

(nested bisection: for each trial nu, get w(nu) from Step 1, evaluate the
left-hand side, bisect on nu until it equals 1). Mode k is detectable iff
nu_k > nu_c.

Step 3 -- overlap.  For a detectable nu_k, get w_k from Step 1 at nu = nu_k, then

    <u_k, u_hat_k>^2 = 1 - (Phi(nu_k) + Phi_2(nu_k)) * Psi_2(w_k)
                           / (Phi(nu_k) * (Psi(w_k) + Psi_2(w_k)))

This formula is confirmed correct against Monte Carlo and against the flat-case
closed form (1 - gamma/nu^2) / (1 + gamma/nu)  (set eps == 0, eta == 1 to check).

SANITY CHECK before trusting any implementation: set eps_a == 0 for all a and
eta_l == 1 for all l. The exact procedure must then return nu_c = sqrt(gamma) and
overlap = (1 - gamma/nu_k^2) / (1 + gamma/nu_k) -- the classical BBP result.
Confirm this numerically first.

================================================================================
APPROXIMATE PROCEDURE (gamma_eff)
================================================================================

Motivation: expanding R(nu) := Psi_2(w) / Psi(w) (with w = w(nu) from the
coupling) in powers of 1/w and keeping two terms removes the need to solve for w
at all. The locus-side heteroskedasticity then enters only through two moments of
eta_l:

    mu2 = L sum_l h_l^2 / (sum_l h_l)^2
    mu3 = L^2 sum_l h_l^3 / (sum_l h_l)^3
    delta = mu3 - mu2^2      (>= 0 always)

The general definition, valid whether or not eps == 0:

    gamma_eff(nu) = gamma * mu2 * [1 + (2 delta / mu2) * gamma * Phi(nu)]

gamma_eff IS A FUNCTION OF nu, NOT A SINGLE NUMBER. Evaluate it at nu_c for the
threshold equation and at nu_k for the overlap -- using the threshold's
gamma_eff for the overlap calculation is a bug.

With gamma_eff(nu) defined, both cases below reduce to the classical Bai-Yao /
BBP formulas with gamma -> gamma_eff:

    gamma_eff(nu_c) Phi_2(nu_c) = 1                       (threshold, general)

    <u_k, u_hat_k>^2 = (1 - gamma_eff(nu_k) Phi_2(nu_k))
                       / (1 + gamma_eff(nu_k) Phi(nu_k))  (overlap, general)

Consistency check: the overlap numerator is zero exactly when the threshold
condition holds, in both cases below -- a detectable mode has positive overlap by
construction.

--- Case A: eps_a = 0 for all a ---

Phi(nu) = 1/nu, Phi_2(nu) = 1/nu^2, so gamma_eff becomes fully explicit:

    gamma_eff(nu) = gamma mu2 + 2 gamma^2 delta / nu

The threshold condition gamma_eff(nu_c) / nu_c^2 = 1 becomes the cubic

    nu_c^3 - gamma mu2 nu_c - 2 gamma^2 delta = 0

which has exactly one positive root (coefficient signs +, -, -). Closed form:

    A = 2 sqrt(gamma mu2 / 3),   Theta = 3 sqrt(3) delta sqrt(gamma) / mu2^(3/2)

    nu_c = A cos((1/3) arccos(Theta))      if Theta <= 1
           A cosh((1/3) arccosh(Theta))    if Theta > 1

IMPLEMENT BOTH BRANCHES -- arccos of an argument above 1 returns NaN, and
unfiltered sequence-scale data crosses to the cosh branch at gamma ~ 0.17; this
is expected and not a sign of a problem. The two branches meet continuously at
Theta = 1 (both give nu_c = A).

A first-order Newton step off nu^(0) = sqrt(gamma mu2) gives the simpler but less
accurate series

    nu_c ~ sqrt(gamma mu2) (1 + sqrt(gamma) kappa),   kappa = delta / mu2^(3/2)

This is a *further* approximation of the cubic, not an independent check -- it
degrades faster than the closed-form cubic root as gamma grows and should not be
used to validate the cubic. Solve the cubic (or its closed form) in the actual
implementation.

Overlap:

    <u_k, u_hat_k>^2 = (1 - gamma_eff(nu_k)/nu_k^2) / (1 + gamma_eff(nu_k)/nu_k),
        gamma_eff(nu_k) = gamma mu2 + 2 gamma^2 delta / nu_k

--- Case B: eps_a != 0 ---

No closed form; solve by fixed-point iteration (converges in ~3 passes):

    gamma_eff = gamma * mu2                      # initialize
    repeat 3 times:
        solve  gamma_eff * Phi_2(nu) = 1  for nu     [1D bisection, nu > max eps]
        gamma_eff = gamma * mu2 * (1 + 2*delta/mu2 * gamma * Phi(nu))
    nu_c = nu   (final value)

Then, for a candidate nu_k, compute gamma_eff(nu_k) from the same formula
(evaluated at nu_k, not re-using nu_c's value) and apply the general overlap
formula above.

================================================================================
ACCURACY OF THE APPROXIMATE PROCEDURE
================================================================================

Benchmarked against the exact procedure across realistic site-frequency spectra
(MAF-filtered and unfiltered; our case is the unfiltered case) and both eps
regimes, for gamma up to 0.5:

  * nu_c: error below ~0.5% at gamma <= 0.05, 1-3% at gamma up to 0.5.
  * Overlap: error below ~1% for modes with nu_k >~ 2 nu_c; degrades near
    threshold (nu_k <~ 1.5 nu_c) at large gamma (>~ 0.2), where errors up to
    ~10-15% are possible -- solve the exact system in that regime.
  * The approximation relies on eta_l having bounded support, which is guaranteed
    for genotype data since h_l <= 1/4. It is not expected to be reliable for
    weight distributions with heavy/unbounded tails (e.g. imputation-dosage
    weights) -- validate against the exact procedure if using non-genotype
    weights.

When implementing, always run the eps = 0, eta = 1 sanity check first (recovers
plain BBP), then check the approximate procedure against the exact procedure on a
simulated site-frequency spectrum before trusting either on real demographic
simulations.

================================================================================
NOTE ON ALPHA
================================================================================

alpha := (1/N) sum_k nu_k asymptotically (a normalizing constant fixed by the
demographic model, not a free parameter). nu_c and the overlap are both invariant
to it -- they depend only on the *shape* of the spike/bulk spectrum, not its
absolute scale -- so it is dropped from Phi, Phi_2 entirely when solving for nu_c
and the overlap. It re-enters only when converting a predicted eigenvalue back to
the observed scale of S = X X^T / L, via

    lambda_obs = h_bar / (1 + alpha) * (1 + nu) * w * Psi(w)

================================================================================
API
================================================================================

Exact route:   solve_w, exact_nu_c, exact_overlap
Approximate:   eta_moments, gamma_eff, approx_nu_c, approx_overlap
Finite-K:      spectrum_psi, is_detectable_spectrum  (N ~ K, stepping-stones)
Observed scale: predicted_eigenvalue

eps_a == 0 is handled as a special case of the general code path, not a separate
branch, except in approx_nu_c where the exact cubic root is used.
"""
import numpy as np
from scipy.optimize import brentq

__all__ = [
    "eta_moments", "Phi", "Phi_2", "Psi", "Psi_2",
    "solve_w", "exact_nu_c", "exact_overlap",
    "gamma_eff", "approx_nu_c", "approx_overlap",
    "predicted_eigenvalue",
    "spectrum_psi", "is_detectable_spectrum",
]

_XTOL = 1e-13
_RTOL = 8.9e-16


def _as_int(v, name):
    """Reject a non-integral loop bound, so a misplaced positional argument fails loudly."""
    if isinstance(v, (bool, np.bool_)) or v != int(v):
        raise TypeError(f"{name} must be an integer, got {v!r}")
    return int(v)


def _clean_phi_eps(phi, eps):
    """Normalize phi, drop empty populations, and broadcast a scalar eps."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    if eps.size == 1 and phi.size > 1:
        eps = np.full(phi.shape, float(eps[0]))
    if phi.shape != eps.shape:
        raise ValueError(f"phi and eps must have the same shape, got {phi.shape} and {eps.shape}")
    if np.any(phi < 0):
        raise ValueError("phi must be non-negative")
    keep = phi > 0
    if not keep.any():
        raise ValueError("phi must have at least one positive entry")
    phi, eps = phi[keep], eps[keep]
    return phi / phi.sum(), eps


def _clean_eta(h):
    """Turn per-locus variances h_l into normalized weights eta_l = h_l / mean(h)."""
    h = np.atleast_1d(np.asarray(h, dtype=float))
    h = h[h > 0]
    if h.size == 0:
        raise ValueError("h must have at least one positive entry")
    return h / h.mean()


def eta_moments(h):
    """
    Moments of the locus weights used by the gamma_eff expansion:

        mu2 = L sum h^2 / (sum h)^2,  mu3 = L^2 sum h^3 / (sum h)^3,  delta = mu3 - mu2^2.

    Equivalently the 2nd and 3rd moments of eta = h / mean(h). delta >= 0 always
    (it is the variance of eta under the eta-tilted measure). Flat h gives
    mu2 = mu3 = 1 and delta = 0.
    """
    eta = _clean_eta(h)
    mu2 = float(np.mean(eta ** 2))
    mu3 = float(np.mean(eta ** 3))
    return mu2, mu3, max(mu3 - mu2 ** 2, 0.0)


# --- the four transforms -------------------------------------------------

def Phi(nu, phi, eps):
    """Phi(nu) = sum_a phi_a (1 + eps_a) / (nu - eps_a). Strictly decreasing for nu > max eps."""
    phi, eps = _clean_phi_eps(phi, eps)
    return float(np.sum(phi * (1.0 + eps) / (nu - eps)))


def Phi_2(nu, phi, eps):
    """Phi_2(nu) = sum_a phi_a ((1 + eps_a) / (nu - eps_a))^2."""
    phi, eps = _clean_phi_eps(phi, eps)
    return float(np.sum(phi * ((1.0 + eps) / (nu - eps)) ** 2))


def Psi(w, h):
    """Psi(w) = (1/L) sum_l eta_l / (w - eta_l). Strictly decreasing for w > max eta."""
    eta = _clean_eta(h)
    return float(np.mean(eta / (w - eta)))


def Psi_2(w, h):
    """Psi_2(w) = (1/L) sum_l (eta_l / (w - eta_l))^2."""
    eta = _clean_eta(h)
    return float(np.mean((eta / (w - eta)) ** 2))


# --- exact procedure -----------------------------------------------------

def solve_w(nu, phi, eps, h, gamma, max_expand=400):
    """
    Step 1 (coupling): solve Psi(w) = gamma * Phi(nu) for w > max(eta).

    Psi is strictly decreasing from +inf at the pole w = max(eta) to 0 at
    infinity, so the root is unique whenever the target gamma*Phi(nu) is
    positive, i.e. whenever nu > max(eps). Bisection on a bracket grown outward
    from the pole; no starting guess needed.
    """
    eta = _clean_eta(h)
    target = gamma * Phi(nu, phi, eps)
    if not np.isfinite(target) or target <= 0:
        raise ValueError(f"gamma*Phi(nu) must be positive (nu must exceed max eps); got {target:g}")

    eta_max = eta.max()
    scale = max(eta_max, 1.0)

    # Walk in toward the pole until Psi > target (Psi -> +inf there).
    lo = eta_max + scale * 1e-12
    for _ in range(max_expand):
        if np.mean(eta / (lo - eta)) > target:
            break
        lo = eta_max + (lo - eta_max) / 2.0
    else:
        raise RuntimeError("could not find a point with Psi > target next to the pole")

    # Grow the upper end until Psi < target; Psi -> 0 at infinity.
    hi = eta_max + 2.0 * scale
    for _ in range(max_expand):
        if np.mean(eta / (hi - eta)) < target:
            break
        hi = eta_max + 2.0 * (hi - eta_max)
    else:
        raise RuntimeError("could not bracket w from above")

    f = lambda w: float(np.mean(eta / (w - eta))) - target
    return float(brentq(f, lo, hi, xtol=_XTOL, rtol=_RTOL))


def _exact_lhs(nu, phi, eps, h, gamma):
    """(Phi_2/Phi)(nu) * (Psi_2/Psi)(w(nu)); equals 1 exactly at nu_c."""
    w = solve_w(nu, phi, eps, h, gamma)
    return (Phi_2(nu, phi, eps) / Phi(nu, phi, eps)) * (Psi_2(w, h) / Psi(w, h))


def exact_nu_c(phi, eps, h, gamma, max_expand=400):
    """
    Step 2: the critical value nu_c, solving (Phi_2/Phi)(nu) * (Psi_2/Psi)(w(nu)) = 1
    jointly with the Step 1 coupling. Mode k is detectable iff nu_k > nu_c.

    The left-hand side decreases from +inf at the pole nu -> max(eps)+ to 0 at
    infinity, so the root in (max eps, inf) is unique; nested bisection (outer on
    nu, inner on w) converges without a starting guess.
    """
    phi, eps = _clean_phi_eps(phi, eps)
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma:g}")
    if np.ndim(h) == 0:
        raise TypeError("exact_nu_c takes the per-locus weights h as its third argument, "
                        "not a scalar; for the (gamma, mu2, delta) route use approx_nu_c")
    max_expand = _as_int(max_expand, "max_expand")
    eps_max = eps.max()
    scale = max(abs(eps_max), np.sqrt(gamma), 1.0)

    lo = eps_max + scale * 1e-10
    for _ in range(max_expand):
        if _exact_lhs(lo, phi, eps, h, gamma) > 1.0:
            break
        lo = eps_max + (lo - eps_max) / 2.0
    else:
        raise RuntimeError("could not find a point with LHS > 1 next to the pole")

    hi = eps_max + 2.0 * scale
    for _ in range(max_expand):
        if _exact_lhs(hi, phi, eps, h, gamma) < 1.0:
            break
        hi = eps_max + 2.0 * (hi - eps_max)
    else:
        raise RuntimeError("could not bracket nu_c from above")

    f = lambda nu: _exact_lhs(nu, phi, eps, h, gamma) - 1.0
    return float(brentq(f, lo, hi, xtol=_XTOL, rtol=_RTOL))


def exact_overlap(nu_k, phi, eps, h, gamma, nu_c=None):
    """
    Step 3: squared eigenvector overlap <u_k, u_hat_k>^2 for a detectable nu_k,

        1 - (Phi + Phi_2)(nu_k) * Psi_2(w_k) / (Phi(nu_k) * (Psi + Psi_2)(w_k)).

    Returns 0.0 for a mode at or below threshold (no consistent recovery). Pass a
    precomputed nu_c to avoid re-solving it across a sweep.
    """
    if nu_c is None:
        nu_c = exact_nu_c(phi, eps, h, gamma)
    if nu_k <= nu_c:
        return 0.0
    w = solve_w(nu_k, phi, eps, h, gamma)
    p1, p2 = Phi(nu_k, phi, eps), Phi_2(nu_k, phi, eps)
    s1, s2 = Psi(w, h), Psi_2(w, h)
    return float(1.0 - (p1 + p2) * s2 / (p1 * (s1 + s2)))


# --- approximate procedure (gamma_eff) -----------------------------------

def gamma_eff(nu, phi, eps, gamma, mu2, delta):
    """
    gamma_eff(nu) = gamma * mu2 * (1 + (2 delta / mu2) * gamma * Phi(nu)).

    A function of nu, not a constant: evaluate it at nu_c for the threshold and
    at nu_k for the overlap. Reusing the threshold's value for the overlap is a bug.
    """
    return gamma * mu2 * (1.0 + (2.0 * delta / mu2) * gamma * Phi(nu, phi, eps))


def _cubic_nu_c(gamma, mu2, delta):
    """
    Positive root of nu^3 - gamma*mu2*nu - 2*gamma^2*delta = 0, the threshold when
    eps == 0 (where Phi = 1/nu, Phi_2 = 1/nu^2). Trigonometric solution, with the
    cosh branch for Theta > 1 (reached by unfiltered SFS data around gamma ~ 0.17).
    """
    A = 2.0 * np.sqrt(gamma * mu2 / 3.0)
    Theta = 3.0 * np.sqrt(3.0) * delta * np.sqrt(gamma) / mu2 ** 1.5
    if Theta <= 1.0:
        return float(A * np.cos(np.arccos(Theta) / 3.0))
    return float(A * np.cosh(np.arccosh(Theta) / 3.0))


def approx_nu_c(phi, eps, gamma, mu2, delta, n_iter=3, max_expand=400):
    """
    Threshold from the gamma_eff expansion: solves gamma_eff(nu_c) * Phi_2(nu_c) = 1.

    General case: fixed-point iteration, alternating a 1D bisection for nu at the
    current gamma_eff with a refresh of gamma_eff at that nu (~3 passes suffice).
    When all eps_a are zero this reduces to the cubic, which is solved in closed
    form instead of iterated.
    """
    phi, eps = _clean_phi_eps(phi, eps)
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma:g}")
    if np.allclose(eps, 0.0):
        return _cubic_nu_c(gamma, mu2, delta)

    eps_max = eps.max()
    scale = max(abs(eps_max), np.sqrt(gamma * mu2), 1.0)
    ge = gamma * mu2
    nu = eps_max + scale
    for _ in range(n_iter):
        # Phi_2 falls from +inf at the pole to 0, so ge*Phi_2 = 1 has a unique root.
        lo = eps_max + scale * 1e-12
        for _ in range(max_expand):
            if ge * Phi_2(lo, phi, eps) > 1.0:
                break
            lo = eps_max + (lo - eps_max) / 2.0
        else:
            raise RuntimeError("could not find a point with gamma_eff*Phi_2 > 1 next to the pole")
        hi = eps_max + 2.0 * scale
        for _ in range(max_expand):
            if ge * Phi_2(hi, phi, eps) < 1.0:
                break
            hi = eps_max + 2.0 * (hi - eps_max)
        else:
            raise RuntimeError("could not bracket nu_c from above")
        nu = float(brentq(lambda v: ge * Phi_2(v, phi, eps) - 1.0, lo, hi,
                          xtol=_XTOL, rtol=_RTOL))
        ge = gamma_eff(nu, phi, eps, gamma, mu2, delta)
    return nu


def approx_overlap(nu_k, phi, eps, gamma, mu2, delta, nu_c=None):
    """
    Overlap from the gamma_eff expansion:

        (1 - gamma_eff(nu_k) Phi_2(nu_k)) / (1 + gamma_eff(nu_k) Phi(nu_k)),

    i.e. the classical Bai-Yao form with gamma -> gamma_eff(nu_k). Returns 0.0 at
    or below threshold; the numerator vanishes there by construction, so the two
    conventions agree.
    """
    if nu_c is None:
        nu_c = approx_nu_c(phi, eps, gamma, mu2, delta)
    if nu_k <= nu_c:
        return 0.0
    ge = gamma_eff(nu_k, phi, eps, gamma, mu2, delta)
    num = 1.0 - ge * Phi_2(nu_k, phi, eps)
    den = 1.0 + ge * Phi(nu_k, phi, eps)
    return float(max(num / den, 0.0))


# --- back to the observed scale ------------------------------------------

def predicted_eigenvalue(nu_k, phi, eps, h, gamma, alpha=0.0):
    """
    The outlier eigenvalue of S = XX^T / L predicted for a spike nu_k:

        lambda_obs = h_bar / (1 + alpha) * (1 + nu_k) * w * Psi(w),  w = w(nu_k).

    This is the only place alpha and the scale of h enter; nu_c and the overlaps
    are invariant to both.
    """
    h_bar = float(np.mean(np.asarray(h, dtype=float)))
    w = solve_w(nu_k, phi, eps, h, gamma)
    return float(h_bar / (1.0 + alpha) * (1.0 + nu_k) * w * Psi(w, h))


# --- finite-K (N ~ K) criterion ------------------------------------------

def spectrum_psi(k, spectrum, L, degenerate_subspace=False, rtol=1e-8):
    """
    Psi_spec(nu_k) = (1/L) sum_{j != k} ((1 + nu_j) / (nu_k - nu_j))^2.

    The N ~ K analogue of the threshold above. Phi_2 weights K population-level
    floor levels by phi_a, which presumes the floor is an overwhelming majority of
    the spectrum; in the stepping-stone models K is comparable to N, no small set
    of levels stands in for the rest, and every other eigenvalue must be counted
    individually. Detectable iff Psi_spec < 1.

    Degeneracy is opt-in because the two cases pull opposite ways. A degenerate
    *structure* eigenvalue (2D stepping-stones give exactly degenerate pairs)
    spans a subspace that is detectable as a subspace even though PCA cannot fix a
    basis inside it, so its copies belong to the candidate, not to the competition.
    A *floor* eigenvalue's degeneracy is the bulk itself, and dropping its copies
    would report the floor as detectable. The default follows the literal j != k
    definition, which keeps floor eigenvalues undetectable (Psi -> inf on a tie);
    pass degenerate_subspace=True only for a known structure eigenvalue.

    rtol must be loose enough for the ~1e-15 spread that diagonalizing a symmetric
    demography leaves on an exact degeneracy, and far below any real spacing.

    Returns +inf if an eigenvalue coincides with nu_k and is not excluded, 0.0 if
    nothing remains to sum over.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 1:
        raise ValueError(f"spectrum must be 1-D, got shape {spectrum.shape}")
    if L <= 0:
        raise ValueError(f"L must be positive, got {L:g}")
    if rtol < 0:
        raise ValueError(f"rtol must be non-negative, got {rtol:g}")

    k = int(k)
    if not -spectrum.size <= k < spectrum.size:
        raise IndexError(f"k={k} out of range for a spectrum of size {spectrum.size}")
    k %= spectrum.size

    nu_k = spectrum[k]
    others = np.delete(spectrum, k)
    if others.size == 0:
        return 0.0

    if degenerate_subspace:
        atol = rtol * max(abs(nu_k), 1.0)
        others = others[np.abs(others - nu_k) > atol]
        if others.size == 0:
            return 0.0

    denom = nu_k - others
    if np.any(denom == 0.0):
        return np.inf
    return float(np.sum(((1.0 + others) / denom) ** 2) / L)


def is_detectable_spectrum(k, spectrum, L, degenerate_subspace=False, rtol=1e-8):
    """Whether spectrum[k] separates from the rest, by the finite-K criterion Psi_spec < 1."""
    return spectrum_psi(k, spectrum, L,
                        degenerate_subspace=degenerate_subspace, rtol=rtol) < 1.0
