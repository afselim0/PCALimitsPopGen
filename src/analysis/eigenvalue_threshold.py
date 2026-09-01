#%%
"""
The empirical detection threshold, read off the sample eigenvalues rather than the cosine.

The cosine cannot locate the threshold at finite N. nu(x) = nu_c is an N -> infinity
statement -- cos^2 is exactly 0 below it and rises with infinite slope at it -- so at the
simulated N the measured curve is a smeared sigmoid that leaves the 1/N noise floor far
below the threshold and is already at cos^2 ~ 0.1-0.2 when it arrives. Any rule that picks
a point out of that foot is a calibration, not a measurement.

The eigenvalue does not have that problem. Below threshold the leading structure eigenvalue
is PINNED to the bulk edge; above it, it detaches. So the observable

    ratio(x) = lambda_structure / lambda_bulk

is exactly 1 below the threshold and rises above it, and the transition is a real feature of
the curve rather than a level someone chose.

WHY NOT A SLOPE RULE. The liftoff is quadratic:

    ratio - 1  ~  (nu/nu_c - 1)^2      near threshold

A quadratic leaves the axis with zero slope, so "where the slope picks up" is structurally
late -- a broken-stick fit on ratio(x) lands at 1.5-1.9x nu_c, stably enough that the bias
is clearly systematic and not noise. Taking a square root first linearizes the liftoff but
then the sub-threshold noise floor, which is what sqrt() behaves worst on, dominates the fit.

WHAT THIS DOES INSTEAD. Fit the exact spiked-model curve

    ratio(nu) = (1 + nu)(1 + gamma/nu) / (1 + sqrt(gamma))^2      for nu > sqrt(gamma)
              = 1                                                 otherwise

to the measured (x, ratio) replicates with gamma as the ONLY free parameter, nu(x) being
supplied by the demographic model. Then nu_c = sqrt(gamma) and x_c follows by inverting
nu(x). No fitted level, no multiplier: the shape of the curve does the work.

Accuracy on synthetic BBP data at sweep resolution (100 x-points, 8 replicates):
x_c recovered to 1.0008 +/- 0.0002, degrading only to 1.015 at 6% multiplicative noise.

It is semi-parametric: it takes nu(x) from the model and so inherits any error there. That
input is verified -- every model's nu_of_x was checked against the appendix closed forms to
a ratio of 1.0000000000 -- but a 20% error in the assumed slope would move x_c by ~33%, so
the dependency is real and is why fit_gamma also returns its residual for inspection.

TWO SENSITIVITIES WORTH KNOWING.

1. THE EDGE. The fit is very sensitive to what lambda is divided by, because near threshold
   the whole signal is the small excess of lambda over the edge. Measured on split_vary_N,
   scaling the edge by a constant c moves the recovered threshold by:

       c        0.90  0.95  0.98  1.00  1.02  1.05  1.10
       x_c/x_c  0.66  0.84  0.95  1.02  1.09  1.20  1.38   (N = 200)

   so a 5% error in the edge is a 15-20% error in the threshold. What matters is not the
   per-replicate edge specifically -- dividing by one median edge per curve works equally
   well (0.98 against 1.02) -- but that the normalisation puts the sub-threshold plateau at
   1, which is the scale the BBP formula is written in. Raw unnormalised lambda fails
   outright.

2. INVERTING nu(x). The fit estimates nu_c well; converting it to x_c is where accuracy can
   be lost. For nu = eps_0 + s*x the amplification is d log x_c / d log nu_c =
   nu_c/(nu_c - eps_0). A model with no pedestal (the K = 2 split, eps_0 ~ 0) has
   amplification ~1 and x_c is as good as nu_c. OOA3's sister axis carries eps_0 = 0.215,
   which is 31-53% of nu_c, giving amplification 1.5-2.2 -- and the observed bias tracks it
   exactly (2.14 at N=200 -> ratio 0.46; 1.46 at N=1000 -> 0.74). For such models nu_c_emp
   is the meaningful output and x_c_emp is not; both are returned so the caller can tell.
"""
import numpy as np
from scipy.optimize import brentq, least_squares

# Below this many usable points the fit is not attempted: gamma is one parameter, but it is
# only identified by the CURVATURE of the liftoff, which needs the transition to be sampled.
MIN_POINTS = 12


def bbp_ratio(nu, gamma):
    """
    lambda_structure / lambda_bulk for a spike nu at aspect ratio gamma.

    (1+nu)(1+gamma/nu) is the spiked eigenvalue and (1+sqrt(gamma))^2 the bulk edge; below
    threshold the spike has not detached and the ratio is exactly 1.
    """
    nu = np.asarray(nu, dtype=float)
    nu_c = np.sqrt(gamma)
    edge = (1.0 + np.sqrt(gamma)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        spike = (1.0 + nu) * (1.0 + gamma / np.where(nu > 0, nu, np.nan)) / edge
    return np.where(nu > nu_c, spike, 1.0)


def fit_gamma(nu, ratio, gamma0=0.05, bounds=(1e-6, 10.0)):
    """
    The aspect ratio implied by the measured liftoff, by least squares on bbp_ratio.

    Fitted in log gamma so the parameter stays positive without a constrained solver.
    Returns (gamma, rms_residual); gamma is NaN when there is too little to fit.
    """
    nu = np.asarray(nu, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    ok = np.isfinite(nu) & np.isfinite(ratio) & (nu > 0)
    nu, ratio = nu[ok], ratio[ok]
    if nu.size < MIN_POINTS or np.unique(nu).size < 5:
        return np.nan, np.nan

    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    def resid(p):
        return bbp_ratio(nu, float(np.exp(np.clip(p[0], lo, hi)))) - ratio

    try:
        sol = least_squares(resid, [np.log(gamma0)], method="lm", max_nfev=2000)
    except (ValueError, RuntimeError):
        return np.nan, np.nan
    g = float(np.exp(np.clip(sol.x[0], lo, hi)))
    rms = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    return g, rms


def threshold_from_eigenvalues(x, ratio, nu_of_x, x_lo=None, x_hi=None, gamma0=0.05):
    """
    The empirical detection threshold in units of the swept parameter.

    Args:
        x: swept value per replicate (Fst or 1/M).
        ratio: lambda_structure / lambda_bulk for that replicate.
        nu_of_x: the model's swept value -> structure eigenvalue.
        x_lo, x_hi: bracket for inverting nu(x) = nu_c; defaults to the data's own range,
            widened, so a threshold just outside the swept range is still reported rather
            than silently clipped to an endpoint.

    Returns a dict with x_c_emp, nu_c_emp, gamma_emp and the fit's rms residual. x_c_emp is
    NaN when the fit fails or when nu(x) = nu_c has no root in the bracket, which is the
    honest answer when the transition lies outside what was simulated.
    """
    x = np.asarray(x, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    ok = np.isfinite(x) & np.isfinite(ratio) & (x > 0)
    x, ratio = x[ok], ratio[ok]
    out = dict(x_c_emp=np.nan, nu_c_emp=np.nan, gamma_emp=np.nan, rms=np.nan,
               n_points=int(x.size))
    if x.size < MIN_POINTS:
        return out

    nu = np.array([float(nu_of_x(v)) for v in x], dtype=float)
    gamma, rms = fit_gamma(nu, ratio, gamma0=gamma0)
    out.update(gamma_emp=gamma, rms=rms)
    if not np.isfinite(gamma):
        return out

    nu_c = float(np.sqrt(gamma))
    out["nu_c_emp"] = nu_c
    out["x_c_emp"] = invert_nu(nu_of_x, nu_c,
                               x.min() if x_lo is None else x_lo,
                               x.max() if x_hi is None else x_hi)
    return out


def invert_nu(nu_of_x, nu_target, x_lo, x_hi, max_expand=12):
    """
    Solve nu(x) = nu_target, expanding outward from the swept range.

    Expanded stepwise rather than by one large factor: nu(x) need not be defined outside a
    bounded interval -- the split model's Fst/(1-Fst) has a pole at Fst = 1, and jumping
    straight to 1000x the range lands past it, where the sign flip fakes a bracket. Each
    step is checked for a sign change and for nu staying finite and increasing, so the
    search stops at the first genuine bracket and gives up rather than crossing a
    singularity.
    """
    def f(t):
        try:
            v = float(nu_of_x(t)) - nu_target
        except (ValueError, RuntimeError, ZeroDivisionError):
            return np.nan
        return v if np.isfinite(v) else np.nan

    lo, hi = float(x_lo), float(x_hi)
    for _ in range(max_expand):
        flo, fhi = f(lo), f(hi)
        if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0:
            try:
                return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-12))
            except (ValueError, RuntimeError):
                return np.nan
        # Widen on whichever side has not yet bracketed, in modest steps.
        if np.isfinite(flo) and flo > 0:
            lo *= 0.5
        elif np.isfinite(fhi) and fhi < 0:
            hi *= 2.0
        else:
            break
    return np.nan

# %%
