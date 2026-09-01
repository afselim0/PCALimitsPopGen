#%%
"""
The theoretical half of a sweep summary: structure eigenvalue, three critical values, and
three crossings, per swept demographic parameter value.

Kept out of summarise.py so the three-route RMT logic can be exercised without rpy2/R.

Two quantities per curve are called "critical" and must not be conflated:

    x_c      the swept value at which nu(x) = nu_c -- the DETECTABILITY threshold. Below it
             the eigenvalue has not separated from the bulk and cos^2 is exactly 0.
    x_cross  the swept value at which the predicted cos^2 reaches `target` (0.9) -- the
             USABILITY threshold, the number the figures mark.

x_c < x_cross always. Both are properties of the curve, constant down a theory.csv block;
only nu varies with x.

Each route carries its own nu_c through to its own overlap: the routes are three different
approximations of the same quantity, and pairing one route's threshold with another's
overlap gives a curve that is neither. See src/threshold_and_overlap.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis"))  # src/analysis/

from crossing_fit import DEFAULT_TARGET, theory_crossing
from threshold_and_overlap import (approx_nu_c, approx_overlap, crude_nu_c, crude_overlap,
                                   exact_nu_c, exact_overlap)

ROUTES = ("exact", "approx", "crude")


def _finite(v):
    """float(v) when it is a real number, else NaN -- CSV round-trips turn absents into NaN."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def nu_c_all(tc):
    """
    The three critical values for one demography: {"nu_c_exact", "nu_c_approx", "nu_c_crude"}.

    None depends on the swept parameter -- that is the whole reason theory_context carries
    no nu. A route that cannot be evaluated yields NaN for itself alone: without the
    per-locus weights h the exact route is unavailable while the other two still stand.
    """
    out = {}
    phi, eps = tc["phi"], tc["eps"]
    h, gamma_raw = tc.get("h"), tc["gamma_raw"]

    try:
        out["nu_c_exact"] = (float(exact_nu_c(phi, eps, h, gamma_raw))
                             if h is not None and len(np.asarray(h)) else np.nan)
    except (ValueError, RuntimeError, TypeError):
        out["nu_c_exact"] = np.nan

    try:
        out["nu_c_approx"] = float(approx_nu_c(phi, eps, gamma_raw,
                                               tc["mu2"], tc["delta"]))
    except (ValueError, RuntimeError, TypeError):
        out["nu_c_approx"] = np.nan

    try:
        out["nu_c_crude"] = float(crude_nu_c(tc["N"], tc["L_eff"]))
    except (ValueError, RuntimeError, TypeError):
        out["nu_c_crude"] = np.nan

    return out


def overlap_fn(tc, route, nu_c=None):
    """nu -> predicted cos^2 by one route, or None when that route is unavailable."""
    phi, eps = tc["phi"], tc["eps"]
    gamma_raw = tc["gamma_raw"]

    if route == "exact":
        h = tc.get("h")
        if h is None or not len(np.asarray(h)):
            return None
        return lambda nu: exact_overlap(nu, phi, eps, h, gamma_raw, nu_c=nu_c)
    if route == "approx":
        mu2, delta = tc["mu2"], tc["delta"]
        if not (np.isfinite(mu2) and np.isfinite(delta)):
            return None
        return lambda nu: approx_overlap(nu, phi, eps, gamma_raw, mu2, delta, nu_c=nu_c)
    if route == "crude":
        N, L_eff = tc["N"], tc["L_eff"]
        if not (np.isfinite(N) and np.isfinite(L_eff)) or L_eff <= 0:
            return None
        return lambda nu: crude_overlap(nu, N, L_eff)
    raise ValueError(f"unknown route {route!r}; expected one of {ROUTES}")


def x_at_nu(nu_fn, nu_target, x_lo, x_hi, n_grid=400):
    """
    Invert nu(x) = nu_target: the swept value at which the eigenvalue reaches the threshold.

    nu(x) is affine and increasing in every model that keeps theory (linear for split and
    the stepping-stones, affine with an eps_0 pedestal for african_split), so the root is
    unique. Brackets on a log grid then bisects, mirroring crossing_fit.theory_crossing --
    the grid step also keeps a non-monotone or failing nu_fn from being silently inverted.

    Returns NaN when nu_target lies outside [nu(x_lo), nu(x_hi)], which is the honest
    answer when the transition sits outside the swept range.
    """
    from scipy.optimize import brentq

    if not np.isfinite(nu_target):
        return np.nan
    grid = np.geomspace(x_lo, x_hi, n_grid)
    vals = np.array([_safe_call(nu_fn, xx) for xx in grid])
    above = np.flatnonzero(np.isfinite(vals) & (vals >= nu_target))
    if above.size == 0:
        return np.nan
    j = above[0]
    if j == 0:
        return np.nan  # already above threshold at the left edge: outside the swept range
    f = lambda xx: _safe_call(nu_fn, xx) - nu_target
    try:
        return float(brentq(f, grid[j - 1], grid[j], xtol=1e-14, rtol=1e-12))
    except (ValueError, RuntimeError):
        return float(grid[j])


def _safe_call(fn, x):
    """Evaluate nu(x), mapping a numerical failure to NaN rather than raising."""
    try:
        v = float(fn(x))
    except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def x_cross_all(nu_fn, tc, x_lo, x_hi, target=DEFAULT_TARGET, nu_cs=None):
    """
    Per route: x_c (where nu = nu_c) and x_cross (where predicted cos^2 = target).

    Returns a flat dict of x_c_<route> and x_cross_<route>. Each route is inverted with its
    own nu_c, never a shared one.
    """
    nu_cs = nu_c_all(tc) if nu_cs is None else nu_cs
    out = {}
    for route in ROUTES:
        nu_c = nu_cs.get(f"nu_c_{route}", np.nan)
        out[f"x_c_{route}"] = x_at_nu(nu_fn, nu_c, x_lo, x_hi)
        fn = overlap_fn(tc, route, nu_c=nu_c)
        out[f"x_cross_{route}"] = (
            theory_crossing(lambda x: fn(nu_fn(x)), x_lo, x_hi, target=target)
            if fn is not None else np.nan)
    return out


def theory_rows(sweep, level, level_param, x_key, nu_fn, tc, x_values,
                target=DEFAULT_TARGET, nu_form="", extra=None):
    """
    One theory row per swept value for a single curve.

    nu varies down the block; the nu_c, x_c and x_cross columns are constants of the curve
    and repeat, which is what lets a reader group by (sweep, level) and take .first().
    """
    x_values = np.asarray(sorted(float(v) for v in np.unique(x_values)), dtype=float)
    if x_values.size == 0:
        return []
    x_lo, x_hi = float(x_values[0]), float(x_values[-1])

    nu_cs = nu_c_all(tc)
    crossings = x_cross_all(nu_fn, tc, x_lo, x_hi, target=target, nu_cs=nu_cs)
    fns = {r: overlap_fn(tc, r, nu_c=nu_cs.get(f"nu_c_{r}", np.nan)) for r in ROUTES}

    has_h = tc.get("h") is not None and len(np.asarray(tc.get("h", [])))
    rows = []
    for x in x_values:
        nu = _safe_call(nu_fn, x)
        row = dict(sweep=sweep, level=level, level_param=level_param, x_key=x_key,
                   x=float(x), nu=nu)
        row[x_key] = float(x)
        row.update(nu_cs)
        row.update(crossings)
        for r in ROUTES:
            row[f"cos2_{r}"] = (_safe_call(fns[r], nu)
                                if fns[r] is not None and np.isfinite(nu) else np.nan)
        row.update(target=target, nu_form=nu_form, has_h=bool(has_h),
                   N=tc.get("N"), L_eff=tc.get("L_eff"), gamma_raw=tc.get("gamma_raw"),
                   mu2=tc.get("mu2"), delta=tc.get("delta"))
        if extra:
            row.update(extra)
        rows.append(row)
    return rows

# %%
