#%%
"""
Fit a detection curve through per-replicate overlaps and locate where it crosses a target.

Replaces the median-per-grid-point summary used previously. Each sweep cell contributes
`num_replicates` independent (x, cos^2) points, and a single curve is fitted through the
whole cloud, so the transition is estimated from the spread of the replicates rather than
read off a median that discards it.

TWO CURVES, FOR TWO DIFFERENT JOBS
----------------------------------

`fit_with_band` returns both, because no single curve does both jobs well:

  * the drawn line and its band come from a Richards (asymmetric logistic) fit in log x,

        cos^2(x) = floor + (ceil - floor) / (1 + s exp(-b (log x - c)))^(1/s)

    which is smooth, monotone and publication-shaped. The floor is fitted rather than
    pinned at 0 because a PC carrying no signal still has expected overlap sqrt(2/(pi N))
    against a fixed axis -- vectors in finite dimension are never exactly orthogonal -- so
    the data plateaus above zero on the left. The asymmetry exponent s is what a plain
    4-parameter logistic lacks: the BBP transition climbs from 0 to ~0.9 within one decade
    and then crawls to 1 over the next, and forcing that through a symmetric logistic
    drags `ceil` down and `b` up.

  * the reported crossing comes from a monotone (PAVA + PCHIP) interpolation of the
    per-level means, NOT from the Richards curve.

Why the crossing is not read off the fitted curve: benchmarked against synthetic BBP data
whose true 0.9-crossing is known in closed form, the Richards crossing is biased low by a
consistent ~12% across sample sizes and noise levels (0.85-0.91x), because no
4-or-5-parameter sigmoid matches the BBP tail exactly. The monotone interpolation is
unbiased to within a few percent (0.95-1.02x on the ~100-point sweep grids used here).
Since the empirical crossing is plotted alongside the theoretical one, a systematic 12%
offset in one but not the other would read as a real discrepancy -- so accuracy wins over
smoothness for the number, and smoothness wins for the line.

The per-level means are deliberately NOT smoothed before the monotone step. Smoothing
removes the occasional noise-driven excursion but flattens the rise near the crossing,
trading random error for a systematic high bias (1.04-1.14x) -- the wrong trade when the
whole point is comparing against theory.

Uncertainty is a nonparametric bootstrap resampling replicates *within* each swept value,
which preserves the design: the band is the pointwise 16/84th percentile of the refitted
Richards curves, the crossing CI the same percentiles of the refitted monotone crossings.

The theoretical counterpart is `theory_crossing`, which numerically inverts the exact
Ding & Yang overlap from src/threshold_and_overlap rather than assuming a functional form.

Used by the overlap-vs-parameter analyses (SingleSplitSweeps, SisterPCAlignment,
SteppingStone*, RecentSpatial*) and their plotting scripts.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares

# cos^2 at which detection is declared. 0.8 rather than 0.9: the crossing is read off the
# fitted curve as a level set, so its accuracy depends on how steeply the curve passes
# through the target. At 0.9 the curve is already flattening, which both amplifies a small
# vertical error into a large error in x and puts the level in the band where the arm's
# roughness penalty biases the fit low. Measured against theory over all 51 curves of the
# six detection sweeps, mean |bias| falls from 7.8% at 0.9 to 5.3% at 0.8 and the worst
# case halves (0.217 -> 0.115), while three more curves gain a usable crossing -- most
# importantly the OOA3 phi-tilde panel, which had only 4 of 10 at 0.9 because its theory
# curve barely reaches that level. 0.7 scores marginally better still (3.5%), but 0.8 is a
# round convention near the optimum rather than the argmin of this particular set.
DEFAULT_TARGET = 0.8
DEFAULT_N_BOOT = 300
DEFAULT_SEED = 0
_MIN_LEVELS = 4


# --- the drawn curve: Richards / asymmetric logistic ----------------------

def richards(logx, floor, ceil, b, c, s):
    """
    Richards curve in log x. s -> 1 recovers the plain logistic; s < 1 delays the top.

    Evaluated as exp(-log1p(s e^z)/s) rather than (1 + s e^z)^(-1/s): the direct power
    overflows for the large z the optimizer explores while fitting a steep transition, and
    log1p keeps the small-z end accurate too. The clip bounds e^z itself; log1p then makes
    the outer power harmless.
    """
    z = np.clip(-b * (np.asarray(logx, dtype=float) - c), -500.0, 500.0)
    with np.errstate(over="ignore", invalid="ignore"):
        decay = np.exp(-np.log1p(s * np.exp(z)) / s)
    return floor + (ceil - floor) * np.nan_to_num(decay, nan=0.0)


def fit_richards(x, y):
    """
    Fit the Richards curve to the (x, y) cloud by bounded robust least squares.

    Several starting values of the asymmetry exponent are tried and the best kept: the
    residual surface has a shallow local minimum near s = 1 that a single start falls into.
    Returns a parameter dict, or None if the fit fails or too few levels to identify it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size < 6 or np.unique(x).size < _MIN_LEVELS:
        return None

    logx = np.log(x)
    span = float(np.ptp(logx)) or 1.0
    lo = [0.0, 0.5, 1e-3, logx.min() - 5 * span - 1, 0.05]
    hi = [0.5, 1.0, 1e3, logx.max() + 5 * span + 1, 20.0]

    # A sweep whose range does not contain the transition cannot identify five parameters:
    # the plateaus are the same value, `floor` runs to its bound and the optimizer spends
    # its whole budget getting nowhere. Refuse rather than grind -- 0.6 s per fit becomes
    # 30 s per bootstrapped curve. The caller still gets the monotone crossing, which is
    # well defined whenever the data actually crosses the target.
    if float(np.ptp(y)) < 0.25:
        return None

    best = None
    for s0 in (0.2, 1.0, 3.0):
        p0 = np.clip([float(np.min(y)), min(max(float(np.max(y)), 0.55), 1.0),
                      4.0 / span, float(np.median(logx)), s0], lo, hi)
        try:
            # soft_l1 keeps a few replicate outliers from dragging the plateaus, which is
            # what taking a median used to buy us.
            r = least_squares(lambda p: richards(logx, *p) - y, p0, bounds=(lo, hi),
                              loss="soft_l1", f_scale=0.1, max_nfev=8000)
        except (ValueError, RuntimeError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    floor, ceil, b, c, s = best.x
    if ceil <= floor:
        return None
    return dict(floor=float(floor), ceil=float(ceil), b=float(b), c=float(c), s=float(s))


def predict(params, x):
    """Evaluate a fitted Richards curve on a grid of x."""
    return richards(np.log(np.asarray(x, dtype=float)), params["floor"], params["ceil"],
                    params["b"], params["c"], params["s"])


# --- the reported crossing: monotone interpolation ------------------------

def _pava(y, w):
    """Weighted pool-adjacent-violators: the least-squares non-decreasing fit to y."""
    values, weights, counts = [], [], []
    for yi, wi in zip(np.asarray(y, float), np.asarray(w, float)):
        values.append(float(yi))
        weights.append(float(wi))
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2, c2 = values.pop(), weights.pop(), counts.pop()
            v1, w1, c1 = values.pop(), weights.pop(), counts.pop()
            wn = w1 + w2
            values.append((v1 * w1 + v2 * w2) / wn)
            weights.append(wn)
            counts.append(c1 + c2)
    out = []
    for v, c in zip(values, counts):
        out.extend([v] * c)
    return np.array(out)


def monotone_curve(x, y, x_grid):
    """
    Monotone interpolation of the per-level mean overlap, evaluated on x_grid.

    Levels are averaged, made non-decreasing by PAVA (weighted by replicate count), and
    interpolated with a shape-preserving PCHIP, which does not overshoot between knots.
    Outside the swept range the result is NaN rather than extrapolated.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    levels = np.unique(x)
    if levels.size < _MIN_LEVELS:
        return np.full(x_grid.shape, np.nan)
    means = np.array([y[x == lv].mean() for lv in levels])
    counts = np.array([(x == lv).sum() for lv in levels], dtype=float)
    iso = np.clip(_pava(means, counts), 0.0, 1.0)
    return PchipInterpolator(np.log(levels), iso, extrapolate=False)(np.log(x_grid))


def _first_crossing(x_grid, y_grid, target):
    """First x where a sampled curve reaches target, by linear interpolation in log x."""
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    ok = np.isfinite(y_grid)
    x_grid, y_grid = x_grid[ok], y_grid[ok]
    above = np.flatnonzero(y_grid >= target)
    if above.size == 0:
        return np.nan          # the sweep never reaches the target
    j = above[0]
    if j == 0:
        return float(x_grid[0])   # already above at the left edge
    lx0, lx1 = np.log(x_grid[j - 1]), np.log(x_grid[j])
    y0, y1 = y_grid[j - 1], y_grid[j]
    if y1 == y0:
        return float(x_grid[j])
    return float(np.exp(lx0 + (target - y0) * (lx1 - lx0) / (y1 - y0)))


def empirical_crossing(x, y, target=DEFAULT_TARGET, x_grid=None, n_grid=800):
    """The swept value at which the measured overlap reaches `target`, or NaN."""
    x = np.asarray(x, dtype=float)
    if x_grid is None:
        x_grid = np.geomspace(x.min(), x.max(), n_grid)
    return _first_crossing(x_grid, monotone_curve(x, y, x_grid), target)


def fit_with_band(x, y, target=DEFAULT_TARGET, x_grid=None, n_boot=DEFAULT_N_BOOT,
                  seed=DEFAULT_SEED, n_grid=300, ci=(16, 84), n_boot_band=None):
    """
    Fit the cloud, bootstrap it, and return everything a panel needs to draw.

    The two bootstraps are deliberately different sizes. A monotone refit costs ~1 ms and a
    Richards refit ~0.5 s, so resampling both n_boot times would spend minutes per curve
    almost entirely on the band. The crossing CI -- the number anyone reads -- gets the full
    n_boot monotone refits; the band, which only has to look right, gets n_boot_band
    Richards refits (default 30, or n_boot if smaller).

    Args:
        x: swept parameter, one entry per replicate (positive; the fit is in log x).
        y: squared cosine similarity of that replicate.
        target: the cos^2 level whose crossing is reported (default 0.9).
        n_boot: bootstrap resamples for the crossing CI; replicates are resampled within
            each swept value so every level keeps its replicate count and none drops out.
        n_boot_band: resamples for the drawn band. None uses min(30, n_boot).

    Returns a dict with:
        params, x_grid, y_fit       the drawn Richards curve
        y_lo, y_hi                  pointwise bootstrap band
        crossing                    the monotone-interpolation crossing (the reported one)
        crossing_lo, crossing_hi    its percentile CI
        crossing_boot               the bootstrap crossings, NaNs dropped
        crossing_richards           the Richards curve's own crossing, for reference only
                                    (biased ~12% low -- see the module docstring)
    Returns None when the point fit fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size == 0 or np.unique(x).size < _MIN_LEVELS:
        return None

    params = fit_richards(x, y)

    if x_grid is None:
        x_grid = np.geomspace(x.min(), x.max(), n_grid)
    x_grid = np.asarray(x_grid, dtype=float)
    # With no identifiable Richards fit, the monotone curve is drawn instead: it is what
    # the crossing already comes from, so the panel stays honest rather than blank.
    y_fit = (predict(params, x_grid) if params is not None
             else monotone_curve(x, y, x_grid))

    fine = np.geomspace(x.min(), x.max(), 800)
    point = _first_crossing(fine, monotone_curve(x, y, fine), target)

    levels = np.unique(x)
    idx = [np.flatnonzero(x == lv) for lv in levels]
    rng = np.random.default_rng(seed)

    n_boot_band = min(30, n_boot) if n_boot_band is None else min(n_boot_band, n_boot)
    curves, crossings = [], []
    for b in range(n_boot):
        take = np.concatenate([rng.choice(i, size=i.size, replace=True) for i in idx])
        xb, yb = x[take], y[take]
        if b < n_boot_band:
            pb = fit_richards(xb, yb) if params is not None else None
            if pb is not None:
                curves.append(predict(pb, x_grid))
            elif params is None:
                curves.append(monotone_curve(xb, yb, x_grid))
        crossings.append(_first_crossing(fine, monotone_curve(xb, yb, fine), target))

    if curves:
        arr = np.array(curves)
        y_lo, y_hi = np.percentile(arr, ci, axis=0)
    else:
        y_lo = y_hi = np.full(x_grid.shape, np.nan)

    cb = np.array(crossings, dtype=float)
    cb = cb[np.isfinite(cb)]
    c_lo, c_hi = (np.percentile(cb, ci) if cb.size >= 2 else (np.nan, np.nan))

    return dict(params=params, x_grid=x_grid, y_fit=y_fit, y_lo=y_lo, y_hi=y_hi,
                crossing=point, crossing_lo=float(c_lo), crossing_hi=float(c_hi),
                crossing_boot=cb,
                crossing_richards=_first_crossing(x_grid, y_fit, target),
                target=target, n_points=int(x.size), n_levels=int(levels.size))


# --- non-monotone curves ------------------------------------------------

def smooth_curve(x, y, x_grid, frac=0.30):
    """
    A local-linear smoother through a cloud that need not be monotone.

    Richards and PAVA both assume the response only ever increases. Where it does not --
    the recent-split spatial axis peaks in 1/M and then decays, because at very weak
    migration the demes drift into independent clusters and a single (1,0)/(0,1) pair
    stops describing the structure -- forcing a monotone fit flattens the hump into a
    plateau and silently misrepresents the data.

    Local linear rather than local mean: a running mean biases the fit down at a peak
    (it averages across the curvature), while a local line reproduces a turning point.
    Fitted in log x, since every sweep here is logarithmically spaced.

    A caveat worth knowing: where a level's cloud is genuinely bumpy the fit will follow
    the bumps, and peak_with_band then reports whichever local maximum is highest. The
    phi=0.7 curve of split_ss_vary_phi is the example -- its raw medians dip to 0.78 at
    1/M ~ 0.30 and recover to 0.93 at 0.63, so there really are two maxima and widening
    `frac` only flattens the peak rather than resolving which one is meant. Read peak_x
    with its CI, not as a single number, on any curve that looks multi-modal.

    Args:
        frac: fraction of the points in each local window (LOWESS's usual span).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = np.log(x[ok]), y[ok]
    grid = np.log(np.asarray(x_grid, dtype=float))

    order = np.argsort(x)
    x, y = x[order], y[order]
    span = max(int(np.ceil(frac * x.size)), 3)

    out = np.empty(grid.size)
    for i, g in enumerate(grid):
        d = np.abs(x - g)
        h = np.partition(d, span - 1)[span - 1] or d.max() or 1.0
        w = np.clip(1.0 - (d / h) ** 3, 0.0, None) ** 3      # tricube
        if w.sum() <= 0:
            out[i] = np.nan
            continue
        # Weighted local linear fit, falling back to the weighted mean if degenerate.
        sw = w.sum(); sx = (w * x).sum(); sxx = (w * x * x).sum()
        sy = (w * y).sum(); sxy = (w * x * y).sum()
        det = sw * sxx - sx * sx
        out[i] = ((sxx * sy - sx * sxy) + (sw * sxy - sx * sy) * g) / det \
            if abs(det) > 1e-12 else sy / sw
    return out


def peak_with_band(x, y, x_grid=None, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED,
                   n_grid=300, ci=(16, 84), frac=0.30, n_boot_band=None):
    """
    Fit a non-monotone cloud and report where and how high it peaks.

    The counterpart of fit_with_band for sweeps whose response is not monotone, where a
    crossing is not the right summary: the peak of the curve, and the swept value at which
    it occurs, are. Replicates are resampled within each swept value, exactly as
    fit_with_band does, so a level never loses its replicate count.

    Returns a dict with x_grid / y_fit / y_lo / y_hi for drawing, and peak_y / peak_x with
    their percentile CIs. Returns None when there is nothing to fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size == 0 or np.unique(x).size < _MIN_LEVELS:
        return None

    if x_grid is None:
        x_grid = np.geomspace(x.min(), x.max(), n_grid)
    x_grid = np.asarray(x_grid, dtype=float)

    y_fit = smooth_curve(x, y, x_grid, frac=frac)
    j = int(np.nanargmax(y_fit))

    levels = np.unique(x)
    idx = [np.flatnonzero(x == lv) for lv in levels]
    rng = np.random.default_rng(seed)
    n_boot_band = min(60, n_boot) if n_boot_band is None else min(n_boot_band, n_boot)

    curves, peak_x, peak_y = [], [], []
    for b in range(n_boot):
        take = np.concatenate([rng.choice(i, size=i.size, replace=True) for i in idx])
        yb = smooth_curve(x[take], y[take], x_grid, frac=frac)
        if not np.isfinite(yb).any():
            continue
        k = int(np.nanargmax(yb))
        peak_x.append(float(x_grid[k]))
        peak_y.append(float(yb[k]))
        if b < n_boot_band:
            curves.append(yb)

    if curves:
        y_lo, y_hi = np.nanpercentile(np.array(curves), ci, axis=0)
    else:
        y_lo = y_hi = np.full(x_grid.shape, np.nan)

    q = lambda a: (np.percentile(a, ci) if len(a) >= 2 else (np.nan, np.nan))
    px_lo, px_hi = q(np.array(peak_x))
    py_lo, py_hi = q(np.array(peak_y))

    return dict(x_grid=x_grid, y_fit=y_fit, y_lo=y_lo, y_hi=y_hi,
                peak_x=float(x_grid[j]), peak_y=float(y_fit[j]),
                peak_x_lo=float(px_lo), peak_x_hi=float(px_hi),
                peak_y_lo=float(py_lo), peak_y_hi=float(py_hi),
                n_points=int(x.size), n_levels=int(levels.size))


# --- the theoretical crossing --------------------------------------------

def theory_crossing(overlap_of_x, x_lo, x_hi, target=DEFAULT_TARGET, n_grid=400):
    """
    The swept value at which the *theoretical* squared overlap reaches `target`.

    overlap_of_x maps a swept value to a predicted cos^2 -- typically a closure calling
    exact_overlap from src/threshold_and_overlap with the nu(x) of that demography.

    The theory is monotone in the swept parameter over the range in play, but the exact
    route returns a flat 0 below threshold, so a plain bisection over the whole range
    would be misled by that dead region: this brackets on a coarse log grid first and
    bisects only inside the bracketing interval.

    Returns NaN when the target is not reached anywhere in [x_lo, x_hi].
    """
    grid = np.geomspace(x_lo, x_hi, n_grid)
    vals = np.array([_safe(overlap_of_x, xx) for xx in grid])
    above = np.flatnonzero(np.isfinite(vals) & (vals >= target))
    if above.size == 0:
        return np.nan
    j = above[0]
    if j == 0:
        return float(grid[0])
    f = lambda xx: _safe(overlap_of_x, xx) - target
    try:
        return float(brentq(f, grid[j - 1], grid[j], xtol=1e-14, rtol=1e-12))
    except (ValueError, RuntimeError):
        return float(grid[j])


def _safe(fn, x):
    """Evaluate a theory closure, mapping a numerical failure to NaN rather than raising."""
    try:
        v = float(fn(x))
    except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError):
        return np.nan
    return v if np.isfinite(v) else np.nan


# --- writing per-replicate rows ------------------------------------------

def replicate_rows(base, x_value, x_name, cos_per_replicate, seed=None,
                   cos_std_per_replicate=None, per_replicate_cols=None):
    """
    Expand one sweep cell into one tidy row per replicate.

    base holds the columns shared by every replicate of the cell (the curve parameter,
    N, L_eff, nu, analytical_cos, ...). Both cos and cos^2 are written: the fit works on
    cos^2, while the existing figures report |cos|.

    cos_std_per_replicate is the same measurement on the standardized GRM (every locus
    weighted equally rather than by p(1-p)), from the same subsample and against the same
    theoretical axis, written as cos_std/cos2_std. It gets its own fit downstream, so that
    the transition and the 0.9 crossing can be compared between the two constructions.

    The theoretical prediction rides along on every row. It is a property of the cell, not
    of the replicate, so it repeats identically down the block -- which is what makes the
    file self-contained: measured and predicted overlap sit side by side without a join
    back to the summary CSV. analytical_cos2 is added next to it so the comparison is on
    the same squared scale as the fit.

    per_replicate_cols is the exception to that: a {name: sequence} of quantities that
    genuinely vary between replicates of one cell rather than repeating. The random-deme
    sweeps need it, because there each replicate draws a different deme subset and so has
    its own nu.
    """
    rows = []
    ana = base.get("analytical_cos")
    cos_std = (np.atleast_1d(cos_std_per_replicate)
               if cos_std_per_replicate is not None else None)
    for rep, c in enumerate(np.atleast_1d(cos_per_replicate)):
        row = dict(base)
        row[x_name] = float(x_value)
        row["replicate"] = int(rep)
        row["cos"] = float(c)
        row["cos2"] = float(c) ** 2
        if cos_std is not None and rep < len(cos_std):
            row["cos_std"] = float(cos_std[rep])
            row["cos2_std"] = float(cos_std[rep]) ** 2
        if ana is not None and np.isfinite(ana):
            row["analytical_cos2"] = float(ana) ** 2
        if per_replicate_cols:
            for name, values in per_replicate_cols.items():
                values = np.atleast_1d(values)
                if rep < len(values):
                    row[name] = float(values[rep])
        if seed is not None:
            row["seed"] = int(seed)
        rows.append(row)
    return rows

# %%

# --- the crossings table -------------------------------------------------

def crossings_table(df, level_keys, x_key, target=DEFAULT_TARGET, value_key="cos2",
                    theory_overlap=None, theory_key="analytical_cos2",
                    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, extra_keys=()):
    """
    One row per curve: where the measurement crosses `target`, and where theory does.

    This is the comparison the figures make, computed once in the analysis layer so it
    exists as a number whether or not anyone drew the plot. `level_keys` names the columns
    that identify a curve (e.g. ["N"], or ["sweep", "phi"] when one file holds several
    sweeps); every distinct combination becomes a row.

    Two theoretical crossings are reported when the inputs allow, because they answer
    slightly different questions:

        theory_crossing_pred   inverts `theory_overlap`, a closure giving the predicted
                               cos^2 at any x. Continuous, so it lands between grid points.
        theory_crossing_grid   interpolates the per-cell predictions already stored in the
                               CSV (theory_key). Needs no closure and is always available,
                               but is only as fine as the simulated grid.

    Args:
        df: per-replicate rows.
        level_keys: columns identifying one curve.
        x_key: the swept parameter column.
        theory_overlap: dict-of-level-values -> (x -> predicted cos^2), or None.
        extra_keys: further columns to carry through (first value per curve), for context
            that is constant within a curve -- N, L_eff, gamma and so on.

    Returns a DataFrame with the level columns, the measured crossing and its CI, both
    theory crossings, and the ratio measured/theory.
    """
    import pandas as pd

    level_keys = list(np.atleast_1d(level_keys))
    rows = []
    for values, d in df.groupby(level_keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        x = d[x_key].to_numpy(dtype=float)
        y = d[value_key].to_numpy(dtype=float)

        fit = fit_with_band(x, y, target=target, n_boot=n_boot, seed=seed)
        row = dict(zip(level_keys, values))
        row["target"] = target
        row["n_replicates"] = int(d.groupby(x_key).size().median()) if len(d) else 0
        row["n_levels"] = int(np.unique(x).size)

        if fit is None:
            row.update(crossing=np.nan, crossing_lo=np.nan, crossing_hi=np.nan,
                       crossing_richards=np.nan)
        else:
            row.update(crossing=fit["crossing"], crossing_lo=fit["crossing_lo"],
                       crossing_hi=fit["crossing_hi"],
                       crossing_richards=fit["crossing_richards"])

        # Theory from the closure, when the caller supplied one.
        x_pred = np.nan
        if theory_overlap is not None:
            fn = theory_overlap(dict(zip(level_keys, values)))
            if fn is not None and x.size:
                x_pred = theory_crossing(fn, float(np.min(x)), float(np.max(x)),
                                         target=target)
        row["theory_crossing_pred"] = x_pred

        # Theory from the per-cell predictions already in the file.
        x_grid_theory = np.nan
        if theory_key in d.columns:
            g = d.groupby(x_key)[theory_key].median().sort_index()
            if g.size >= 2:
                x_grid_theory = _first_crossing(g.index.to_numpy(dtype=float),
                                                g.to_numpy(dtype=float), target)
        row["theory_crossing_grid"] = x_grid_theory

        ref = x_pred if np.isfinite(x_pred) else x_grid_theory
        row["ratio_measured_over_theory"] = (row["crossing"] / ref
                                             if np.isfinite(row["crossing"])
                                             and np.isfinite(ref) and ref else np.nan)
        for k in extra_keys:
            if k in d.columns:
                row[k] = d[k].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)
