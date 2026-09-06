#%%
"""
Shape-constrained smooths for the overlap sweeps, via R's `scam` package through rpy2.

Replaces the parametric Richards sigmoid used previously. A sigmoid imposes a specific
functional form the data does not actually have -- its crossing came out ~12% biased
against known closed-form answers, which is why the reported crossing had to be taken from
a separate monotone interpolation instead. A shape-constrained spline imposes only the
qualitative shape we can actually defend, and `scam` supplies a proper Bayesian confidence
band for it, so one fit serves both the drawn curve and the reported number.

Shape constraints used here (scam's `bs=` codes):

    "mpi"  monotone increasing -- every overlap-vs-Fst and overlap-vs-(1/M) sweep, where
           more structure (or weaker migration) can only help recovery.
    "tp"   UNCONSTRAINED       -- the recent-split spatial sweeps, where cos^2 rises to a
           peak near 1/M ~ 0.3 and then decays: at very weak migration the demes drift into
           independent clusters and a single (1,0)/(0,1) pair stops describing the
           structure. This is not a shape-constrained fit at all: it is an ordinary
           penalized thin-plate smooth through mgcv::gam, whose only restraint is the
           smoothness the GCV penalty selects, and the peak is read off wherever the smooth
           puts it. Concavity ("cv") was used here previously; it forces the whole curve to
           bend one way, which pulls the fitted maximum toward the middle of the swept
           range when the real rise and decay are asymmetric.

Everything is fitted against log(x), matching the log x axis of the figures, so "monotone
in x" and "monotone in log x" are the same constraint and the knots are evenly spread
across the plotted range.

The band is scam's own `se.fit` from predict(), i.e. the Bayesian credible interval for the
smooth, scaled to 99% here -- see DEFAULT_LEVEL for why not 95%. That is what makes this route worth the R dependency: no
bootstrap is needed for the band, so a curve costs one fit rather than n_boot of them.

The crossing (where the fitted curve first reaches the target) is read off the *same*
fitted curve the figure draws, so the vertical marker always sits exactly on the line. Its
uncertainty comes from the band: the interval spanned by where the upper and lower band
edges cross the target. That is the honest reading of "how well is this crossing resolved"
and it costs nothing extra.

Requires: R with the `scam` package, and rpy2. `available()` reports whether the stack is
importable so callers can fail with a clear message rather than an rpy2 traceback.
"""
import numpy as np

# 99%, not the conventional 95%. These curves are fitted through ~800 replicate points, so
# the posterior SD of the smooth is small and a 95% band comes out around a line-width
# wide -- present, but not legible as a band. 99% is ~1.31x wider (2.576/1.96) and is still
# a standard interval rather than an arbitrary inflation.
DEFAULT_LEVEL = 0.99

# The bases scam::scam owns. Anything else is fitted with mgcv::gam instead.
_SCAM_SHAPES = ("mpi", "mpd", "cv", "cx", "micv", "micx", "mdcv", "mdcx")

_R_CACHE = {}


def available():
    """Whether rpy2 and R's scam package can both be loaded."""
    try:
        _r()
        return True
    except Exception:
        return False


def _r():
    """Import rpy2 and load scam once, caching the handles."""
    if "scam" in _R_CACHE:
        return _R_CACHE

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    # No numpy2ri converter: every array crosses the boundary as an explicit FloatVector
    # and comes back through np.asarray, so the global conversion context is not needed
    # (and activate() is removed in rpy2 3.6+).
    _R_CACHE["ro"] = ro
    _R_CACHE["scam"] = importr("scam")
    _R_CACHE["mgcv"] = importr("mgcv")     # unconstrained smooths (bs="tp")
    _R_CACHE["stats"] = importr("stats")
    return _R_CACHE


def scam_fit(x, y, shape="mpi", k=10, x_grid=None, n_grid=300, level=DEFAULT_LEVEL,
             logit=False):
    """
    Fit a shape-constrained spline of y on log(x) and predict it with a confidence band.

    Args:
        x: swept parameter, one entry per replicate (must be positive).
        y: squared cosine similarity of that replicate.
        shape: basis code. "mpi" (monotone increasing, via scam) is the constrained case;
            "tp" is an unconstrained thin-plate smooth fitted with mgcv::gam instead, for
            the curves that turn over.
        k: spline basis dimension. 10 is scam's default and is ample for these sweeps,
            which have 40-100 distinct x values; the shape constraint, not k, is what
            controls the wiggliness.
        level: confidence level for the band. Defaults to 99% rather than 95%: with
            ~800 replicate points per curve the posterior SD of the smooth is small, and
            a 95% band is around a line-width wide -- too thin to read as a band at all.
            99% is ~1.3x wider and still a standard interval.
        logit: fit on the logit scale so the curve cannot leave (0, 1). Off by default --
            these responses sit well inside the interval and the identity scale keeps the
            band symmetric and the constraint literal.

    Returns a dict with x_grid, y_fit, y_lo, y_hi, plus `edf` and `shape`, or None if the
    fit fails or there is too little data to identify it.
    """
    r = _r()
    ro = r["ro"]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size < 10 or np.unique(x).size < 5:
        return None

    logx = np.log(x)
    if x_grid is None:
        x_grid = np.geomspace(x.min(), x.max(), n_grid)
    x_grid = np.asarray(x_grid, dtype=float)

    # k cannot exceed the number of distinct covariate values scam has to work with.
    k_eff = int(min(k, max(4, np.unique(logx).size - 1)))

    y_fit_scale = _to_logit(y) if logit else y

    ro.globalenv["xx"] = ro.FloatVector(logx)
    ro.globalenv["yy"] = ro.FloatVector(y_fit_scale)
    ro.globalenv["xg"] = ro.FloatVector(np.log(x_grid))
    # scam's shape-constrained bases live in scam::scam; an unconstrained smooth is an
    # ordinary mgcv::gam. Both expose the same predict(se.fit) interface and the same
    # $edf, so only the fitting call differs.
    fitter = "scam::scam" if shape in _SCAM_SHAPES else "mgcv::gam"
    try:
        ro.r(f'''
            df <- data.frame(xx = xx, yy = yy)
            fit <- {fitter}(yy ~ s(xx, k = {k_eff}, bs = "{shape}"), data = df)
            pr  <- predict(fit, newdata = data.frame(xx = xg), se.fit = TRUE)
            fitv <- as.numeric(pr$fit)
            sev  <- as.numeric(pr$se.fit)
            edfv <- sum(fit$edf)
        ''')
    except Exception as exc:                       # rpy2 raises RRuntimeError subclasses
        print(f"    scam fit failed ({type(exc).__name__}: {exc})")
        return None

    fitv = np.asarray(ro.globalenv["fitv"], dtype=float)
    sev = np.asarray(ro.globalenv["sev"], dtype=float)
    edf = float(np.asarray(ro.globalenv["edfv"], dtype=float)[0])

    # Normal quantile: scam's se.fit is the Bayesian posterior sd of the smooth.
    z = _z_for(level)
    lo, hi = fitv - z * sev, fitv + z * sev

    if logit:
        fitv, lo, hi = _from_logit(fitv), _from_logit(lo), _from_logit(hi)

    # cos^2 is a squared cosine: clipping keeps the band inside the interval it must lie in
    # without distorting the fit itself.
    return dict(x_grid=x_grid,
                y_fit=np.clip(fitv, 0.0, 1.0),
                y_lo=np.clip(lo, 0.0, 1.0),
                y_hi=np.clip(hi, 0.0, 1.0),
                edf=edf, shape=shape, k=k_eff, level=level,
                n_points=int(x.size), n_levels=int(np.unique(x).size))


def _z_for(level):
    """Two-sided normal quantile for a confidence level."""
    from scipy.stats import norm
    return float(norm.ppf(0.5 + level / 2.0))


def _to_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _from_logit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def first_crossing(x_grid, y_grid, target):
    """
    First x where a sampled curve reaches `target`, interpolating in log x.

    Returns NaN when the curve never reaches it, and the left edge when it starts above.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    ok = np.isfinite(y_grid)
    x_grid, y_grid = x_grid[ok], y_grid[ok]
    above = np.flatnonzero(y_grid >= target)
    if above.size == 0:
        return np.nan
    j = above[0]
    if j == 0:
        return float(x_grid[0])
    lx0, lx1 = np.log(x_grid[j - 1]), np.log(x_grid[j])
    y0, y1 = y_grid[j - 1], y_grid[j]
    if y1 == y0:
        return float(x_grid[j])
    return float(np.exp(lx0 + (target - y0) * (lx1 - lx0) / (y1 - y0)))


def crossing_with_ci(fit, target, edge_frac=0.02):
    """
    Where the fitted curve reaches `target`, with a CI read off the confidence band.

    The point estimate comes from the drawn curve, so the marker sits on the line by
    construction. The interval is where the band edges reach the same level: the upper edge
    reaches it first (giving the lower/earlier bound) and the lower edge last. A bound is
    NaN when that edge never reaches the target within the swept range.

    Returns (crossing, lo, hi, at_edge). `at_edge` is True when the curve only reaches the
    target within the last `edge_frac` of the grid, i.e. it grazes it right as the sweep
    ends. Such a crossing is a property of where the sweep stopped, not of the demography:
    the true crossing is somewhere beyond the simulated range and the reported value is a
    lower bound on it. Callers should mark these rather than compare them against theory.
    """
    if fit is None:
        return np.nan, np.nan, np.nan, False
    xg = fit["x_grid"]
    point = first_crossing(xg, fit["y_fit"], target)
    lo = first_crossing(xg, fit["y_hi"], target)   # upper band -> earliest crossing
    hi = first_crossing(xg, fit["y_lo"], target)   # lower band -> latest crossing

    reached = np.flatnonzero(np.asarray(fit["y_fit"]) >= target)
    at_edge = bool(reached.size and reached[0] >= len(xg) * (1.0 - edge_frac))
    return point, lo, hi, at_edge


def peak_with_ci(fit):
    """
    The maximum of the fitted curve, with a CI on its height from the band.

    Used for the concave (recent-split) sweeps, where the summary of interest is the peak
    rather than a crossing. The peak's x-location CI is the range of x whose *upper* band
    still reaches the fitted peak height -- i.e. the set of locations the data cannot rule
    out as the true argmax.
    """
    if fit is None:
        return dict(peak_x=np.nan, peak_y=np.nan, peak_x_lo=np.nan, peak_x_hi=np.nan,
                    peak_y_lo=np.nan, peak_y_hi=np.nan)
    xg, yf = fit["x_grid"], fit["y_fit"]
    j = int(np.nanargmax(yf))
    peak_x, peak_y = float(xg[j]), float(yf[j])

    plausible = np.flatnonzero(fit["y_hi"] >= peak_y)
    if plausible.size:
        peak_x_lo, peak_x_hi = float(xg[plausible[0]]), float(xg[plausible[-1]])
    else:
        peak_x_lo = peak_x_hi = np.nan

    return dict(peak_x=peak_x, peak_y=peak_y,
                peak_x_lo=peak_x_lo, peak_x_hi=peak_x_hi,
                peak_y_lo=float(fit["y_lo"][j]), peak_y_hi=float(fit["y_hi"][j]))


def pickup_with_ci(fit, frac=0.10, band_offset=0.02):
    """
    Where the fitted curve starts to rise -- the empirical phase transition.

    Defined as the first x whose slope d(cos^2)/d(log x) reaches `frac` of the curve's
    maximum slope over the swept range. The fit is already against log x, so the gradient
    is taken on that native scale and the answer is invariant to the units of x.

    NaN when that point is the first grid node: the curve was already rising when the
    sweep began, so its transition lies below the swept range and cannot be located from
    this data. That is the requested behaviour, and it is what separates a real onset from
    a sweep that started too late -- every ss1d_vary_n and split_ss_* curve returns NaN
    here, while every split_vary_N curve returns an interior value that decreases with N.

    Why not the finite-N noise floor. The obvious definition -- where cos^2 first exceeds
    the random-direction floor 2/(pi N) -- does not work. The measured sub-threshold cos^2
    is flat at ~0.005 regardless of N, while 2/(pi N) falls as 1/N (off by ~8x at N=1000):
    the sample PC below threshold is not a uniformly random direction, it retains residual
    alignment with the structure axis. The fitted curve therefore starts above that assumed
    floor and the crossing collapses to the left edge for nearly every curve.

    A relative-slope rule needs a constant, but the answer barely depends on it: the 5%,
    10% and 20% variants land within ~20% of each other across all four sweep groups, so
    what is being measured is a property of the curve rather than of `frac`.

    pickup_x_band is a secondary, band-aware cross-check: the first x at which the lower
    band edge clears the curve's starting height by `band_offset`. It tracks the primary
    closely but rests on two arbitrary choices (the offset, and y_fit[0] as a baseline --
    which several fits clip to exactly 0), so it is recorded as a diagnostic and not
    promoted.

    Returns pickup_x, pickup_y, pickup_frac, pickup_slope_max, pickup_at_edge, pickup_x_band.
    """
    nan_row = dict(pickup_x=np.nan, pickup_y=np.nan, pickup_frac=frac,
                   pickup_slope_max=np.nan, pickup_at_edge=False, pickup_x_band=np.nan)
    if fit is None:
        return nan_row

    xg = np.asarray(fit["x_grid"], dtype=float)
    yf = np.asarray(fit["y_fit"], dtype=float)
    if xg.size < 3 or not np.isfinite(yf).any():
        return nan_row

    slope = np.gradient(yf, np.log(xg))
    slope_max = float(np.nanmax(slope))
    if not np.isfinite(slope_max) or slope_max <= 0:
        return nan_row  # flat or falling throughout: no onset to find

    idx = np.flatnonzero(slope >= frac * slope_max)
    if idx.size == 0:
        return nan_row
    j = int(idx[0])

    row = dict(nan_row, pickup_slope_max=slope_max)
    if j > 0:
        row.update(pickup_x=float(xg[j]), pickup_y=float(yf[j]),
                   pickup_at_edge=bool(j >= len(xg) - 1))

    # Diagnostic: first x whose lower band clears where the curve started.
    y_lo = np.asarray(fit["y_lo"], dtype=float)
    above = np.flatnonzero(y_lo >= yf[0] + band_offset)
    if above.size and above[0] > 0:
        row["pickup_x_band"] = float(xg[int(above[0])])
    return row

# %%
