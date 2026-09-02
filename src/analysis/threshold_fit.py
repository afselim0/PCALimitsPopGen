#%%
"""
Threshold fit for the detection sweeps: flat below a break, monotone increasing above.

Replaces the shape-constrained scam spline. A penalized spline cannot turn a corner: it
minimises roughness, so it trades a small bias over a wide flat region against a smaller
second derivative at the elbow and starts rising well before detection actually begins.
That is a property of smoothing, not of the data, and it made the fitted curve disagree
with the phase transition measured from the eigenvalues (scam's slope-based pickup landed
at ~0.6x the eigenvalue transition, and pinned to the sweep's left edge at large N).

The model imposes the shape the RMT phase transition actually has:

    y(t) = c                                t <= tau     (slope exactly zero)
    y(t) = c + sum_j beta_j I_j(t - tau)    t >  tau     (monotone increasing)
    beta_j >= 0,   t = log x

`I_j` are I-spline basis functions -- each is increasing and starts at 0 -- so nonnegative
coefficients make the arm monotone increasing AND continuous at tau by construction, rather
than by a constraint applied afterwards. Below tau the curve is exactly constant, so the
figure stops implying detection where there is none.

This is one fit, not two: `c` and `beta` are solved in closed form at each candidate `tau`,
leaving a one-dimensional profile search over `tau`. The search is a grid, not an
optimizer, because RSS is non-smooth in `tau` -- points switch segments discretely
(Hinkley 1969; Chiu, Lockhart & Routledge 2006 for the flat-then-rising "bent cable").
Every candidate uses the same basis and the same criterion (weighted RSS), so nothing in
the comparison depends on which hypothesis is being tested. Whether a curve has a plateau
at all is decided separately and directly from the data -- see `below_noise_floor`.

Two estimands come off the SAME fitted curve, so the drawn line and the reported numbers
can never disagree:
    tau       the phase transition -- where the curve leaves the plateau
    crossing  where that curve reaches the target, read off by interpolation

**Inverse-variance weighting is what makes tau work**, more than the spline or the penalty.
The replicate noise sd varies by more than two orders of magnitude across a sweep (0.097
near the elbow against 0.00063 at the top, a 150x range), so unweighted least squares lets
the tight high-x points dominate and drags the break low: pooled tau/onset is 0.879 +- 0.107 unweighted against 1.001 +- 0.093 weighted, over
the K=2 and OOA3 sweeps. Weights come from the replicate spread at each x level, which the
sweeps already provide.

Accuracy, against where the theoretical cos^2 curve leaves zero (`lam=8`):

    K=2  vary N      tau/onset 1.024 +- 0.033     crossing/theory 1.066 +- 0.007
    K=2  vary phi    tau/onset 0.982 +- 0.160     crossing/theory 1.062 +- 0.007
    OOA3 vary N      tau/onset 0.984 +- 0.072     crossing/theory 0.927 +- 0.009
    OOA3 vary phi    tau/onset 1.014 +- 0.062     crossing/theory 0.917 +- 0.022
    pooled           tau/onset 1.001 +- 0.093     crossing/theory 1.005 +- 0.071

The crossing matches scam's accuracy (scam pooled: 1.010 +- 0.077), so nothing regresses.

**Compare tau against the theoretical cos^2 ONSET, not against the BBP threshold x_c.**
For the OOA3 sweeps (eps != 0) the theory's own cos^2 is identically zero at x_c and stays
zero to ~1.3-1.7x beyond it: the eigenvalue separates before the eigenvector aligns
measurably. Judged against x_c, tau looks 1.3-2.0x "wrong"; against the onset it is
0.93-1.03. For K=2 the two nearly coincide, which is why this only shows up in the sister
case.

`lam` has a floor, not a sweet spot. It must be big enough that the arm cannot bend at its
own origin and absorb the elbow; past that tau is insensitive to it. Pooled tau/onset over
the split and OOA3 sweeps: 0.35 (lam=0), 0.80 (0.25), 0.92 (1), 0.95 (2), 0.97 (4),
1.00 (8), 1.01 (16), 1.02 (32). Below ~4 the break slides left into the plateau; from ~8 up
it is stable. The crossing does not care at all (0.98-1.02 across that whole range), and
neither does the no-plateau verdict (22/22 everywhere) -- the penalty buys the BREAK, not
smoothness of the drawn line.

That stiffness requirement is what rules out a flexible arm generally, not just a small
`lam`. A plain isotonic (PAVA) right arm removes lam, the knots and the basis in one step,
but it interpolates: with n distinct u it has up to n free levels, so sliding tau left
costs nothing and the break is unidentified. It recovers tau/onset 0.23 on the real sweeps
and 0.56 on SYNTHETIC data with a known sharp break, against 1.04 for this arm. The RSS
profile shows why: isotonic's minimum sits at 5.9e-4 on a curve whose onset is 1.03e-3.

Nothing here uses the theoretical curve or the BBP closed form, so the fitted transition is
an independent measurement rather than a restatement of the theory it is checked against.

CAVEAT, worth carrying into any writeup: against synthetic curves with a KNOWN break this
estimator reads ~3% high with a sharp corner and 12-45% high once the corner is soft and
the sub-threshold segment drifts -- the regime the real sweeps sit in. The close agreement
with theory on real data (0.996) therefore involves some cancellation between that upward
bias and where the theoretical onset is defined. tau is a location estimate with a few
percent of systematic uncertainty, not a sharply identified threshold; the softness is
physical (the theory's own cos^2 leaves zero smoothly, and finite N smears it further),
not an artifact of the fit. Holding lam fixed and varying the true break gives a slope of
0.88 in log-log with correlation 0.998, so tau does track the data rather than the tuning.

Depends only on numpy/scipy -- no R, rpy2 or scam.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import nnls
from scipy.stats import spearmanr

DEFAULT_LAM = 8.0
DEFAULT_N_BASIS = 10
DEFAULT_LEVEL = 0.95
DEFAULT_N_BOOT = 100  # CIs agree to the 4th digit with 300; see the module docstring


def available():
    """Kept for symmetry with the scam module the pipeline used to import."""
    return True


def _ispline_basis(u, knots, degree=3):
    """I-spline basis on u >= 0: each column increases from 0, so nonneg coefs give a
    monotone increasing curve anchored at the break."""
    t = np.concatenate([np.repeat(knots[0], degree + 1), knots[1:-1],
                        np.repeat(knots[-1], degree + 1)])
    n = len(t) - degree - 1
    B = np.zeros((len(u), n))
    uc = np.clip(u, knots[0], knots[-1])
    for j in range(n):
        cj = np.zeros(n)
        cj[j] = 1.0
        ap = BSpline(t, cj, degree, extrapolate=False).antiderivative()
        B[:, j] = np.nan_to_num(ap(uc)) - ap(knots[0])
    end = np.nan_to_num(B.max(axis=0))
    return B / np.where(np.abs(end) < 1e-12, 1.0, end)


def _knots(u, n_basis, degree=3):
    """Knots on the quantiles of the arm's own domain, so the arm keeps the same
    flexibility whatever tau is (a fixed domain makes RSS incomparable across tau)."""
    qs = np.linspace(0, 1, max(n_basis - degree + 1, 2))
    k = np.unique(np.quantile(u, qs))
    if len(k) < 2:
        k = np.array([0.0, max(float(u.max()), 1e-9)])
    k[0] = 0.0
    return k


def _fit_arm(u, y, w, n_basis=DEFAULT_N_BASIS, lam=DEFAULT_LAM, degree=3):
    """Weighted nonneg-coefficient I-spline fit with a second-difference penalty."""
    knots = _knots(u, n_basis, degree)
    B = _ispline_basis(u, knots, degree)
    sw = np.sqrt(w)
    A, rhs = B * sw[:, None], y * sw
    if lam > 0 and B.shape[1] >= 3:
        D = np.diff(np.eye(B.shape[1]), 2, axis=0)
        scale = np.sqrt((A ** 2).sum() / max(len(u), 1))
        A = np.vstack([A, np.sqrt(lam) * scale * D])
        rhs = np.concatenate([rhs, np.zeros(D.shape[0])])
    coef, _ = nnls(A, rhs)
    return coef, knots, B @ coef, int((coef > 1e-10).sum())


def replicate_weights(t, y, n_bin=20, floor_frac=0.05):
    """
    Inverse-variance weights from the local spread of replicates in log x.

    `n_bin` sits on the plateau of its own sensitivity curve. Accuracy improves
    monotonically with resolution and then stops: pooled tau/onset over the 39 split and
    OOA3 curves runs 0.919 (5 bins), 0.957 (10), 0.980 (15), 0.996 (20), 1.006 (25),
    1.008 (40). Too few bins average the variance over a wide stretch of x, smearing the
    150x range into a coarse step function and blunting the weight contrast -- which
    reverts partway towards unweighted least squares and biases tau low, the same failure
    weighting exists to prevent. Past ~20 there is nothing left to gain (20 -> 40 moves the
    mean by 0.012 and the sd by 0.000). At ~800 replicates per curve, 20 bins leaves ~40
    points each, well above what a variance estimate needs.
    """
    edges = np.unique(np.quantile(t, np.linspace(0, 1, n_bin + 1)))
    if len(edges) < 3:
        return np.ones_like(y)
    b = np.clip(np.digitize(t, edges[1:-1]), 0, len(edges) - 2)
    seen = [y[b == j].var(ddof=1) for j in range(len(edges) - 1) if (b == j).sum() >= 3]
    floor = max(np.median(seen) * floor_frac, 1e-8) if seen else 1e-8
    var = np.full_like(y, floor)
    for j in range(len(edges) - 1):
        m = b == j
        if m.sum() >= 3:
            var[m] = max(y[m].var(ddof=1), floor)
    return 1.0 / var


NOISE_FLOOR = 0.15


def below_noise_floor(x, y, n_levels=3, floor=NOISE_FLOOR):
    """
    Does this curve start at the finite-N noise floor, i.e. is its transition inside the
    swept range?

    Returns True when the curve is ALREADY detecting at the smallest swept x -- the median
    over its lowest `n_levels` x values sits above `floor` -- so the transition lies below
    the data and tau is a bound rather than a measurement.

    This is a property of the data, deliberately not of the fit. The alternative, letting
    the no-plateau candidate win or lose on fit, does not work once every candidate shares
    a basis: over the 22 stepping-stone curves the two hypotheses are separated by RSS on
    only ~15, and raising the basis count to help buys nothing (the errors are all "claims
    a plateau", never the reverse) while degrading tau on the split sweeps from 1.01 to
    0.72. Making the call explicitly on the data keeps the estimator symmetric.

    `floor` is a convention, not a derived constant, and is stated as one. The physically
    motivated alternative -- the chance-alignment level k/N for a random unit vector -- is
    the right order for the split and OOA3 sweeps (measured plateau is 0.3-5.4x k/N) but
    does not separate the stepping-stone curves (0.4x to 1170x, overlapping), and the
    measured sub-threshold floor is known not to follow the 1/N scaling anyway: it sits
    flat at ~0.005 regardless of N. 0.15 sits in a wide gap for these sweeps -- curves with
    a real plateau start at 0.01-0.12, curves past their transition at 0.17-0.73 -- but it
    is a reporting choice about when tau is worth quoting, not a detected physical level.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lv = np.unique(x)[:max(int(n_levels), 1)]
    lead = y[np.isin(x, lv)]
    return bool(lead.size and float(np.median(lead)) > floor)


def _profile(t, y, w, grid, n_basis, lam, min_left, min_right, no_plateau=False):
    """
    Profile the weighted RSS over the break. Returns (rss, rss, tau, c, coef, knots).

    Every candidate -- including tau below all the data -- is fitted with the SAME basis
    and scored by the same criterion, so nothing about the comparison depends on which
    hypothesis is being tested.

    The no-plateau candidate (tau below every observation) matters because a sweep that
    begins above its transition has no plateau at all, and forcing one makes the model
    flatten a genuinely rising left segment and report a break that is an artifact of
    `min_left`. Whether that candidate wins is NOT how the pipeline decides a curve has no
    plateau, though: that call is made directly from the data (see `below_noise_floor`),
    because with a common basis the two hypotheses are not reliably separated by fit alone.

    Raw RSS rather than GCV. GCV was here to charge the no-plateau candidate for having one
    parameter fewer, but with a common basis the two criteria pick the same tau on 38 of 39
    split/OOA3 curves and 22 of 23 stepping-stone curves, so the correction earns nothing.
    """
    best = None

    tau0 = t.min() - 1e-9
    c0 = float(np.average(y[:min_left], weights=w[:min_left]))
    coef, knots, pred, _ = _fit_arm(t - tau0, y - c0, w, n_basis, lam)
    rss0 = (w * (y - c0 - pred) ** 2).sum()
    best = (rss0, rss0, tau0, c0, coef, knots)
    if no_plateau:
        # The data already say the curve starts above the noise floor, so a plateau is not
        # on the table: return the no-plateau fit rather than letting an interior break win
        # on a hair of RSS and draw a flat segment partway up the rise.
        return best

    for tau in grid:
        left = t <= tau
        right = ~left
        if right.sum() < min_right or left.sum() < min_left:
            continue
        c = float(np.average(y[left], weights=w[left]))
        coef, knots, pred, _ = _fit_arm(t[right] - tau, y[right] - c, w[right],
                                        n_basis, lam)
        rss = (w[left] * (y[left] - c) ** 2).sum() + \
              (w[right] * (y[right] - c - pred) ** 2).sum()
        if rss < best[0]:
            best = (rss, rss, tau, c, coef, knots)
    return best


def _curve(x_grid, tau, c, coef, knots):
    lg = np.log(x_grid)
    B = _ispline_basis(np.maximum(lg - tau, 0.0), knots)
    return np.clip(c + np.where(lg > tau, B @ coef, 0.0), 0.0, 1.0)


def threshold_fit(x, y, n_grid=300, lo_q=0.02, hi_q=0.90, n_basis=DEFAULT_N_BASIS,
                  lam=DEFAULT_LAM, min_left=15, min_right=30, x_grid=None,
                  n_boot=200, level=DEFAULT_LEVEL, seed=0, weighted=True,
                  n_jobs=None):
    """
    Fit the flat-then-increasing model and return the curve with a bootstrap band.

    Args:
        x: swept parameter, one entry per replicate (must be positive).
        y: squared cosine similarity of that replicate.
        lam: roughness penalty on the arm. tau is insensitive to it over a wide range.
        n_boot: replicate bootstrap resamples for the band and the tau CI. The whole
            model is refitted on each resample -- tau included -- so the interval covers
            the break's own uncertainty rather than treating it as known. 0 skips the band.
        n_jobs: processes for the bootstrap (default 1). Parallelising here barely pays
            -- the pipeline runs whole curves in parallel instead.

    Returns a dict with x_grid, y_fit, y_lo, y_hi, tau, tau_lo, tau_hi, plateau, edf,
    or None when there is too little data to identify the model.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size < 30 or np.unique(x).size < 8:
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]
    t = np.log(x)
    w = replicate_weights(t, y) if weighted else np.ones_like(y)

    # Decide from the DATA whether this curve has a plateau at all, then fit accordingly.
    # The verdict has to reach the fit, not just the reported flag: on ss1d n=4 the interior
    # break beat the no-plateau candidate by 1.02x in RSS and the drawn curve carried a flat
    # segment at cos^2 = 0.75, which is exactly the artifact the flag exists to rule out.
    below_range = below_noise_floor(x, y)
    grid = np.linspace(np.quantile(t, lo_q), np.quantile(t, hi_q), n_grid)
    best = _profile(t, y, w, grid, n_basis, lam, min_left, min_right,
                    no_plateau=below_range)
    if best is None:
        return None
    _, rss, tau, c, coef, knots = best

    if x_grid is None:
        x_grid = np.geomspace(x.min(), x.max(), n_grid)
    x_grid = np.asarray(x_grid, dtype=float)
    y_fit = _curve(x_grid, tau, c, coef, knots)

    y_lo, y_hi, tau_lo, tau_hi = _bootstrap(
        t, y, w, grid, x_grid, n_basis, lam, min_left, min_right, n_boot, level, seed,
        n_jobs, no_plateau=below_range)

    mask = t <= tau
    rho = np.nan
    if mask.sum() >= 8:
        rho = float(spearmanr(t[mask], y[mask]).statistic)


    return dict(x_grid=x_grid, y_fit=y_fit,
                y_lo=y_lo if y_lo is not None else y_fit,
                y_hi=y_hi if y_hi is not None else y_fit,
                tau=float(np.exp(tau)), tau_lo=tau_lo, tau_hi=tau_hi,
                plateau=float(c), edf=float(2 + int((coef > 1e-10).sum())),
                plateau_rho=rho, below_range=below_range,
                shape="flat_then_mono", lam=float(lam), level=level,
                n_points=int(x.size), n_levels=int(np.unique(x).size))


def _boot_one(args):
    """One bootstrap resample: refit the whole model and return its curve and break.

    `no_plateau` is carried over from the point estimate rather than re-decided per
    resample: the band has to describe the uncertainty of the model that was actually
    fitted. Letting resamples pick an interior break for a curve fitted without one drew a
    band around a different model, which put the fitted line outside its own band on 28% of
    the grid for ss1d n=64.
    """
    (t, y, w, grid, x_grid, n_basis, lam, min_left, min_right, s, no_plateau) = args
    rng = np.random.default_rng(s)
    n = len(y)
    idx = np.sort(rng.integers(0, n, n))
    b = _profile(t[idx], y[idx], w[idx], grid, n_basis, lam, min_left, min_right,
                 no_plateau=no_plateau)
    if b is None:
        return None
    _, _, tb, cb, coefb, knotsb = b
    return _curve(x_grid, tb, cb, coefb, knotsb), float(np.exp(tb))


def _bootstrap(t, y, w, grid, x_grid, n_basis, lam, min_left, min_right,
               n_boot, level, seed, n_jobs=None, no_plateau=False):
    """Resample replicates and refit the whole model, tau included.

    Parallel over resamples: one profile is ~0.6 s, so a serial bootstrap would dominate.
    """
    if not n_boot:
        return None, None, np.nan, np.nan
    n_boot = int(n_boot)
    jobs = [(t, y, w, grid, x_grid, n_basis, lam, min_left, min_right, seed + 1 + i,
             no_plateau) for i in range(n_boot)]
    # Serial by default: one resample is ~0.16 s, so process startup eats most of the
    # gain (48 s serial against 35 s on 16 workers, n_boot=100). The pipeline parallelises
    # across curves instead, which is the granularity that actually pays.
    if n_jobs is None:
        n_jobs = 1
    n_jobs = max(1, min(int(n_jobs), n_boot))

    if n_jobs == 1:
        out = [_boot_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            out = list(ex.map(_boot_one, jobs, chunksize=max(1, n_boot // (4 * n_jobs))))

    got = [o for o in out if o is not None]
    if len(got) < max(20, n_boot // 10):
        return None, None, np.nan, np.nan
    curves = np.vstack([g[0] for g in got])
    taus = np.array([g[1] for g in got])
    lo_q = 100 * (1 - level) / 2
    return (np.percentile(curves, lo_q, axis=0),
            np.percentile(curves, 100 - lo_q, axis=0),
            float(np.percentile(taus, lo_q)),
            float(np.percentile(taus, 100 - lo_q)))


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
    Where the fitted curve reaches `target`, with a CI from the bootstrap band.

    The point estimate comes from the drawn curve, so the marker sits on the line by
    construction. `at_edge` is True when the curve only reaches the target within the last
    `edge_frac` of the grid: such a crossing says where the sweep stopped rather than where
    detection happens, and callers should mark it rather than compare it against theory.
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


def transition_with_ci(fit, edge_frac=0.02):
    """
    The fitted phase transition: where the curve leaves the plateau.

    This is a parameter of the model rather than a level someone chose, which is what
    separates it from scam's slope-based pickup.

    Two flags say when the number is not a measurement of a transition:

    `tau_at_edge`  the break sits in the first or last sliver of the swept range, so the
        transition is outside the data and the value is a bound.

    `tau_below_range`  the fit chose the no-plateau candidate, i.e. the curve is already
        rising at the smallest swept x, so the transition is BELOW the data. `tau` is then
        the left edge and is a bound, not a measurement -- report only the crossing. Every
        1D stepping-stone curve is in this regime, and so are the larger-n 2D ones.

    `plateau_rho` is a diagnostic: the rank correlation of y against x below the break. It
    is not used to decide either flag, but a high value alongside a high plateau is the
    signature of a break that the model was forced to invent.
    """
    if fit is None or not np.isfinite(fit.get("tau", np.nan)):
        return dict(tau=np.nan, tau_lo=np.nan, tau_hi=np.nan, plateau=np.nan,
                    tau_at_edge=False, tau_below_range=False, plateau_rho=np.nan)
    xg = fit["x_grid"]
    lo_edge, hi_edge = xg[int(len(xg) * edge_frac)], xg[int(len(xg) * (1 - edge_frac)) - 1]
    return dict(tau=fit["tau"], tau_lo=fit["tau_lo"], tau_hi=fit["tau_hi"],
                plateau=fit["plateau"],
                tau_at_edge=bool(fit["tau"] <= lo_edge or fit["tau"] >= hi_edge),
                tau_below_range=bool(fit.get("below_range", False)),
                plateau_rho=fit.get("plateau_rho", np.nan))


def peak_with_ci(fit):
    """
    The maximum of the fitted curve, with a CI on its height from the band.

    For the concave (recent-split) sweeps, whose summary of interest is the peak rather
    than a crossing. The peak's x-location CI is the range of x whose upper band still
    reaches the fitted peak height -- the locations the data cannot rule out as the argmax.
    """
    if fit is None:
        return dict(peak_x=np.nan, peak_y=np.nan, peak_x_lo=np.nan, peak_x_hi=np.nan,
                    peak_y_lo=np.nan, peak_y_hi=np.nan)
    xg, yf = fit["x_grid"], fit["y_fit"]
    j = int(np.nanargmax(yf))
    peak_x, peak_y = float(xg[j]), float(yf[j])

    plausible = np.flatnonzero(np.asarray(fit["y_hi"]) >= peak_y)
    if plausible.size:
        peak_x_lo, peak_x_hi = float(xg[plausible[0]]), float(xg[plausible[-1]])
    else:
        peak_x_lo = peak_x_hi = np.nan

    return dict(peak_x=peak_x, peak_y=peak_y,
                peak_x_lo=peak_x_lo, peak_x_hi=peak_x_hi,
                peak_y_lo=float(fit["y_lo"][j]), peak_y_hi=float(fit["y_hi"][j]))

# %%
