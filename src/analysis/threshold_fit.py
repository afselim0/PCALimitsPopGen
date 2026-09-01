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

`lam` is not a tuned knob: tau moves by <6% over a 16x range of it (2 -> 32). Nothing here
uses the theoretical curve or the BBP closed form, so the fitted transition is an
independent measurement rather than a restatement of the theory it is checked against.

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


def replicate_weights(t, y, n_bin=25, floor_frac=0.05):
    """Inverse-variance weights from the local spread of replicates in log x."""
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


def _profile(t, y, w, grid, n_basis, lam, min_left, min_right):
    """
    Profile the weighted GCV over the break. Returns (gcv, rss, tau, c, coef, knots).

    The candidate set includes tau below every observation: a sweep that begins above its
    transition has NO plateau, and forcing one makes the model flatten a genuinely rising
    left segment and report a break that is an artifact of `min_left`. That no-plateau
    candidate has one parameter fewer (no plateau level to estimate), so GCV chooses it on
    equal terms; its tau is a bound (the transition is below the swept range), which
    `tau_below_range` records.
    """
    n = len(y)
    best = None

    # Candidate 0: NO PLATEAU -- the curve rises across the whole swept range because its
    # transition lies below it. Without this candidate the model must invent a break, and
    # with the arm too stiff to follow a full-range rise it "fits" one by flattening the
    # part the arm cannot reach: the 1D stepping-stone sweeps came out with a plateau at
    # cos^2 ~ 0.7 and a break that was an artifact of min_left.
    #
    # The arm gets more basis functions HERE ONLY, because it has to span the entire range
    # rather than just the part above a break. That extra flexibility must not be offered
    # to the interior-break candidates: there it lets the arm absorb the elbow itself, and
    # the break drifts low and noisy (tau/onset 1.00 -> 0.72 on the split and OOA3 sweeps).
    # Stiff arm for locating a break; flexible arm only when there is no break to locate.
    tau0 = t.min() - 1e-9
    c0 = float(np.average(y[:min_left], weights=w[:min_left]))
    for nb in (n_basis, 16, 24):
        coef, knots, pred, k = _fit_arm(t - tau0, y - c0, w, nb, lam)
        # Only admit it if the arm is genuinely climbing where the data begins. A flexible
        # arm can also imitate a flat start, and then it would win on GCV over a real
        # plateau purely by having one parameter fewer -- which is what happened to
        # african_split N=200 (a true plateau at cos^2 = 0.01 read as no plateau).
        lead = pred[t <= np.quantile(t, 0.1)]
        if lead.size < 2 or (lead[-1] - lead[0]) < 0.02:
            continue
        rss = (w * (y - c0 - pred) ** 2).sum()
        gcv = n * rss / max((n - (1 + k)) ** 2, 1)
        if best is None or gcv < best[0]:
            best = (gcv, rss, tau0, c0, coef, knots)

    for tau in grid:
        left = t <= tau
        right = ~left
        if right.sum() < min_right or left.sum() < min_left:
            continue
        c = float(np.average(y[left], weights=w[left]))
        coef, knots, pred, k = _fit_arm(t[right] - tau, y[right] - c, w[right],
                                        n_basis, lam)
        rss = (w[left] * (y[left] - c) ** 2).sum() + \
              (w[right] * (y[right] - c - pred) ** 2).sum()
        # GCV rather than raw RSS: a shorter right segment buys freedom that RSS alone
        # would reward, which biases the break.
        gcv = n * rss / max((n - (2 + k)) ** 2, 1)
        if best is None or gcv < best[0]:
            best = (gcv, rss, tau, c, coef, knots)
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

    grid = np.linspace(np.quantile(t, lo_q), np.quantile(t, hi_q), n_grid)
    best = _profile(t, y, w, grid, n_basis, lam, min_left, min_right)
    if best is None:
        return None
    _, rss, tau, c, coef, knots = best

    if x_grid is None:
        x_grid = np.geomspace(x.min(), x.max(), n_grid)
    x_grid = np.asarray(x_grid, dtype=float)
    y_fit = _curve(x_grid, tau, c, coef, knots)

    y_lo, y_hi, tau_lo, tau_hi = _bootstrap(
        t, y, w, grid, x_grid, n_basis, lam, min_left, min_right, n_boot, level, seed,
        n_jobs)

    # Is the segment below the break actually flat? The model always returns some break,
    # so a sweep that begins above its transition gets one partway up the rise; the flag
    # separates that from a real plateau. See transition_with_ci.
    # The no-plateau candidate won: the curve rises across the whole swept range, so its
    # transition lies below the data and tau is a bound rather than a measurement.
    below_range = bool(tau <= t.min())
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
    """One bootstrap resample: refit the whole model and return its curve and break."""
    (t, y, w, grid, x_grid, n_basis, lam, min_left, min_right, s) = args
    rng = np.random.default_rng(s)
    n = len(y)
    idx = np.sort(rng.integers(0, n, n))
    b = _profile(t[idx], y[idx], w[idx], grid, n_basis, lam, min_left, min_right)
    if b is None:
        return None
    _, _, tb, cb, coefb, knotsb = b
    return _curve(x_grid, tb, cb, coefb, knotsb), float(np.exp(tb))


def _bootstrap(t, y, w, grid, x_grid, n_basis, lam, min_left, min_right,
               n_boot, level, seed, n_jobs=None):
    """Resample replicates and refit the whole model, tau included.

    Parallel over resamples: one profile is ~0.6 s, so a serial bootstrap would dominate.
    """
    if not n_boot:
        return None, None, np.nan, np.nan
    n_boot = int(n_boot)
    jobs = [(t, y, w, grid, x_grid, n_basis, lam, min_left, min_right, seed + 1 + i)
            for i in range(n_boot)]
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
