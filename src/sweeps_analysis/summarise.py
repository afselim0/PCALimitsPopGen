#%%
"""
Stage 3: sweep directories -> the summary files the plotting layer reads.

    python summarise.py <sweep_dir> [<sweep_dir> ...] --out results/sweep_summaries/<name>

Reads each sweep's replicates.csv and writes three files, so that the plotting layer does
no fitting and holds no theory:

    crossings.csv  one row per curve: measured crossing + bootstrap CI, the theoretical
                   crossing (from inverting the model's closure, and from interpolating
                   the stored per-cell predictions), and their ratio
    curves.csv     the drawn curve on a grid: x, y_fit, y_lo, y_hi, y_theory. A panel
                   plots these columns directly
    points.csv     the replicate scatter, so a panel keeps its cloud without loading the
                   full replicate file

The monotone detection sweeps are fitted with the flat-then-increasing threshold model in
src/analysis/threshold_fit.py: constant below a fitted break, monotone increasing above it.
A smoothing spline cannot turn a corner, so it starts rising well before detection begins
and its transition disagreed with the one measured from the eigenvalues; the threshold
model imposes the shape the phase transition actually has, and reports the break itself as
a parameter. Both the transition and the target crossing are read off the SAME fitted curve,
so the drawn line and the reported numbers cannot disagree.

The recent-split spatial sweeps (`non_monotone: true`) turn over rather than saturating, so
they are not flat-then-increasing and keep the concave scam smooth, summarised by their
peak. See src/analysis/scam_fit.py.

Bands and CIs are a replicate bootstrap that refits the whole model, the break included --
conditioning on a fitted break would treat it as known and understate every interval.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis"))  # src/analysis/
from crossing_fit import DEFAULT_TARGET, theory_crossing
from scam_fit import available as scam_available
from eigenvalue_threshold import threshold_from_eigenvalues
from scam_fit import peak_with_ci as scam_peak_with_ci
from scam_fit import scam_fit
from threshold_fit import crossing_with_ci, threshold_fit, transition_with_ci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/sweeps_analysis/
from cli_common import DEFAULT_N_BOOT, DEFAULT_N_GRID
from models import get_model
from theory_table import theory_rows

VALUE_KEY = "cos2"
STD_VALUE_KEY = "cos2_std"
THEORY_KEY = "analytical_cos2"


def _load_h(sweep_dir):
    """level -> per-locus weights, from the sidecar the runner's aggregate step writes.

    Absent for sweeps run before the sidecar existed; the models then fall back to their
    two-moment route, which is why this returns an empty dict rather than raising.
    """
    path = os.path.join(sweep_dir, "h.npz")
    if not os.path.exists(path):
        return {}
    with np.load(path) as z:
        return {k: z[k].astype(float) for k in z.files}


def _theory_closure(cfg, cell_row, h=None):
    """
    The model's x -> cos^2 for one curve, or None when it cannot be built.

    Every model's theory_overlap is written to rebuild itself from what the runner already
    recorded per cell (nu, gamma_raw, mu2, delta), so the summariser never re-runs
    prepare() -- which for african_split would be a 15-20 min msprime integration per
    curve. A model that genuinely cannot do that returns None and the theoretical crossing
    falls back to interpolating the stored analytical_cos2.
    """
    model = get_model(cfg["model"])
    if not hasattr(model, "theory_overlap"):
        return None      # theory-free model (split_stepping_stone); empirical output only
    try:
        ctx = model.prepare_cached(cfg, cell_row) if hasattr(model, "prepare_cached") \
            else model.prepare(cfg, cell_row)
        measured = dict(cell_row)
        if h is not None:
            measured["h"] = h
        return model.theory_overlap(ctx, dict(cell_row), cell_row, cfg, measured)
    except Exception as exc:
        print(f"    theory closure unavailable ({type(exc).__name__}: {exc})")
        return None


def _theory_rows_for(cfg, cells, level, level_param, x_key, sweep, h, x_values, target):
    """
    The theory.csv rows for one curve: nu at each swept value, plus the three critical
    values and the three crossings, which are constants of the curve.

    Rebuilt from the model's theory_context/nu_of_x on the curve's median cell, the same
    anchor _theory_closure uses -- the per-cell context (gamma_raw, mu2, delta, L_eff) is
    constant down a curve by construction.
    """
    model = get_model(cfg["model"])
    if not all(hasattr(model, a) for a in ("theory_context", "nu_of_x")):
        return []            # theory-free model
    cell = cells[cells["level"] == level]
    if not len(cell):
        return []
    cell_row = cell.iloc[len(cell) // 2].to_dict()
    try:
        ctx = model.prepare_cached(cfg, cell_row) if hasattr(model, "prepare_cached") \
            else model.prepare(cfg, cell_row)
        measured = dict(cell_row)
        if h is not None:
            measured["h"] = h
        tc = model.theory_context(ctx, dict(cell_row), cell_row, cfg, measured)
        nu_fn = model.nu_of_x(ctx, dict(cell_row), cell_row, cfg, measured)
    except Exception as exc:
        print(f"    theory table unavailable ({type(exc).__name__}: {exc})")
        return []
    if tc is None or nu_fn is None:
        return []

    return theory_rows(sweep, level, level_param, x_key, nu_fn, tc, x_values,
                       target=target, nu_form=getattr(model, "NU_FORM", ""),
                       extra={level_param: level, "sweep_param": level})


def _nu_of_x_for(cfg, cells, level):
    """The model's nu(x) for one curve, or None for a theory-free model.

    The eigenvalue threshold needs nu(x) to convert the fitted nu_c back into the swept
    parameter; it is the same closure _theory_rows_for builds, anchored on the same cell.
    """
    model = get_model(cfg["model"])
    if not hasattr(model, "nu_of_x"):
        return None
    cell = cells[cells["level"] == level]
    if not len(cell):
        return None
    cell_row = cell.iloc[len(cell) // 2].to_dict()
    try:
        ctx = model.prepare_cached(cfg, cell_row) if hasattr(model, "prepare_cached") \
            else model.prepare(cfg, cell_row)
        return model.nu_of_x(ctx, dict(cell_row), cell_row, cfg, dict(cell_row))
    except Exception:
        return None


def _joint_frame(empirical, cells, cfg, h_by_level):
    """
    Attach each replicate's own theory to its own measurement (the random-deme sweeps).

    These sweeps redraw the deme subset per replicate, so nu is a property of the replicate
    and the runner records it per row. The critical values depend only on the demography
    and the locus weights -- not on which demes were drawn -- so they are computed once per
    curve and repeat; only nu and therefore cos2_* vary between replicates.

    No crossing columns: a crossing is a property of a curve, and here each replicate is a
    different realisation of the demography.
    """
    if not len(empirical) or "nu" not in empirical.columns:
        return empirical
    from theory_table import ROUTES, nu_c_all, overlap_fn

    model = get_model(cfg["model"])
    out = empirical.copy()
    for c in ROUTES:
        out[f"nu_c_{c}"] = np.nan
        out[f"cos2_{c}"] = np.nan

    for level in out["level"].unique():
        cell = cells[cells["level"] == level]
        if not len(cell) or not hasattr(model, "theory_context"):
            continue
        cell_row = cell.iloc[len(cell) // 2].to_dict()
        try:
            ctx = model.prepare_cached(cfg, cell_row) if hasattr(model, "prepare_cached") \
                else model.prepare(cfg, cell_row)
            measured = dict(cell_row)
            h = h_by_level.get(f"{level}")
            if h is not None:
                measured["h"] = h
            tc = model.theory_context(ctx, dict(cell_row), cell_row, cfg, measured)
        except Exception as exc:
            print(f"    per-replicate theory unavailable ({type(exc).__name__}: {exc})")
            continue
        if tc is None:
            continue

        nu_cs = nu_c_all(tc)
        m = out["level"] == level
        for route in ROUTES:
            nu_c = nu_cs[f"nu_c_{route}"]
            out.loc[m, f"nu_c_{route}"] = nu_c
            fn = overlap_fn(tc, route, nu_c=nu_c)
            if fn is not None:
                out.loc[m, f"cos2_{route}"] = [
                    fn(v) if np.isfinite(v) else np.nan for v in out.loc[m, "nu"]]
    return out


def _fit_summary(fit, target, non_monotone, suffix=""):
    """
    The scalar summaries of one fitted curve: crossing (or peak) and the transition onset.

    suffix is appended to every key, so the standardized fit can be written alongside the
    unstandardized one in the same row without a name clash. The two are summarised by
    exactly this function, so a difference between them is a difference in the data rather
    than in how they were reduced.
    """
    out = {}
    if non_monotone:
        pk = scam_peak_with_ci(fit)
        out.update({f"{k}{suffix}": v for k, v in pk.items()})
        out[f"reaches_target{suffix}"] = bool(pk["peak_y"] >= target)
    else:
        cx, cx_lo, cx_hi, at_edge = crossing_with_ci(fit, target)
        out[f"crossing{suffix}"] = cx
        out[f"crossing_lo{suffix}"] = cx_lo
        out[f"crossing_hi{suffix}"] = cx_hi
        out[f"crossing_y{suffix}"] = target
        out[f"crossing_at_edge{suffix}"] = at_edge
        # The fitted break: a parameter of the model, not a level someone chose. NaN for
        # the concave sweeps, which have no plateau to leave. tau_below_range marks a curve
        # already rising at the smallest swept x -- there tau is a bound, not a measurement.
        out.update({f"{k}{suffix}": v for k, v in transition_with_ci(fit).items()})
    out[f"shape{suffix}"] = fit["shape"] if fit is not None else None
    out[f"edf{suffix}"] = fit["edf"] if fit is not None else np.nan
    return out


def summarise_sweep(sweep_dir, target=DEFAULT_TARGET, n_boot=DEFAULT_N_BOOT,
                    n_grid=DEFAULT_N_GRID, seed=0, verbose=True, n_jobs=None):
    """Fit every curve of one sweep. Returns (crossings, curves, points) DataFrames."""
    rep_path = os.path.join(sweep_dir, "replicates.csv")
    cell_path = os.path.join(sweep_dir, "cells.csv")
    cfg_path = os.path.join(sweep_dir, "config.yaml")
    for p in (rep_path, cell_path, cfg_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{sweep_dir}: missing {os.path.basename(p)}; run the "
                                    f"sweep and its --aggregate step first")

    df = pd.read_csv(rep_path)
    cells = pd.read_csv(cell_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("fixed", {})

    sweep = cfg["name"]
    level_param = cfg["sampling"]["vary"]["param"]
    x_key = get_model(cfg["model"]).X_KEY
    non_monotone = bool(cfg.get("non_monotone", False))

    per_replicate_theory = bool(cfg.get("per_replicate_theory", False))
    h_by_level = _load_h(sweep_dir)
    cross_rows, curve_rows, point_rows = [], [], []
    theory_out, empirical_out = [], []
    for level in np.sort(df["level"].unique()):
        d = df[df["level"] == level]
        x = d["x"].to_numpy(float)
        y = d[VALUE_KEY].to_numpy(float)

        # The shape constraint is the only thing imposed on the curve. A sweep whose
        # response turns over gets "cv" (concave) and is summarised by its peak; every
        # other sweep is monotone increasing in the swept parameter and gets "mpi" and a
        # crossing. The recent-split spatial axis is the turnover case: cos^2 rises to
        # ~0.93 near 1/M ~ 0.3 and then decays, because at very weak migration the demes
        # drift into independent clusters and a single (1,0)/(0,1) pair stops describing
        # the structure.
        # A sweep whose response turns over keeps the concave scam smooth and is
        # summarised by its peak: it has no plateau, so the threshold model does not
        # describe it. Every other sweep is flat-then-increasing in the swept parameter.
        shape = "cv" if non_monotone else "mpi"
        if non_monotone:
            fit = scam_fit(x, y, shape=shape, n_grid=n_grid)
        else:
            fit = threshold_fit(x, y, n_grid=n_grid, n_boot=n_boot, seed=seed,
                                n_jobs=n_jobs)
        if fit is None:
            if verbose:
                print(f"  {sweep} level={level}: no identifiable fit, skipped")
            continue

        # The same curve measured off the standardized GRM, fitted identically on the same
        # grid so the two are compared like for like. Absent for sweeps run before the
        # standardized measurement existed, in which case every *_std column stays NaN.
        fit_std = None
        if STD_VALUE_KEY in d.columns:
            y_std = d[STD_VALUE_KEY].to_numpy(float)
            if np.isfinite(y_std).any():
                fit_std = (scam_fit(x, y_std, shape=shape, x_grid=fit["x_grid"],
                                    n_grid=n_grid) if non_monotone
                           else threshold_fit(x, y_std, x_grid=fit["x_grid"],
                                              n_grid=n_grid, n_boot=n_boot, seed=seed,
                                              n_jobs=n_jobs))

        # The theory closure is anchored on one representative cell of this curve; the
        # per-cell context it needs (gamma_raw, mu2, delta, nu) is constant down the curve
        # by construction, so the median cell serves.
        cell = cells[cells["level"] == level]
        fn = (_theory_closure(cfg, cell.iloc[len(cell) // 2].to_dict(),
                              h=h_by_level.get(f"{level}"))
              if len(cell) else None)
        x_star = (theory_crossing(fn, float(x.min()), float(x.max()), target=target)
                  if fn is not None else np.nan)

        # A second theoretical crossing: interpolating the analytical_cos2 the analysis
        # already stored per cell, rather than re-inverting the closure. Two routes to the
        # same number, so a disagreement is visible instead of silent.
        x_grid_theory = np.nan
        if THEORY_KEY in d.columns:
            td = d.groupby("x")[THEORY_KEY].first().reset_index().sort_values("x")
            above = td[td[THEORY_KEY] >= target]
            if len(above):
                x_grid_theory = float(above["x"].iloc[0])

        y_theory = (np.array([fn(xx) for xx in fit["x_grid"]]) if fn is not None
                    else np.full(len(fit["x_grid"]), np.nan))

        # The empirical detection threshold, from the sample eigenvalues. This is the
        # measurement the cosine cannot make: below threshold the structure eigenvalue is
        # pinned to the bulk edge, so lambda/lambda_bulk = 1 is a real feature rather than
        # a level someone chose (see eigenvalue_threshold).
        eig = dict(x_c_emp=np.nan, nu_c_emp=np.nan, gamma_emp=np.nan, rms=np.nan)
        if {"lam", "lam_bulk"}.issubset(d.columns):
            nu_fn = _nu_of_x_for(cfg, cells, level)
            if nu_fn is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = d["lam"].to_numpy(float) / d["lam_bulk"].to_numpy(float)
                eig = threshold_from_eigenvalues(x, ratio, nu_fn)

        # Crossing/peak and transition onset, for the unstandardized fit and then the
        # standardized one under a _std suffix. Both go through _fit_summary, so any
        # difference between them comes from the data and not from the reduction.
        row = dict(sweep=sweep, level=level, level_param=level_param, x_key=x_key,
                   target=target, non_monotone=non_monotone,
                   theory_crossing_pred=x_star, theory_crossing_grid=x_grid_theory,
                   x_c_emp=eig["x_c_emp"], nu_c_emp=eig["nu_c_emp"],
                   gamma_emp=eig["gamma_emp"], eig_fit_rms=eig["rms"],
                   n_replicates=fit["n_points"], n_levels=fit["n_levels"])
        row.update(_fit_summary(fit, target, non_monotone))
        row.update(_fit_summary(fit_std, target, non_monotone, suffix="_std"))

        if not non_monotone:
            # A crossing that only happens in the last sliver of the swept range says where
            # the sweep stopped, not where detection happens, so it is not compared against
            # theory.
            cx, at_edge = row["crossing"], row["crossing_at_edge"]
            row["ratio_measured_over_theory"] = (
                cx / x_star if np.isfinite(x_star) and x_star
                and np.isfinite(cx) and not at_edge else np.nan)

        for k in ("N", "L_eff", "gamma", "nu", "K", "M", "phi", "H", "n_per_deme",
                  "lam", "lam_bulk", "nu_hat"):
            if k in d.columns:
                row[k] = float(d[k].median())
        cross_rows.append(row)

        curve_rows.append(pd.DataFrame(dict(
            sweep=sweep, level=level, level_param=level_param, x_key=x_key,
            x=fit["x_grid"], y_fit=fit["y_fit"], y_lo=fit["y_lo"], y_hi=fit["y_hi"],
            y_fit_std=(fit_std["y_fit"] if fit_std is not None
                       else np.full(len(fit["x_grid"]), np.nan)),
            y_lo_std=(fit_std["y_lo"] if fit_std is not None
                      else np.full(len(fit["x_grid"]), np.nan)),
            y_hi_std=(fit_std["y_hi"] if fit_std is not None
                      else np.full(len(fit["x_grid"]), np.nan)),
            y_theory=y_theory,
            fit_kind=("scam_" + fit["shape"]) if non_monotone else fit["shape"])))

        pts = dict(sweep=sweep, level=level, level_param=level_param, x_key=x_key,
                   x=x, cos2=y)
        if STD_VALUE_KEY in d.columns:
            pts["cos2_std"] = d[STD_VALUE_KEY].to_numpy(float)
        point_rows.append(pd.DataFrame(pts))

        # DF1: the replicate-level measurements, both constructions side by side.
        keep = [c for c in ("sweep", "simID", "sim_index", "level", "level_param", "x_key",
                            "x", x_key, "replicate", "cos", "cos2", "cos_std", "cos2_std",
                            # The measured spectrum: the structure eigenvalue actually
                            # recovered, the first bulk level, and their ratio - 1.
                            "lam", "lam_bulk", "nu_hat",
                            "N", "L_raw", "L_eff", "gamma", "gamma_raw", "mu2", "delta",
                            "nu", "n_demes_sampled", "seed") if c in d.columns]
        empirical_out.append(d[keep].copy())

        # DF2: the theory table. Skipped entirely for a theory-free model, and for the
        # per-replicate sweeps, whose theory travels on the replicate rows instead.
        if fn is not None and not per_replicate_theory:
            tr = _theory_rows_for(cfg, cells, level, level_param, x_key, sweep,
                                  h_by_level.get(f"{level}"), np.unique(x), target)
            if tr:
                theory_out.append(pd.DataFrame(tr))

    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    points = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    empirical = (pd.concat(empirical_out, ignore_index=True) if empirical_out
                 else pd.DataFrame())
    theory = pd.concat(theory_out, ignore_index=True) if theory_out else pd.DataFrame()
    if per_replicate_theory:
        # Each replicate sampled a different set of demes, so its theory is its own: the
        # measurement and the nu it should be compared against travel on the same row, and
        # there is no separate theory table to join back to.
        empirical = _joint_frame(empirical, cells, cfg, h_by_level)
    return pd.DataFrame(cross_rows), curves, points, empirical, theory


def run(sweep_dirs, out, target=DEFAULT_TARGET, n_boot=DEFAULT_N_BOOT,
        n_grid=DEFAULT_N_GRID, seed=0, verbose=True, n_jobs=None):
    """Summarise several sweeps into one set of summary files."""
    crossings, curves, points, empiricals, theories = [], [], [], [], []
    joint = False
    for sweep_dir in sweep_dirs:
        if verbose:
            print(f"summarising {sweep_dir}")
        c, cu, p, e, t = summarise_sweep(sweep_dir, target=target, n_boot=n_boot,
                                         n_grid=n_grid, seed=seed, verbose=verbose,
                                         n_jobs=n_jobs)
        crossings.append(c)
        curves.append(cu)
        points.append(p)
        empiricals.append(e)
        theories.append(t)
        with open(os.path.join(sweep_dir, "config.yaml")) as f:
            joint |= bool(yaml.safe_load(f).get("per_replicate_theory", False))

    # The per-replicate sweeps carry their theory on the measurement rows, so they get one
    # joint table instead of the empirical/theory pair.
    empirical_name = "joint.csv" if joint else "empirical.csv"
    os.makedirs(out, exist_ok=True)
    for name, frames in (("crossings.csv", crossings), ("curves.csv", curves),
                         ("points.csv", points), (empirical_name, empiricals),
                         ("theory.csv", theories)):
        keep = [f for f in frames if len(f)]
        df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
        path = os.path.join(out, name)
        df.to_csv(path, index=False)
        if verbose:
            print(f"wrote {len(df)} rows -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dirs", nargs="+", help="one or more results/sweeps/<name> dirs")
    p.add_argument("--out", required=True, help="output directory for the summary files")
    p.add_argument("--target", type=float, default=DEFAULT_TARGET)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=None,
                   help="processes for the per-curve bootstrap (default: serial)")
    args = p.parse_args()
    # Only the concave (recent-split) sweeps still need scam; the monotone ones are fitted
    # by threshold_fit, which is numpy/scipy only. Check R just for the sweeps that use it,
    # so a missing R does not block the rest.
    needs_scam = []
    for d in args.sweep_dirs:
        cfg_path = os.path.join(d, "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                if yaml.safe_load(f).get("non_monotone", False):
                    needs_scam.append(d)
    if needs_scam and not scam_available():
        raise SystemExit(f"scam is unavailable but {', '.join(needs_scam)} are concave "
                         "sweeps that need it (needs r-base, r-scam and rpy2 in the "
                         "pca_rmt env, and R on PATH -- activate the env first)")
    run(args.sweep_dirs, args.out, target=args.target, n_boot=args.n_boot,
        n_grid=args.n_grid, seed=args.seed, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()

# %%
