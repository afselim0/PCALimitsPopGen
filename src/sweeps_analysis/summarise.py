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

Curves are fitted with a shape-constrained spline (R's `scam` via rpy2), not a parametric
sigmoid: the constraint is only the qualitative shape we can defend -- monotone increasing
for the detection sweeps, concave for the recent-split ones that turn over -- and scam
supplies a Bayesian confidence band directly, so the drawn curve and the reported crossing
come from the same fit. See src/analysis/scam_fit.py.

The 99% band is scam's own se.fit, not a bootstrap, so a curve costs one fit. The crossing
CI is read off that band: where its upper and lower edges reach the target. 99% rather than
95% because these curves are fitted through ~800 replicate points, which makes a 95% band
about a line-width wide -- see scam_fit.DEFAULT_LEVEL.
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
from scam_fit import crossing_with_ci, peak_with_ci, scam_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/sweeps_analysis/
from cli_common import DEFAULT_N_BOOT, DEFAULT_N_GRID
from models import get_model

VALUE_KEY = "cos2"
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


def summarise_sweep(sweep_dir, target=DEFAULT_TARGET, n_boot=DEFAULT_N_BOOT,
                    n_grid=DEFAULT_N_GRID, seed=0, verbose=True):
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

    h_by_level = _load_h(sweep_dir)
    cross_rows, curve_rows, point_rows = [], [], []
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
        shape = "cv" if non_monotone else "mpi"
        fit = scam_fit(x, y, shape=shape, n_grid=n_grid)
        if fit is None:
            if verbose:
                print(f"  {sweep} level={level}: no identifiable fit, skipped")
            continue

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

        if non_monotone:
            # No crossing is reported: the curve turns over, so "the x where it first
            # reaches 0.9" is not a well-defined summary of it. The peak is.
            pk = peak_with_ci(fit)
            row = dict(
                sweep=sweep, level=level, level_param=level_param, x_key=x_key,
                target=target, non_monotone=True, shape=fit["shape"], edf=fit["edf"],
                peak_y=pk["peak_y"], peak_y_lo=pk["peak_y_lo"], peak_y_hi=pk["peak_y_hi"],
                peak_x=pk["peak_x"], peak_x_lo=pk["peak_x_lo"], peak_x_hi=pk["peak_x_hi"],
                reaches_target=bool(pk["peak_y"] >= target),
                theory_crossing_pred=x_star, theory_crossing_grid=x_grid_theory,
                n_replicates=fit["n_points"], n_levels=fit["n_levels"])
        else:
            # Crossing and CI both come off the drawn curve and its band, so the marker
            # the figure puts on the line is exactly this number.
            cx, cx_lo, cx_hi, at_edge = crossing_with_ci(fit, target)
            # y at the crossing is the target by construction; carried so the panel can
            # place its marker without re-deriving it.
            row = dict(
                sweep=sweep, level=level, level_param=level_param, x_key=x_key,
                target=target, non_monotone=False, shape=fit["shape"], edf=fit["edf"],
                crossing=cx, crossing_lo=cx_lo, crossing_hi=cx_hi,
                crossing_y=target, crossing_at_edge=at_edge,
                theory_crossing_pred=x_star, theory_crossing_grid=x_grid_theory,
                # A crossing that only happens in the last sliver of the swept range says
                # where the sweep stopped, not where detection happens, so it is not
                # compared against theory.
                ratio_measured_over_theory=(cx / x_star
                                            if np.isfinite(x_star) and x_star
                                            and np.isfinite(cx) and not at_edge
                                            else np.nan),
                n_replicates=fit["n_points"], n_levels=fit["n_levels"])
        for k in ("N", "L_eff", "gamma", "nu", "K", "M", "phi", "H", "n_per_deme"):
            if k in d.columns:
                row[k] = float(d[k].median())
        cross_rows.append(row)

        curve_rows.append(pd.DataFrame(dict(
            sweep=sweep, level=level, level_param=level_param, x_key=x_key,
            x=fit["x_grid"], y_fit=fit["y_fit"], y_lo=fit["y_lo"], y_hi=fit["y_hi"],
            y_theory=y_theory,
            fit_kind=f"scam_{fit['shape']}")))

        point_rows.append(pd.DataFrame(dict(
            sweep=sweep, level=level, level_param=level_param, x_key=x_key,
            x=x, cos2=y)))

    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    points = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    return pd.DataFrame(cross_rows), curves, points


def run(sweep_dirs, out, target=DEFAULT_TARGET, n_boot=DEFAULT_N_BOOT,
        n_grid=DEFAULT_N_GRID, seed=0, verbose=True):
    """Summarise several sweeps into one set of summary files."""
    crossings, curves, points = [], [], []
    for sweep_dir in sweep_dirs:
        if verbose:
            print(f"summarising {sweep_dir}")
        c, cu, p = summarise_sweep(sweep_dir, target=target, n_boot=n_boot,
                                   n_grid=n_grid, seed=seed, verbose=verbose)
        crossings.append(c)
        curves.append(cu)
        points.append(p)

    os.makedirs(out, exist_ok=True)
    for name, frames in (("crossings.csv", crossings), ("curves.csv", curves),
                         ("points.csv", points)):
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
    args = p.parse_args()
    run(args.sweep_dirs, args.out, target=args.target, n_boot=args.n_boot,
        n_grid=args.n_grid, seed=args.seed)


if __name__ == "__main__":
    main()

# %%
