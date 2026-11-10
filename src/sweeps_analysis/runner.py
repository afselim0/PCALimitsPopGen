"""
The shared per-task loop: one array task = one simulation, all sampling cells.

Mirrors src/simulations/sweep.py. Every policy that differs between models lives in the
model module (see models/__init__.py for the contract); this file only sequences them:

    prepare() once per simulation  ->  for each sampling cell: axes() -> measure -> rows

prepare() is where african_split spends its 15-20 min msprime integration, which is the
whole reason the array axis is the simulation file rather than the (simulation, cell) pair:
one task pays it once for its own t_split, with no shared cache to race on.

Existence of partial/sim<i>.csv is the completion marker, so a re-submitted array skips
what already finished -- the same idempotency the simulation pipeline relies on.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis"))  # src/analysis/
from pc_recovery import load_simulations, pc_recovery
from crossing_fit import replicate_rows

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/sweeps_analysis/
from cli_common import DEFAULT_N_REPLICATES, DEFAULT_SEED, resolve_path
from models import get_model
from task_table import build_task_table, sampling_grid


def _measure(model, ctx, design, sim_row, cfg, Z, spec, n_replicates, seed):
    """Run one cell, through pc_recovery or the model's own per-replicate loop."""
    if spec.get("per_replicate"):
        # The theoretical axis changes per replicate (configs 6 and 8 redraw the deme
        # subset each time), so pc_recovery -- which takes one fixed V_theory -- cannot
        # express this cell. The model supplies an equivalent loop returning the same keys.
        return model.cell_replicates(ctx, design, sim_row, cfg, Z, n_replicates, seed)
    return pc_recovery(
        Z, spec["n"], spec["V_theory"], indices=spec["indices"],
        L=cfg["fixed"].get("L"), num_replicates=n_replicates, seed=seed,
        nu=spec.get("nu"), eps=spec.get("eps"),
        select_by_overlap=spec.get("select_by_overlap", False),
        n_available=spec.get("n_available"),
        L_polymorphic=cfg["fixed"].get("L_polymorphic", False),
        squared_cosine=bool(cfg.get("squared_cosine", False)))


def run_one_task(index, cfg, outdir, verbose=True, force=False):
    """
    Every sampling cell for a single simulation. Writes partial/sim<index>{,_cells}.csv.

    Skips entirely when the partial already exists, so a partially-failed array can be
    resubmitted without redoing finished work.
    """
    model = get_model(cfg["model"])
    n_replicates = cfg.get("n_replicates", DEFAULT_N_REPLICATES)
    seed = cfg.get("seed", DEFAULT_SEED)

    partial_dir = os.path.join(outdir, "partial")
    os.makedirs(partial_dir, exist_ok=True)
    rep_path = os.path.join(partial_dir, f"sim{index}.csv")
    cell_path = os.path.join(partial_dir, f"sim{index}_cells.csv")
    h_path = os.path.join(partial_dir, f"sim{index}_h.npz")
    if os.path.exists(rep_path) and os.path.exists(cell_path) and not force:
        if verbose:
            print(f"  sim{index}: already done, skipping")
        return

    table = build_task_table(cfg)
    if not 0 <= index < len(table):
        raise IndexError(f"task index {index} out of range for {len(table)} simulations")
    sim_row = table.iloc[index]

    Z = np.load(sim_row["path"], mmap_mode="r")
    ctx = model.prepare(cfg, sim_row)

    x_value = float(sim_row[model.SIM_PARAM])
    cell_rows, rep_rows, h_by_cell = [], [], {}
    for level, design in sampling_grid(cfg):
        spec = model.axes(ctx, design, sim_row, cfg)
        if spec is None:
            # Outside what this simulation can subsample (split_stepping_stone's phi grid
            # asks for more of A than was simulated at some corners). Skipping keeps the
            # cell out of the fit rather than recording a degenerate measurement.
            if verbose:
                print(f"  sim{index}: skipping infeasible cell {level}")
            continue
        res = _measure(model, ctx, design, sim_row, cfg, Z, spec, n_replicates, seed)

        row = dict(sweep=cfg["name"], simID=sim_row["simID"],
                   sim_index=int(sim_row["sim_index"]),
                   level=level, level_param=cfg["sampling"]["vary"]["param"],
                   x_key=model.X_KEY)
        # The swept value and the curve value are each written twice: under the generic
        # name the summariser reads, and under the model's own name so the file is
        # readable on its own. inv_M = 1/M is computed here -- no downstream layer does
        # that arithmetic any more.
        row["x"] = model.x_value(sim_row, cfg)
        row[model.X_KEY] = row["x"]
        row[model.SIM_PARAM] = x_value
        row[cfg["sampling"]["vary"]["param"]] = level

        for key in ("N", "K", "k", "L_raw", "L_eff", "gamma", "gamma_raw", "mu2", "delta",
                    "median_cos", "se_cos", "median_cos_std", "se_cos_std",
                    "median_lam", "median_lam_bulk", "median_nu_hat",
                    "analytical_cos", "procrustes_d",
                    "se_procrustes", "num_replicates", "seed"):
            if key in res:
                row[key] = res[key]
        nu = spec.get("nu")
        row["nu"] = float(np.mean(np.atleast_1d(nu))) if nu is not None else np.nan
        row["nu_c"] = spec.get("nu_c", np.nan)
        row.update(spec.get("extra", {}))
        # A per-replicate model reports quantities it can only know after running (e.g. how
        # many demes the draw actually covered), so those override the cell's own guesses.
        for key in ("n_demes_sampled",):
            if key in res:
                row[key] = res[key]
        # nu likewise, but reduced the same way spec's is: a per-replicate model computes
        # the real nu only after drawing its demes, and may hand back the whole array.
        if res.get("nu") is not None:
            row["nu"] = float(np.mean(np.atleast_1d(res["nu"])))
        cell_rows.append(row)
        # The per-locus weights the exact threshold/overlap route needs. Too wide for a
        # CSV column, so they travel in a sidecar keyed by level, which the summariser
        # loads alongside cells.csv.
        if res.get("h") is not None:
            h_by_cell[f"{level}"] = np.asarray(res["h"], dtype=np.float32)

        # replicate_rows re-adds the x column per row, so it must not also sit in base.
        base = {k: v for k, v in row.items() if k not in ("x", model.X_KEY)}
        # Quantities recorded per replicate. The sample eigenvalues are always
        # per-replicate; nu joins them only on the random-deme path, where each replicate
        # draws its own deme subset.
        per_rep = {}
        if res.get("nu_replicates") is not None:
            per_rep["nu"] = res["nu_replicates"]
        for key, col in (("lam_replicates", "lam"),
                         ("lam_bulk_replicates", "lam_bulk"),
                         ("nu_hat_replicates", "nu_hat")):
            if res.get(key) is not None:
                per_rep[col] = res[key]
        per_rep = per_rep or None
        # squared_cosine configs already reduced their k principal cosines as mean(c^2),
        # so cos_replicates IS cos2 and replicate_rows must not square it a second time.
        reps = replicate_rows(base, row["x"], "x", res["cos_replicates"], seed=seed,
                              cos_std_per_replicate=res.get("cos_replicates_std"),
                              per_replicate_cols=per_rep,
                              already_squared=bool(cfg.get("squared_cosine", False)))
        for r in reps:
            r[model.X_KEY] = r["x"]
        rep_rows.extend(reps)

    pd.DataFrame(cell_rows).to_csv(cell_path, index=False)
    pd.DataFrame(rep_rows).to_csv(rep_path, index=False)
    if h_by_cell:
        np.savez_compressed(h_path, **h_by_cell)
    if verbose:
        print(f"  sim{index} ({sim_row['simID']}): {len(cell_rows)} cells, "
              f"{len(rep_rows)} replicate rows -> {rep_path}", flush=True)


def run_all_tasks(cfg, outdir, n_simulations=None, verbose=True):
    """Serial, in-process run over the first n_simulations -- the dry-run path."""
    table = build_task_table(cfg, verbose=verbose)
    n = len(table) if n_simulations is None else min(n_simulations, len(table))
    for i in range(n):
        run_one_task(i, cfg, outdir, verbose=verbose)
    aggregate(cfg, outdir, n_simulations=n, verbose=verbose)


def aggregate(cfg, outdir, n_simulations=None, verbose=True):
    """Concatenate partial/sim*.csv into replicates.csv and cells.csv."""
    table = build_task_table(cfg)
    n = len(table) if n_simulations is None else min(n_simulations, len(table))
    partial_dir = os.path.join(outdir, "partial")

    for suffix, out_name in (("", "replicates.csv"), ("_cells", "cells.csv")):
        paths = [os.path.join(partial_dir, f"sim{i}{suffix}.csv") for i in range(n)]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(paths)} partial files missing for "
                f"{out_name}, e.g. {os.path.basename(missing[0])}")
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        out = os.path.join(outdir, out_name)
        df.to_csv(out, index=False)
        if verbose:
            print(f"wrote {len(df)} rows -> {out}")

    # One representative per-locus weight vector per curve, for the summariser's exact
    # theory route. The summariser anchors each curve on its median cell, so the sidecar
    # of the corresponding simulation is the one that matches.
    h_paths = [os.path.join(partial_dir, f"sim{i}_h.npz") for i in range(n)]
    h_paths = [p for p in h_paths if os.path.exists(p)]
    if h_paths:
        merged = {}
        with np.load(h_paths[len(h_paths) // 2]) as z:
            for k in z.files:
                merged[k] = z[k]
        h_out = os.path.join(outdir, "h.npz")
        np.savez_compressed(h_out, **merged)
        if verbose:
            print(f"wrote {len(merged)} weight vectors -> {h_out}")
