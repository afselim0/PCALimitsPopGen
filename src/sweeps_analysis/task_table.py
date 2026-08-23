"""
The array's task table: one row per simulation file, plus the sampling grid.

The task table is cheap and deterministic, so every array task rebuilds it independently
rather than reading a coordinator's file -- the same property that lets the simulation
pipeline run without shared setup.

Stale .npy files. load_simulations globs *.npy and left-merges the parameter CSV, so a
directory holding output from a superseded grid yields rows whose parameters are all NaN.
Both stepping-stone directories are in that state (200 .npy against 100 CSV rows, with a
simulation_parameters_OLD_mgrid.csv still beside the 1D one). Sizing an array off the raw
glob would launch 100 tasks that can only fail, so rows without the model's swept parameter
are dropped here and the count is reported.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis"))  # src/analysis/
from pc_recovery import load_simulations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/sweeps_analysis/
from cli_common import resolve_path
from models import get_model


def build_task_table(cfg, verbose=False):
    """
    One row per usable simulation file: sim_index, simID, path, and the swept parameter.

    Returns a DataFrame whose length is the array size. Rows whose SIM_PARAM is missing are
    dropped (see the module docstring).
    """
    model = get_model(cfg["model"])
    simdir = resolve_path(cfg["simdir"])
    params, _ = load_simulations(simdir)

    if model.SIM_PARAM not in params.columns:
        raise KeyError(f"{simdir}: no {model.SIM_PARAM!r} column in the parameter table; "
                       f"have {sorted(c for c in params.columns if c != 'path')}")

    n_before = len(params)
    params = params[params[model.SIM_PARAM].notna()].reset_index(drop=True)
    n_dropped = n_before - len(params)
    if n_dropped and verbose:
        print(f"  dropped {n_dropped} of {n_before} .npy files in {simdir} with no "
              f"{model.SIM_PARAM} in the parameter CSV (stale output from a superseded grid)")
    if params.empty:
        raise ValueError(f"{simdir}: no simulation files with a usable "
                         f"{model.SIM_PARAM}")

    params.insert(0, "sim_index", np.arange(len(params)))
    return params


def sampling_grid(cfg):
    """[(level_value, design), ...] for the varied sampling parameter."""
    return get_model(cfg["model"]).designs(cfg)


def describe(cfg, verbose=True):
    """The dry-run report: task count, sampling grid, and the array bound they imply."""
    table = build_task_table(cfg, verbose=verbose)
    grid = sampling_grid(cfg)
    model = get_model(cfg["model"])
    if verbose:
        print(f"  model={cfg['model']}  swept={model.SIM_PARAM}  x_key={model.X_KEY}")
        print(f"  {len(table)} simulations -> --array=0-{len(table) - 1}")
        print(f"  {len(grid)} sampling cells on {cfg['sampling']['vary']['param']}: "
              f"{[lv for lv, _ in grid]}")
        print(f"  {cfg.get('n_replicates')} replicates per cell -> "
              f"{len(table) * len(grid) * cfg.get('n_replicates', 8)} measurements total")
    return table, grid
