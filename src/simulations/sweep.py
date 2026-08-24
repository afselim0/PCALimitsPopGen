"""
Shared sweep-axis math and generic run/run_one loop helpers, used by every module in
demographies/ so build_parameter_table / run / run_one aren't reimplemented per kind.

    axis_values          log- or linear-spaced grid for a simple min/max/num sweep
    check_no_collisions  raise if simID rounding causes two rows to collide
    run_table            the local sequential loop: simulate every row, skip existing
    run_one_row          the cluster path: simulate a single row by index, skip if it exists
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def axis_values(min_val, max_val, num, spacing="log"):
    """
    num values from min_val to max_val.

    "log" is a plain np.geomspace, "linear" an np.linspace. The stepping-stone kinds do
    not come through here: they sweep 1/M and build their grid with
    stepping_stone_grid.inverse_M_grid.
    """
    if spacing == "log":
        return np.geomspace(min_val, max_val, num)
    if spacing == "linear":
        return np.linspace(min_val, max_val, num)
    raise ValueError(f"spacing must be 'log' or 'linear', got {spacing!r}")


def check_no_collisions(simIDs, what="values"):
    """Raise if any simID appears twice (rounding collision in the sweep grid)."""
    simIDs = pd.Series(simIDs)
    duplicates = simIDs.duplicated()
    if duplicates.any():
        raise ValueError(
            f"{int(duplicates.sum())} {what} collide on simID after rounding "
            f"(e.g. {simIDs[duplicates].iloc[0]}); use fewer values or a wider range."
        )


def _write_params(table, outdir):
    os.makedirs(outdir, exist_ok=True)
    table.to_csv(os.path.join(outdir, "simulation_parameters.csv"), index=False)


def run_table(table, simulate_fn, outdir, dry_run=False, verbose=True, desc=""):
    """
    The local sequential loop shared by every kind's run(): write the parameter table up
    front, then simulate each row not already on disk.

    simulate_fn(row) does the actual simulation and must save "<outdir>/<row.simID>.npy"
    itself, or "<outdir>/<row.outdir_suffix>/<row.simID>.npy" when the table has an
    outdir_suffix column (the two-level stepping-stone kinds) -- see run_one_row for the
    matching per-suffix .npy existence check and CSV split.
    """
    if dry_run:
        return table

    has_suffix = "outdir_suffix" in table.columns
    if has_suffix:
        for suffix, group in table.groupby("outdir_suffix", sort=False):
            _write_params(group, os.path.join(outdir, suffix))
    else:
        _write_params(table, outdir)

    bar = tqdm(table.itertuples(), total=len(table), disable=not verbose, desc=desc)
    for i, r in enumerate(bar, start=1):
        row_dir = os.path.join(outdir, r.outdir_suffix) if has_suffix else outdir
        path = os.path.join(row_dir, f"{r.simID}.npy")
        if os.path.exists(path):
            bar.set_description(f"{desc} {i}/{len(table)} (exists)")
            continue
        bar.set_description(f"{desc} {i}/{len(table)}")
        simulate_fn(r)

    if verbose:
        print(f"wrote {len(table)} simulations to {outdir}")
    return table


def run_one_row(task_index, table, simulate_fn, outdir, verbose=True):
    """
    The cluster path shared by every kind's run_one(): simulate a single row by index,
    skipping if its .npy already exists. table is rebuilt fresh by the caller (cheap, no
    simulation cost) so no shared setup step is needed before array tasks run.
    """
    if not (0 <= task_index < len(table)):
        raise IndexError(f"task_index {task_index} out of range for {len(table)} simulations")
    r = table.iloc[task_index]

    has_suffix = "outdir_suffix" in table.columns
    row_dir = os.path.join(outdir, r.outdir_suffix) if has_suffix else outdir
    os.makedirs(row_dir, exist_ok=True)

    params_path = os.path.join(row_dir, "simulation_parameters.csv")
    if not os.path.exists(params_path):
        group = table[table.outdir_suffix == r.outdir_suffix] if has_suffix else table
        _write_params(group, row_dir)

    out_path = os.path.join(row_dir, f"{r.simID}.npy")
    if os.path.exists(out_path):
        if verbose:
            print(f"task {task_index}: {r.simID} already exists, skipping")
        return

    if verbose:
        print(f"task {task_index}/{len(table)}: {r.simID}")
    simulate_fn(r)
