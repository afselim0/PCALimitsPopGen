"""
Stage 0 for the OOA3 sweeps: the msprime coalescent-time integration, once per t_split.

pop_coal_times is a ~15-20 min numerical integration that depends ONLY on t_split, not on
any sampling design. Running it inside the sweep array meant configs 03 and 04 each paid
it in all 100 of their tasks -- 200 integrations for 100 distinct values, and every one of
them recomputed on a resubmit.

This array computes each t_split once and writes it to its own .npy under
results/analysis/coal_times/, keyed by the same "%.4e" string the old npz cache used.
models/african_split.prepare() then reads that file instead of integrating. One shard per
t_split rather than a single shared .npz because 100 array tasks writing one file would
race; the aggregate step collects them into coal_times.npz for anything that wants the
bundle.

Usage:
    python precompute_coal_times.py --simdir results/simulations/african_split \
        --task-index $SLURM_ARRAY_TASK_ID
    python precompute_coal_times.py --simdir ... --aggregate
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis"))  # src/analysis/
from exact_sister_nu import pop_coal_times

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cli_common import PROJECT_ROOT, resolve_path

CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "analysis", "coal_times")


def key(t_split):
    """The cache key, matching exact_sister_nu.cached_coal_times' format."""
    return f"{float(t_split):.4e}"


def shard_path(t_split, cache_dir=CACHE_DIR):
    return os.path.join(cache_dir, f"t{key(t_split)}.npy")


def load(t_split, cache_dir=CACHE_DIR):
    """The cached K x K matrix for this t_split, or None if it has not been computed."""
    path = shard_path(t_split, cache_dir)
    return np.load(path) if os.path.exists(path) else None


def t_splits(simdir):
    """The t_split of every simulation, in the parameter table's order."""
    params = pd.read_csv(os.path.join(resolve_path(simdir), "simulation_parameters.csv"))
    return params["t_split"].astype(float).tolist()


def run_one(index, simdir, cache_dir=CACHE_DIR, force=False):
    """One t_split. Existence of the shard is the completion marker, as in runner.py."""
    t = t_splits(simdir)[index]
    path = shard_path(t, cache_dir)
    if os.path.exists(path) and not force:
        print(f"  t_split={t:.4f} already cached at {path}")
        return path

    os.makedirs(cache_dir, exist_ok=True)
    print(f"  integrating t_split={t:.4f} ...", flush=True)
    T = pop_coal_times(t)
    # Write via a temp file so a killed task cannot leave a half-written shard that the
    # existence check would then treat as complete. The suffix stays ".npy" because
    # np.save APPENDS ".npy" to any path lacking it -- a bare ".tmp<pid>" suffix makes it
    # write "<name>.npy.tmp<pid>.npy" while os.replace looks for "<name>.npy.tmp<pid>",
    # so every task fails at the rename after doing the full 15 min integration.
    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, T)
    os.replace(tmp, path)
    print(f"  wrote {path}")
    return path


def aggregate(simdir, cache_dir=CACHE_DIR, out=None):
    """Collect the shards into one .npz, and report anything still missing."""
    ts = t_splits(simdir)
    found, missing = {}, []
    for t in ts:
        T = load(t, cache_dir)
        if T is None:
            missing.append(t)
        else:
            found[key(t)] = T

    if missing:
        raise SystemExit(f"{len(missing)} of {len(ts)} t_splits are missing, "
                         f"e.g. {missing[:5]}")

    out = out or os.path.join(cache_dir, "coal_times.npz")
    np.savez(out, **found)
    print(f"  aggregated {len(found)} t_splits into {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simdir", default="results/simulations/african_split")
    ap.add_argument("--task-index", type=int)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--count", action="store_true", help="print the array size and exit")
    args = ap.parse_args()

    if args.count:
        print(len(t_splits(args.simdir)))
    elif args.aggregate:
        aggregate(args.simdir, args.cache_dir)
    elif args.task_index is not None:
        run_one(args.task_index, args.simdir, args.cache_dir, args.force)
    else:
        ap.error("pass --task-index, --aggregate or --count")


if __name__ == "__main__":
    main()
