#%%
"""
Run one sweep analysis from a YAML config, locally or as one row of a cluster array job.

    python run_sweep_analysis.py <config>.yaml                 # full local sweep
    python run_sweep_analysis.py <config>.yaml --dry-run       # print the task table, analyse nothing
    python run_sweep_analysis.py <config>.yaml --task-index 3  # analyse only simulation 3
    python run_sweep_analysis.py <config>.yaml --aggregate     # concatenate partial/ outputs
    python run_sweep_analysis.py <config>.yaml --submit        # render + submit a SLURM array job

The array is over simulation files: task i loads one .npy, runs every sampling cell for it,
and writes partial/sim<i>.csv. See src/sweeps_analysis/configs/ for one YAML per sweep and
models/__init__.py for the model registry. Every model module implements the same contract
(SIM_PARAM, X_KEY, x_value, prepare, designs, axes, theory_overlap, and optionally
cell_replicates) -- see any module under models/.
"""
import argparse
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/sweeps_analysis/
from cli_common import DEFAULT_N_REPLICATES, DEFAULT_SEED, default_outdir, resolve_path
from models import get_model


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("model", "name", "simdir", "sampling"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required top-level key {key!r}")
    cfg.setdefault("fixed", {})
    cfg.setdefault("n_replicates", DEFAULT_N_REPLICATES)
    cfg.setdefault("seed", DEFAULT_SEED)
    return cfg


def resolve_outdir(cfg, cli_outdir):
    if cli_outdir is not None:
        return cli_outdir
    if cfg.get("outdir"):
        return resolve_path(cfg["outdir"])
    return default_outdir(cfg["name"])


def resolve_task_index(cli_task_index):
    if cli_task_index is not None:
        return cli_task_index
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="path to a sweep-analysis YAML config")
    p.add_argument("--task-index", type=int, default=None,
                   help="analyse only this 0-based simulation and exit; reads "
                        "$SLURM_ARRAY_TASK_ID if omitted and present")
    p.add_argument("--n-simulations", type=int, default=None,
                   help="limit the serial path to the first N simulations, for a dry run")
    p.add_argument("--outdir", default=None, help="override the config's output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="print the task table and sampling grid, analyse nothing")
    p.add_argument("--aggregate", action="store_true",
                   help="concatenate partial/ outputs into replicates.csv / cells.csv")
    p.add_argument("--submit", action="store_true",
                   help="render and sbatch a cluster array job instead of running locally")
    p.add_argument("--no-submit", action="store_true",
                   help="with --submit, only write the .sbatch files, don't call sbatch")
    p.add_argument("--force", action="store_true",
                   help="recompute cells whose partial file already exists")
    p.add_argument("--after", default=None, metavar="JOBID",
                   help="with --submit, hold the array until this job id succeeds "
                        "(the OOA3 configs use it to wait on the coalescent-time array)")
    args = p.parse_args()

    cfg = load_config(args.config)
    outdir = resolve_outdir(cfg, args.outdir)

    if args.dry_run:
        from task_table import describe
        print(f"=== {cfg['name']} ({os.path.basename(args.config)}) ===")
        print(f"  simdir  {resolve_path(cfg['simdir'])}")
        print(f"  outdir  {outdir}")
        describe(cfg)
        return

    if args.submit:
        from sbatch import render_all, submit_all
        paths = render_all(cfg, args.config, outdir)
        for path in paths:
            print(f"wrote {path}")
        if not args.no_submit:
            submit_all(paths, array_dependency=args.after)
        return

    os.makedirs(outdir, exist_ok=True)
    # The resolved config is copied in for provenance, and because summarise.py reads it
    # back. Array tasks skip the copy: 100 of them writing one path concurrently is a race
    # for no benefit, and the aggregate job (and any serial run) writes it anyway.
    if resolve_task_index(args.task_index) is None:
        shutil.copyfile(args.config, os.path.join(outdir, "config.yaml"))

    from runner import aggregate, run_all_tasks, run_one_task
    if args.aggregate:
        aggregate(cfg, outdir, n_simulations=args.n_simulations)
        return

    task_index = resolve_task_index(args.task_index)
    if task_index is not None:
        run_one_task(task_index, cfg, outdir, force=args.force)
        return

    run_all_tasks(cfg, outdir, n_simulations=args.n_simulations)


if __name__ == "__main__":
    main()

# %%
