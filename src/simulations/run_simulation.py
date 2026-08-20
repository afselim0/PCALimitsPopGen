#%%
"""
Run one simulation kind from a YAML config, locally or as one row of a cluster array job.

    python run_simulation.py <config>.yaml                  # full local sweep
    python run_simulation.py <config>.yaml --dry-run         # print the parameter table, simulate nothing
    python run_simulation.py <config>.yaml --task-index 3    # simulate only row 3 (or $SLURM_ARRAY_TASK_ID)
    python run_simulation.py <config>.yaml --estimate        # probe & print mem/time, don't submit
    python run_simulation.py <config>.yaml --submit          # render + submit a SLURM array job

See src/simulations/configs/ for one example YAML per kind, and demographies/__init__.py
for the kind -> module registry. Every kind module implements the same four functions
(build_parameter_table, simulate_one, run, run_one) -- see any module under demographies/
for the contract.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/simulations/
from cli_common import DEFAULT_N_WORKERS, DEFAULT_RANDOM_SEED, default_outdir
from demographies import get_kind


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "kind" not in cfg:
        raise ValueError(f"{path}: missing required top-level key 'kind'")
    return cfg


def resolve_outdir(cfg, cli_outdir):
    if cli_outdir is not None:
        return cli_outdir
    if cfg.get("outdir"):
        return cfg["outdir"]
    return default_outdir(cfg["kind"], cfg.get("name"))


def resolve_task_index(cli_task_index):
    if cli_task_index is not None:
        return cli_task_index
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="path to a simulation YAML config")
    p.add_argument("--task-index", type=int, default=None,
                   help="run only this 0-based row and exit; reads $SLURM_ARRAY_TASK_ID "
                        "if omitted and present")
    p.add_argument("--n-workers", type=int, default=None,
                   help=f"processes per simulation (default: config's n_workers, "
                        f"or {DEFAULT_N_WORKERS})")
    p.add_argument("--outdir", default=None, help="override the config's output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="print the parameter table and exit without simulating")
    p.add_argument("--submit", action="store_true",
                   help="render and sbatch a cluster array job instead of running locally")
    p.add_argument("--no-submit", action="store_true",
                   help="with --submit, only write the .sbatch file, don't call sbatch")
    p.add_argument("--estimate", action="store_true",
                   help="probe and print the mem/time estimate a cluster job would use, "
                        "without writing or submitting anything")
    args = p.parse_args()

    cfg = load_config(args.config)
    kind_module = get_kind(cfg["kind"])
    outdir = resolve_outdir(cfg, args.outdir)
    cfg.setdefault("random_seed", DEFAULT_RANDOM_SEED)

    if args.estimate:
        import shutil
        import tempfile
        from resource_estimate import DEFAULT_CPUS, estimate, format_mem, format_time
        cluster_n_workers = args.n_workers or cfg.get("cluster", {}).get("cpus_per_task", DEFAULT_CPUS)
        probe_dir = tempfile.mkdtemp(prefix="pca_rmt_probe_")
        try:
            est = estimate(cfg, probe_dir, n_workers=cluster_n_workers)
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
        print(f"estimate: cpus_per_task={est['n_workers']}, mem={format_mem(est['mem_gb'])}, "
              f"time={format_time(est['time_seconds'])}")
        return

    n_workers = args.n_workers or cfg.get("n_workers") or DEFAULT_N_WORKERS

    if args.submit:
        from sbatch import render_sbatch, submit
        path = render_sbatch(cfg, args.config, outdir)
        print(f"wrote {path}")
        if not args.no_submit:
            submit(path)
        return

    task_index = resolve_task_index(args.task_index)
    if task_index is not None:
        kind_module.run_one(task_index, cfg, outdir, n_workers)
        return

    kind_module.run(cfg, outdir, n_workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

# %%
