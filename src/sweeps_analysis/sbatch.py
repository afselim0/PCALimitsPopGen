"""
Render (and optionally submit) a SLURM array job plus its dependent aggregate job.

The --array bound is derived from the task table's row count, after the stale-.npy filter
(see task_table.py) -- sizing it off a raw *.npy glob would launch 100 tasks that can only
fail on both stepping-stone directories.

mem/time come from the config's cluster: block rather than a probe. Unlike a simulation,
whose cost depends on the demography being simulated, a sweep task's cost is set by the
sampling grid the config itself declares, so the config is the right place for the number.

The aggregate job is submitted with --dependency=afterok on the array, so it runs exactly
once every task has succeeded. That needs the array's job id, so submit() parses it out of
sbatch's output rather than discarding it.
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/sweeps_analysis/
from cli_common import PROJECT_ROOT
from task_table import build_task_table

_HERE = os.path.dirname(os.path.abspath(__file__))
ARRAY_TEMPLATE = os.path.join(_HERE, "sbatch.template")
AGGREGATE_TEMPLATE = os.path.join(_HERE, "sbatch_aggregate.template")

DEFAULT_MEM = "4G"
DEFAULT_TIME = "02:00:00"
DEFAULT_CPUS = 1


def render_all(cfg, config_path, outdir, sbatch_outdir=None, verbose=True):
    """Write the array .sbatch and its aggregate .sbatch. Returns both paths, in order."""
    n_tasks = len(build_task_table(cfg, verbose=verbose))
    cluster = cfg.get("cluster", {})
    job_name = cluster.get("job_name", cfg["name"])

    fields = dict(
        job_name=job_name, n_tasks_minus_1=n_tasks - 1,
        cpus_per_task=cluster.get("cpus_per_task", DEFAULT_CPUS),
        mem=cluster.get("mem", DEFAULT_MEM),
        time=cluster.get("time", DEFAULT_TIME),
        config_path=os.path.abspath(config_path), outdir=outdir)

    sbatch_outdir = sbatch_outdir or os.path.join(PROJECT_ROOT, "slurm_runs", "sweeps")
    os.makedirs(sbatch_outdir, exist_ok=True)

    paths = []
    for template, suffix in ((ARRAY_TEMPLATE, ""), (AGGREGATE_TEMPLATE, "_agg")):
        with open(template) as f:
            rendered = f.read().format(**fields)
        path = os.path.join(sbatch_outdir, f"{job_name}{suffix}.sbatch")
        with open(path, "w") as f:
            f.write(rendered)
        paths.append(path)

    if verbose:
        print(f"  {job_name}: --array=0-{n_tasks - 1}, mem={fields['mem']}, "
              f"time={fields['time']}, cpus={fields['cpus_per_task']}")
    return paths


def _submit(sbatch_path, dependency=None):
    """sbatch <path>, run from the repo root so the template's relative paths resolve.
    Returns the job id."""
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(sbatch_path)
    out = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    match = re.search(r"(\d+)", out.stdout)
    if not match:
        raise RuntimeError(f"could not parse a job id from sbatch output: {out.stdout!r}")
    job_id = match.group(1)
    print(f"  submitted {os.path.basename(sbatch_path)} as job {job_id}"
          + (f" (after {dependency})" if dependency else ""))
    return job_id


def submit_all(paths):
    """Submit the array, then the aggregate job depending on it."""
    array_path, aggregate_path = paths
    array_id = _submit(array_path)
    _submit(aggregate_path, dependency=array_id)
    return array_id
