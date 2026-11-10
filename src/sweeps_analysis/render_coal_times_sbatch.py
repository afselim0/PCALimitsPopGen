"""
Render (and optionally submit) the stage-0 coalescent-time array plus its aggregate job.

Mirrors sbatch.py's array + dependent-aggregate shape. The array size is the number of
simulations in the OOA3 directory, one t_split each.
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from slurm_config import slurm_settings
from cli_common import PROJECT_ROOT
from precompute_coal_times import t_splits

_HERE = os.path.dirname(os.path.abspath(__file__))
ARRAY_TEMPLATE = os.path.join(_HERE, "sbatch_coal_times.template")
AGG_TEMPLATE = os.path.join(_HERE, "sbatch_coal_times_agg.template")


def render(simdir, job_name="coal_times", mem="4G", time="01:00:00", outdir=None):
    n = len(t_splits(simdir))
    fields = dict(job_name=job_name, n_tasks_minus_1=n - 1, mem=mem, time=time,
                  simdir=simdir, **slurm_settings())
    outdir = outdir or os.path.join(PROJECT_ROOT, "slurm_runs", "coal_times")
    os.makedirs(outdir, exist_ok=True)

    paths = []
    for template, suffix in ((ARRAY_TEMPLATE, ""), (AGG_TEMPLATE, "_agg")):
        with open(template) as f:
            rendered = f.read().format(**fields)
        path = os.path.join(outdir, f"{job_name}{suffix}.sbatch")
        with open(path, "w") as f:
            f.write(rendered)
        paths.append(path)
    print(f"  {job_name}: --array=0-{n - 1}, mem={mem}, time={time}")
    return paths


def _submit(path, dependency=None):
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(path)
    out = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    job_id = re.search(r"(\d+)", out.stdout).group(1)
    print(f"  submitted {os.path.basename(path)} as job {job_id}")
    return job_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simdir", default="results/simulations/african_split")
    ap.add_argument("--job-name", default="coal_times")
    ap.add_argument("--mem", default="4G")
    ap.add_argument("--time", default="01:00:00")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    array_path, agg_path = render(args.simdir, args.job_name, args.mem, args.time)
    if args.submit:
        job_id = _submit(array_path)
        _submit(agg_path, dependency=job_id)
    else:
        print(f"  rendered {array_path}\n  rendered {agg_path}\n"
              f"  (pass --submit to queue them)")


if __name__ == "__main__":
    main()
