"""
Submit every per-figure analysis as its own SLURM job (they read the raw simulations, so
they need the simulations of step 1 to have finished). Each job writes
results/analysis/<name>/, which the plotting scripts read.

    python src/analysis/run_all_analyses.py                    # submit all
    python src/analysis/run_all_analyses.py --no-submit        # render the .sbatch files only
    python src/analysis/run_all_analyses.py --only EffectiveL  # a subset
    python src/analysis/run_all_analyses.py --after 123:124    # hold until those jobs succeed
"""
import argparse
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # src/
from slurm_config import PROJECT_ROOT, slurm_settings

TEMPLATE = os.path.join(_HERE, "sbatch.template")
SBATCH_DIR = os.path.join(PROJECT_ROOT, "slurm_runs", "analysis")
POOL = "--num-replicates 50 --n-workers $SLURM_CPUS_PER_TASK"

# name: (script and arguments, cpus, mem, time, jobs whose output it reads).
# Resources are ~2-3x the measured usage of the runs behind the manuscript figures.
JOBS = {
    "EffectiveL": ("EffectiveL.py", 2, "4G", "01:00:00", ()),
    "SingleSplitSweeps": ("SingleSplitSweeps.py", 1, "24G", "06:00:00", ()),
    "RMTPhenomena": ("RMTPhenomena.py", 8, "20G", "02:00:00", ("SingleSplitSweeps",)),
    "SisterEigenvalue": ("SisterEigenvalue.py", 1, "24G", "12:00:00", ()),
    "OOA3Demography": ("OOA3Demography.py", 1, "8G", "00:30:00", ()),
    "SS_EvenSampling_1D": (f"SteppingStoneEvenSampling.py --dimension 1 {POOL}",
                           16, "16G", "02:00:00", ()),
    "SS_EvenSampling_2D": (f"SteppingStoneEvenSampling.py --dimension 2 {POOL}",
                           16, "32G", "24:00:00", ()),
    "SS_EigenvectorShapes": ("SteppingStoneEigenvectorShapes.py", 1, "4G", "00:30:00",
                             ("SS_EvenSampling_1D", "SS_EvenSampling_2D")),
    "SS_RandomSampling_1D": (f"SteppingStoneRandomSampling.py --dimension 1 {POOL}",
                             16, "8G", "01:00:00", ()),
    "SS_RandomSampling_2D": (f"SteppingStoneRandomSampling.py --dimension 2 {POOL}",
                             16, "16G", "02:00:00", ()),
    "SS_RandomDemo": ("SteppingStoneRandomDemo.py", 1, "8G", "00:30:00",
                      ("SS_RandomSampling_1D", "SS_RandomSampling_2D")),
    "SparseSamplingCeiling": ("SparseSamplingCeiling.py", 1, "2G", "00:15:00", ()),
    "RecentSpatialSpectrum": ("RecentSpatialSpectrum.py", 1, "2G", "00:15:00", ()),
    "RecentSpatialParticipation": ("RecentSpatialParticipation.py", 1, "4G", "00:15:00", ()),
    "RecentSpatialDemography": ("RecentSpatialDemography.py", 16, "16G", "01:00:00", ()),
}


def render(name):
    """Write slurm_runs/analysis/<name>.sbatch and return its path."""
    command, cpus, mem, time, _ = JOBS[name]
    # A process pool gets one BLAS thread per worker; a single large eigh gets them all.
    threads = 1 if "--n-workers" in command else cpus
    with open(TEMPLATE) as f:
        text = f.read().format(name=name, command=f"python src/analysis/{command}",
                               cpus=cpus, mem=mem, time=time, threads=threads,
                               **slurm_settings())
    os.makedirs(SBATCH_DIR, exist_ok=True)
    path = os.path.join(SBATCH_DIR, f"{name}.sbatch")
    with open(path, "w") as f:
        f.write(text)
    return path


def submit(path, after):
    """sbatch from the repo root, held until the job ids in `after` succeed."""
    cmd = ["sbatch"] + ([f"--dependency=afterok:{':'.join(after)}"] if after else []) + [path]
    out = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return re.search(r"(\d+)", out.stdout).group(1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(JOBS), default=list(JOBS),
                   metavar="NAME", help=f"subset of: {', '.join(JOBS)}")
    p.add_argument("--after", default=None, help="job id(s), colon-separated, to wait for")
    p.add_argument("--no-submit", action="store_true", help="render the .sbatch files only")
    args = p.parse_args()

    job_ids = {}
    for name in JOBS:  # insertion order puts every job after the ones it depends on
        if name not in args.only:
            continue
        path = render(name)
        if args.no_submit:
            print(f"  rendered {os.path.relpath(path, PROJECT_ROOT)}")
            continue
        after = [job_ids[d] for d in JOBS[name][4] if d in job_ids]
        after += args.after.split(":") if args.after else []
        job_ids[name] = submit(path, after)
        print(f"  submitted {name} as job {job_ids[name]}"
              + (f" (after {', '.join(after)})" if after else ""))


if __name__ == "__main__":
    main()
