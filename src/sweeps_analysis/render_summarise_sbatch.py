#%%
"""
Stage 3: fit every sweep's curves and write results/sweep_summaries/<group>/.

    python src/sweeps_analysis/render_summarise_sbatch.py --local    # run here, group by group
    python src/sweeps_analysis/render_summarise_sbatch.py --submit   # one SLURM job per group

Several sweeps share a summary directory (the two split sweeps land in single_split, and
so on), so the grouping is a property of the project rather than of any one config, and it
lives here. Stage 3 costs a couple of minutes per group; the jobs exist because it needs R
through rpy2 and because the login node's 8G cap makes anything touching the sweep files
uncomfortable interactively.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/sweeps_analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from slurm_config import slurm_settings
from cli_common import PROJECT_ROOT

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sbatch_summarise.template")

# group -> the sweeps that make it up.
GROUPS = {
    "rmt_phenomena": ["split_rmt_phenomena"],
    "single_split": ["split_vary_N", "split_vary_phi"],
    "african_split": ["african_split_vary_N", "african_split_vary_harmonic_phi"],
    "stepping_stone_even": ["ss1d_vary_n", "ss2d_vary_n"],
    "stepping_stone_random": ["ss1d_vary_deme_fraction", "ss2d_vary_deme_fraction"],
    "split_stepping_stone": ["split_ss_vary_N", "split_ss_vary_phi"],
    # standardized, MAF >= 0.05 (supplementary figures)
    "single_split_std": ["split_vary_N_std", "split_vary_phi_std"],
    "african_split_std": ["african_split_vary_N_std", "african_split_vary_harmonic_phi_std"],
    "stepping_stone_even_std": ["ss1d_vary_n_std", "ss2d_vary_n_std"],
}


def sweep_dirs(sweeps):
    """The group's sweep directories (repo-relative) that have been aggregated."""
    return [os.path.join("results", "sweeps", s) for s in sweeps
            if os.path.exists(os.path.join(PROJECT_ROOT, "results", "sweeps", s,
                                           "replicates.csv"))]


def run_local(group, sweeps, n_jobs=None):
    """Run summarise.py for one group in this shell."""
    dirs = sweep_dirs(sweeps)
    if not dirs:
        print(f"  {group}: no sweep has replicates.csv yet, skipped")
        return
    cmd = [sys.executable, os.path.join("src", "sweeps_analysis", "summarise.py"), *dirs,
           "--out", os.path.join("results", "sweep_summaries", group)]
    if n_jobs:
        cmd += ["--n-jobs", str(n_jobs)]
    print(f"=== {group} ===", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def render(group, sweeps, sbatch_dir, verbose=True):
    """Write one group's .sbatch; returns its path, or None if no sweep has data yet."""
    dirs = sweep_dirs(sweeps)
    if not dirs:
        if verbose:
            print(f"  {group}: no sweep has replicates.csv yet, skipped")
        return None

    with open(TEMPLATE) as f:
        text = f.read()
    text = text.format(group=group, sweep_dirs=" ".join(dirs),
                       out=os.path.join("results", "sweep_summaries", group), **slurm_settings())

    os.makedirs(sbatch_dir, exist_ok=True)
    path = os.path.join(sbatch_dir, f"sum_{group}.sbatch")
    with open(path, "w") as f:
        f.write(text)
    if verbose:
        print(f"  {group}: {len(dirs)} sweep(s) -> {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submit", action="store_true", help="sbatch each rendered file")
    p.add_argument("--local", action="store_true", help="run summarise.py here instead")
    p.add_argument("--n-jobs", type=int, default=None, help="processes for --local")
    p.add_argument("--groups", nargs="*", default=sorted(GROUPS),
                   help="which groups to run (default: all)")
    args = p.parse_args()

    if args.local:
        for g in args.groups:
            run_local(g, GROUPS[g], args.n_jobs)
        return

    sbatch_dir = os.path.join(PROJECT_ROOT, "slurm_runs", "summarise")
    paths = [q for q in (render(g, GROUPS[g], sbatch_dir) for g in args.groups) if q]
    if args.submit:
        for path in paths:
            out = subprocess.run(["sbatch", path], cwd=PROJECT_ROOT,
                                 capture_output=True, text=True)
            print(f"    {out.stdout.strip() or out.stderr.strip()}")


if __name__ == "__main__":
    main()

# %%
