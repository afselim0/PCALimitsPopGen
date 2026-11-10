"""
Cluster-specific SLURM settings, read from slurm_config.yaml at the repository root.

Templated scripts get `account` and `partition` substituted when rendered. To submit a
hand-written .sbatch file with the configured account and partition:

    python src/slurm_config.py src/analysis/RecentSpatialSpectrum.sbatch
"""
import os
import subprocess
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "slurm_config.yaml")


def slurm_settings():
    """{'account': ..., 'partition': ...} from slurm_config.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return {"account": str(cfg["account"]), "partition": str(cfg["partition"])}


def submit(sbatch_path):
    """sbatch with the configured account/partition, run from the repo root."""
    s = slurm_settings()
    cmd = ["sbatch", f"--account={s['account']}", f"--partition={s['partition']}",
           os.path.abspath(sbatch_path)]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        submit(path)
