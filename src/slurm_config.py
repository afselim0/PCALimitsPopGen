"""
Cluster-specific SLURM settings, read from slurm_config.yaml at the repository root.

Every templated sbatch script gets `account`, `partition` and `conda_activate` (the two
lines that activate the conda environment) substituted when rendered.
"""
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "slurm_config.yaml")


def slurm_settings():
    """{'account', 'partition', 'conda_activate'} from slurm_config.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    activate = f"source {cfg['conda_sh']}\nconda activate {cfg['conda_env']}"
    return {"account": str(cfg["account"]), "partition": str(cfg["partition"]),
            "conda_activate": activate}
