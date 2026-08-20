"""Shared CLI defaults for run_simulation.py -- replaces the copy-pasted argparse block
that used to live in every simulations script."""
import os
from multiprocessing import cpu_count

DEFAULT_N_WORKERS = max(cpu_count() - 3, 1)
DEFAULT_RANDOM_SEED = 42

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMULATION_ROOT = os.path.join(_PROJECT_ROOT, "results", "simulations")


def default_outdir(kind, name=None):
    """results/simulations/<kind>[/<name>], name omitted when it's "default"."""
    if name and name != "default":
        return os.path.join(SIMULATION_ROOT, kind, name)
    return os.path.join(SIMULATION_ROOT, kind)
