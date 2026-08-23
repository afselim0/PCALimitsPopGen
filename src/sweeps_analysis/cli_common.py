"""Shared paths and defaults for the sweep-analysis pipeline.

Mirrors src/simulations/cli_common.py: the stage-2 and stage-3 output roots live here so
no analysis module has to rediscover the project root, and nothing imports ANALYSIS_ROOT
from a sweep driver the way src/analysis does today.
"""
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIMULATION_ROOT = os.path.join(_PROJECT_ROOT, "results", "simulations")
SWEEP_ROOT = os.path.join(_PROJECT_ROOT, "results", "sweeps")
SUMMARY_ROOT = os.path.join(_PROJECT_ROOT, "results", "sweep_summaries")
PROJECT_ROOT = _PROJECT_ROOT

DEFAULT_N_REPLICATES = 8
DEFAULT_SEED = 0
DEFAULT_N_GRID = 300
DEFAULT_N_BOOT = 300


def default_outdir(name):
    """results/sweeps/<name>."""
    return os.path.join(SWEEP_ROOT, name)


def resolve_path(path):
    """Absolute path, resolving a repo-relative config entry against the project root."""
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
