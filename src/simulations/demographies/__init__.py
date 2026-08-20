"""
Registry of simulation kinds -> their module name under demographies/.

Modules are imported lazily (only the requested kind), not all six up front, since each
pulls in msprime/stdpopsim at import time.
"""
import importlib

KIND_REGISTRY = {
    "panmictic": "panmictic",
    "split": "split",
    "african_split": "african_split",
    "stepping_stone_1d": "stepping_stone_1d",
    "stepping_stone_2d": "stepping_stone_2d",
    "split_stepping_stone": "split_stepping_stone",
}


def get_kind(kind):
    """Import and return the demographies.<module> for a kind name."""
    if kind not in KIND_REGISTRY:
        raise ValueError(f"unknown kind {kind!r}; must be one of {sorted(KIND_REGISTRY)}")
    return importlib.import_module(f"demographies.{KIND_REGISTRY[kind]}")
