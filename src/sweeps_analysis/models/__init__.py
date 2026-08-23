"""Registry of sweep models -> their module name under models/.

Imported lazily (only the requested model), exactly like the simulation kind registry:
african_split pulls in msprime at import time, and the free-theory models should not pay
for that.
"""
import importlib

MODEL_REGISTRY = {
    "split": "split",
    "african_split": "african_split",
    "stepping_stone": "stepping_stone",
    "split_stepping_stone": "split_stepping_stone",
}


def get_model(model):
    """Import and return the models.<module> for a model name."""
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model!r}; must be one of {sorted(MODEL_REGISTRY)}")
    return importlib.import_module(f"models.{MODEL_REGISTRY[model]}")
