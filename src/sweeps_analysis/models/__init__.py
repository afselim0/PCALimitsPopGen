"""Registry of sweep models -> their module name under models/.

Imported lazily (only the requested model), exactly like the simulation kind registry:
african_split pulls in msprime at import time, and the free-theory models should not pay
for that.

CONTRACT. Every model defines SIM_PARAM, X_KEY, x_value, prepare, designs and axes, and
optionally cell_replicates (when its spec sets per_replicate) and prepare_cached (to spare
the summariser an expensive prepare).

Theory is optional and is supplied as three functions that stand or fall together:

    theory_context(ctx, design, sim_row, cfg, measured) -> dict | None
        The RMT inputs -- phi, eps, h, gamma_raw, mu2, delta, N, L_eff. Carries NO nu, so
        that a critical value computed from it cannot accidentally depend on the swept
        parameter.
    nu_of_x(ctx, design, sim_row, cfg, measured) -> callable | None
        The swept value -> the structure eigenvalue at that value.
    theory_overlap(ctx, design, sim_row, cfg, measured) -> callable | None
        The swept value -> predicted cos^2; the composition of the two above.

A model that defines NONE of them is theory-free: the summariser writes its empirical
outputs and no theory table, and leaves y_theory NaN. split_stepping_stone is the one such
model -- see the note at the foot of that module for why.
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
