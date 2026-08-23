"""
Two-population split: PC1 recovery against Fst.

The free-theory model. T is closed form, so prepare() costs nothing and the whole array
task is subsampling and eigendecomposition.

Policies (see models/__init__.py for why these are explicit rather than defaulted):
    axis location   fixed rank [0] -- the split axis is PC1 by construction
    theory route    approx_moments -- eps = 0 exactly here (the split is symmetric, so both
                    within-population coalescent times are 1), which is the special case of
                    the same code path, not a separate formula
    replicate loop  pc_recovery
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "analysis"))  # src/analysis/
from pc_recovery import theoretical_eigenvectors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # src/
from threshold_and_overlap import approx_nu_c, approx_overlap

SIM_PARAM = "Fst"
X_KEY = "Fst"


def x_value(sim_row, cfg):
    """Fst is already increasing in detectability, so the x axis is the raw value."""
    return float(sim_row[SIM_PARAM])


def split_T(Fst):
    """Coalescent time matrix for a symmetric two-population split, in units of 2*Ne."""
    return np.array([[1.0, 1.0 + Fst], [1.0 + Fst, 1.0]])


def n_from_phi(N, phi):
    """
    Per-population sample sizes for a total N split at fraction phi.

    The smaller population is rounded to at least 1 and the remainder goes to the larger,
    so sum(n) == N exactly and no cell is degenerate at the small-phi end. Ported verbatim
    from src/analysis/SingleSplitSweeps.n_from_phi so the two pipelines produce identical
    designs (and therefore identical subsamples at a shared seed).
    """
    nA = max(int(round(phi * N)), 1)
    return np.array([nA, N - nA])


def prepare(cfg, sim_row):
    """Closed-form T for this simulation's Fst. Free."""
    return {"T": split_T(float(sim_row[SIM_PARAM]))}


def designs(cfg):
    """[(level, n_vector), ...] -- varying either N at fixed phi, or phi at fixed N."""
    vary = cfg["sampling"]["vary"]
    fixed = cfg["sampling"]["fixed"]
    param, values = vary["param"], vary["values"]
    if param == "N":
        phi = float(fixed["phi"])
        return [(int(N), n_from_phi(int(N), phi)) for N in values]
    if param == "phi":
        N = int(fixed["N"])
        return [(float(phi), n_from_phi(N, float(phi))) for phi in values]
    raise ValueError(f"split: cannot vary {param!r}; expected 'N' or 'phi'")


def axes(ctx, design, sim_row, cfg):
    """The single structure axis, at fixed rank 0."""
    n = np.asarray(design, dtype=int)
    V, nu = theoretical_eigenvectors(ctx["T"], n, indices=[0])
    N = int(n.sum())
    return dict(V_theory=V, nu=nu, n=n, n_available=None, indices=[0],
                select_by_overlap=False, eps=np.zeros(len(n)),
                theory_route="approx_moments", per_replicate=False,
                extra=dict(N=N, phi=float(n.min() / N),
                           n_a=int(n[0]), n_b=int(n[1])))


def theory_overlap(ctx, design, sim_row, cfg, measured):
    """
    Fst -> predicted cos^2, via the two-moment route.

    nu = 2*N*phi*(1-phi)*Fst is exact for the symmetric split, so the closure evaluates it
    directly rather than regressing nu on Fst. gamma_raw/mu2/delta come from what the
    subsamples actually contained, so nothing here is fitted.
    """
    # design is the recorded cell row when called from the summariser, and the n vector
    # when called from the runner; n_a/n_b are recorded per cell precisely so the closure
    # can be rebuilt from a CSV row alone.
    n = (np.array([int(design["n_a"]), int(design["n_b"])])
         if isinstance(design, dict) else np.asarray(design, dtype=int))
    N = int(n.sum())
    phi_a = float(n[0]) / N
    phi = n / N
    eps = np.zeros(len(n))
    gamma_raw = float(measured["gamma_raw"])
    mu2, delta = float(measured["mu2"]), float(measured["delta"])
    nu_c = approx_nu_c(phi, eps, gamma_raw, mu2, delta)

    def overlap(fst):
        nu = 2.0 * N * phi_a * (1.0 - phi_a) * float(fst)
        return approx_overlap(nu, phi, eps, gamma_raw, mu2, delta, nu_c=nu_c)
    return overlap
