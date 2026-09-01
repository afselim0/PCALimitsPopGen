"""
OOA3 with a recently split sister pair (YRI1/YRI2): PC alignment with the sister axis
against Fst.

The expensive model. prepare() runs pop_coal_times -- an msprime numerical integration per
population pair, ~15-20 min for one t_split. That cost is why the array axis is the
simulation file: one task pays it once for its own t_split, with no shared cache to race
on, exactly as src/analysis/SisterPCAlignment.py's job array already does.

Policies (see models/__init__.py for why these are explicit rather than defaulted):
    axis location   fixed rank SISTER_RANK = 2 (PC3), NOT an overlap search. The deep OOA
                    splits always carry the three largest structure eigenvalues, so the
                    sister axis sits deterministically at rank 2 (measured on all 175 cells
                    of an earlier run, zero exceptions). A search does not change which
                    eigenvector is picked, but it does inflate the no-signal floor -- a max
                    over several candidates finds a larger baseline overlap by chance.
    theory route    exact_h. eps is genuinely nonzero here (tau/tau_bar spans 0.81-1.17),
                    so the flat sqrt(gamma) threshold is not usable: at low Fst it sits
                    below the exact nu_c and would report a large false-positive plateau.
                    A dedicated subsample at replicate 0's seed supplies the per-locus h
                    that pc_recovery discards after reducing to (mu2, delta).
    replicate loop  pc_recovery
"""
import os
import sys
from itertools import combinations_with_replacement

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "analysis"))  # src/analysis/
from exact_sister_nu import analytical_nu, exact_sister_nu, pop_coal_times
from pc_recovery import subsample

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # src/
from threshold_and_overlap import (approx_nu_c, approx_overlap, exact_nu_c, exact_overlap)

SIM_PARAM = "Fst"
X_KEY = "Fst"

POPS = ["CEU", "CHB", "YRI1", "YRI2"]
SISTER = (2, 3)          # YRI1, YRI2 in alphabetical block order
N_PER_POP_FULL = 5000    # per-population sample count in the african_split config
SISTER_RANK = 2          # see the module docstring


def x_value(sim_row, cfg):
    return float(sim_row[SIM_PARAM])


def harmonic_phi(n):
    """H = 2*phi_1*phi_2/(phi_1+phi_2), the harmonic mean of the sister sampling fractions."""
    n = np.asarray(n, dtype=float)
    phi = n / n.sum()
    a, b = SISTER
    return float(2.0 * phi[a] * phi[b] / (phi[a] + phi[b]))


def sister_axis(n):
    """
    The predicted eigenvector u_1, lifted from population space to sample space.

    Population-space entry w_a is spread over that population's n_a samples as
    w_a/sqrt(n_a), which keeps the lifted vector unit-norm. Ported from
    src/analysis/SisterEigenvalue.sister_axis.
    """
    n = np.asarray(n, dtype=int)
    phi = n / n.sum()
    a, b = SISTER
    s = phi[a] + phi[b]

    w = np.zeros(len(n))
    w[a] = np.sqrt(phi[b] / s)
    w[b] = -np.sqrt(phi[a] / s)
    return np.repeat(w / np.sqrt(n), n)


def design_N(N, phi):
    """n = [n_other, n_other, n_sister, n_sister] with each sister at fraction phi of N."""
    n_s = int(round(phi * N))
    n_o = (N - 2 * n_s) // 2
    n = np.array([n_o, n_o, n_s, n_s], dtype=int)
    n[SISTER[1]] += N - n.sum()          # absorb the rounding residual
    return n


def design_phi(phi1, phi2, N):
    """
    n = [n_other, n_other, N*phi1, N*phi2], CEU and CHB splitting the remainder.

    The sisters take their own fractions of the fixed total, so each cell spends a
    different number of samples on the pair; CEU and CHB absorb N*(1-(phi1+phi2))/2 each.
    """
    n1, n2 = int(round(phi1 * N)), int(round(phi2 * N))
    n_o = (N - n1 - n2) // 2
    n = np.array([n_o, n_o, 0, 0], dtype=int)
    n[SISTER[0]], n[SISTER[1]] = n1, n2
    n[0] += N - n.sum()          # absorb the rounding residual
    return n


def _check_feasible(n, label):
    if n.max() > N_PER_POP_FULL:
        raise ValueError(f"design {label} asks for {n.max()} samples from one population "
                         f"but only {N_PER_POP_FULL} were simulated: {list(n)}")
    if n.min() < 2:
        raise ValueError(f"design {label} has a population with < 2 samples: {list(n)}")


def prepare(cfg, sim_row):
    """The msprime coalescent-time integration for this simulation's t_split. ~15-20 min."""
    return {"T": pop_coal_times(float(sim_row["t_split"]))}


def prepare_cached(cfg, cell_row):
    """
    The summariser's cheap stand-in for prepare().

    prepare() is a 15-20 min msprime integration, which a summariser must never pay -- and
    does not need to: theory_overlap uses T only for the diagonal (tau, to form eps), and
    it anchors on the exact nu the runner already recorded. The within-population
    coalescent times are a property of the demography, not of the sampling design, so they
    are reconstructed from what the cell row carries. Falls back to prepare() only if a row
    predates this and lacks the tau columns.
    """
    tau = [cell_row.get(f"tau{i}") for i in range(len(POPS))]
    if all(t is not None and np.isfinite(t) for t in tau):
        T = np.zeros((len(POPS), len(POPS)))
        np.fill_diagonal(T, tau)
        return {"T": T}
    return prepare(cfg, cell_row)


def designs(cfg):
    """[(level, n_vector), ...] -- varying either N at fixed H, or the sister balance."""
    vary = cfg["sampling"]["vary"]
    fixed = cfg["sampling"]["fixed"]
    param, values = vary["param"], vary["values"]

    if param == "N":
        phi = float(fixed["phi_sister"])
        out = []
        for N in values:
            n = design_N(int(N), phi)
            _check_feasible(n, f"N{N}")
            out.append((int(N), n))
        return out

    if param == "phi_sister_pair":
        N = int(fixed["N"])
        out = []
        # Every unordered pair drawn from the list, phi_1 = phi_2 included: (a, b) and
        # (b, a) give the same H and the same design up to relabelling the sisters, so
        # only one of each is run.
        for phi1, phi2 in combinations_with_replacement([float(v) for v in values], 2):
            n = design_phi(phi1, phi2, N)
            _check_feasible(n, f"phi_{phi1}_{phi2}")
            # The level is the harmonic mean H, which is what the figure keys on -- the
            # (phi1, phi2) pair is only how it is realised.
            out.append((harmonic_phi(n), n))
        return out

    raise ValueError(f"african_split: cannot vary {param!r}; expected 'N' or "
                     f"'phi_sister_pair'")


def axes(ctx, design, sim_row, cfg):
    """The sister axis at fixed rank 2, with the exact heteroskedastic threshold."""
    n = np.asarray(design, dtype=int)
    N = int(n.sum())
    T = ctx["T"]
    V = sister_axis(n)[:, None]
    nu = exact_sister_nu(T, n, sister_axis(n))["nu_exact"]

    # eps_a = tau_a/tau_bar - 1, weighted by the sampling proportions: the within-population
    # coalescent times are heterogeneous on this demography, so eps = 0 is not available.
    tau = np.diag(T)
    phi_all = n / N
    eps = tau / float(np.sum(phi_all * tau)) - 1.0

    return dict(V_theory=V, nu=[nu], n=n, n_available=None, indices=[SISTER_RANK],
                select_by_overlap=False, eps=eps, theory_route="exact_h",
                per_replicate=False,
                extra=dict(N=N, H=harmonic_phi(n), t_split=float(sim_row["t_split"]),
                           **{f"tau{i}": float(tau[i]) for i in range(len(tau))},
                           n_sister1=int(n[SISTER[0]]), n_sister2=int(n[SISTER[1]]),
                           n_other=int(n[0]),
                           phi1=float(n[SISTER[0]]) / N, phi2=float(n[SISTER[1]]) / N,
                           nu_approx=N * harmonic_phi(n) * float(sim_row[SIM_PARAM])))


NU_FORM = "analytical_sister"


def _n_vector(design):
    """Per-population sample sizes, from either a recorded cell row or the n vector."""
    if isinstance(design, dict):
        n_o = int(design["n_other"])
        return np.array([n_o, n_o, int(design["n_sister1"]), int(design["n_sister2"])])
    return np.asarray(design, dtype=int)


def theory_context(ctx, design, sim_row, cfg, measured):
    """
    The RMT inputs for this cell, with no nu in them.

    phi is the full K = 4 sampling-proportion vector, NOT the harmonic mean H, which is
    only the curve label. eps is genuinely non-zero here (tau/tau_bar spans 0.81-1.17),
    which is why the flat threshold is not usable for this model.
    """
    n = _n_vector(design)
    N = int(n.sum())
    phi = n / N
    tau = np.diag(ctx["T"])
    try:
        return dict(phi=phi, eps=tau / float(np.sum(phi * tau)) - 1.0,
                    h=measured.get("h"), gamma_raw=float(measured["gamma_raw"]),
                    mu2=float(measured["mu2"]), delta=float(measured["delta"]),
                    N=N, L_eff=float(measured["L_eff"]))
    except (KeyError, TypeError, ValueError):
        return None


def nu_of_x(ctx, design, sim_row, cfg, measured):
    """
    Fst -> nu for the sister axis: nu = eps_0 + (2 N phi_1 phi_2 / s) (tau/tau_bar) Fst,
    evaluated exactly at each Fst from the coalescent-time matrix.

    Equivalently the appendix's eps_0 + N phi_tilde Fst (1 + eps_0), since
    tau/tau_bar = 1 + eps_0 and phi_tilde = 2 phi_1 phi_2 / s.
    """
    n = _n_vector(design)
    T = ctx["T"]
    return lambda fst: analytical_nu(T, n, float(fst))


def theory_overlap(ctx, design, sim_row, cfg, measured):
    """
    Fst -> predicted cos^2, computed rather than fitted or rescaled.

    nu is evaluated exactly at each Fst from the coalescent-time matrix,

        nu(Fst) = eps_0 + (tau/tau_bar) * N * H * Fst

    (exact_sister_nu.analytical_nu). The additive pedestal eps_0 is a constant of the
    demography, so the earlier approach -- rescaling one anchor cell's nu proportionally,
    nu_ref * Fst/Fst_ref -- multiplied eps_0 by Fst/Fst_ref instead of holding it fixed.
    That over-fed nu above the anchor and made the predicted crossing too early, by up to
    a factor of two at small N where eps_0 is the larger share of nu.

    The overlap then goes through the exact route (exact_nu_c/exact_overlap) on the true
    K = 4 phi and eps, with the per-locus weights h when the sidecar supplies them. phi
    here is the full sampling-proportion vector, NOT the harmonic mean H, which is only
    the curve label.
    """
    tc = theory_context(ctx, design, sim_row, cfg, measured)
    nu_fn = nu_of_x(ctx, design, sim_row, cfg, measured)
    if tc is None or nu_fn is None:
        return None
    phi, eps, gamma_raw = tc["phi"], tc["eps"], tc["gamma_raw"]
    h = tc.get("h")

    if h is not None and len(np.asarray(h)):
        h = np.asarray(h, dtype=float)
        nu_c = exact_nu_c(phi, eps, h, gamma_raw)

        def overlap(fst):
            return exact_overlap(nu_fn(fst), phi, eps, h, gamma_raw, nu_c=nu_c)
        return overlap

    mu2, delta = tc["mu2"], tc["delta"]
    nu_c = approx_nu_c(phi, eps, gamma_raw, mu2, delta)

    def overlap(fst):
        return approx_overlap(nu_fn(fst), phi, eps, gamma_raw, mu2, delta, nu_c=nu_c)
    return overlap
