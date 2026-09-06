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
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "analysis"))  # src/analysis/
from exact_sister_nu import analytical_nu, exact_sister_nu, pop_coal_times
from pc_recovery import subsample

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # src/
from threshold_and_overlap import (Phi_2, approx_nu_c, approx_overlap, crude_nu_alpha,
                                   exact_nu_c, exact_overlap)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from precompute_coal_times import load as load_coal_times

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
    """
    This simulation's coalescent-time matrix, from the stage-0 cache.

    pop_coal_times is a ~15-20 min integration depending only on t_split, so it is
    precomputed once per t_split by precompute_coal_times.py rather than repeated in every
    sweep task (configs 03 and 04 together cover the same 100 t_splits, so integrating
    here cost 200 runs for 100 distinct values). Falls back to integrating in-process if
    the shard is absent, so a sweep still runs without the precompute stage.
    """
    t_split = float(sim_row["t_split"])
    T = load_coal_times(t_split)
    if T is None:
        print(f"  no cached coalescent times for t_split={t_split:.4f}; integrating "
              f"in-process (~15-20 min). Run precompute_coal_times.py to avoid this.",
              flush=True)
        return {"T": pop_coal_times(t_split)}
    return {"T": T}


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

# Whether x_alpha's crude nu_alpha carries the individual-side heteroskedasticity through
# Phi_2 as well as through the eps_0 pedestal. TRUE, and the margin is not close.
# Measured against the numerically-inverted approximate route over all 11 curves of the two
# OOA3 sweeps (misc/test_scripts/x_alpha_check.py), as mean |x_crude/x_approx - 1|:
#
#                        flat 1/nu^2      eps in Phi_2
#     Fst_0 (threshold)      0.73             0.093
#     Fst_0.8 (crossing)     0.53             0.34
#
# The flat route does not merely lose accuracy here, it changes sign: on this demography
# eps spans 0.81-1.17 of tau_bar, and at N = 100 the flat sqrt(N/L_eff) = 0.094 falls BELOW
# eps_0, so subtracting the pedestal returns Fst_0 = -0.0045 -- the sister axis declared
# detectable at zero divergence. Solving gamma_eff Phi_2(nu) = 1 - alpha on the real K = 4
# phi/eps instead keeps nu_alpha above the pole at eps_max by construction, so Fst_0 stays
# positive at every N. One bisection, ~0.1 ms.
# Set False to recover the flat sqrt(N/(L_eff(1-alpha))) of the other three demographies.
X_ALPHA_EPS_IN_PHI2 = True


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


def x_alpha(ctx, design, sim_row, cfg, measured, eps_in_phi2=X_ALPHA_EPS_IN_PHI2):
    """
    alpha -> the Fst at which the crude (L_eff) route predicts cos^2 = alpha, in closed form.

    nu(Fst) = eps_0 + (1 + eps_0) N phi_tilde Fst carries an additive pedestal, so
    inverting nu = nu_alpha subtracts it before dividing:

        Fst_alpha = (nu_alpha - eps_0) / ((1 + eps_0) N phi_tilde)
                  = 1 / ((1 + eps_0) phi_tilde sqrt(N L_eff (1 - alpha)))
                    - eps_0 / (N phi_tilde (1 + eps_0))

    with eps_0 = tau_sister/tau_bar - 1 and phi_tilde the harmonic mean of the sisters'
    sampling proportions. The pedestal is subtracted from nu, not from Fst, which is why
    the second term has no sqrt: it is the Fst that eps_0 alone already buys.

    TWO ROUTES TO nu_alpha, selected by eps_in_phi2 (module default X_ALPHA_EPS_IN_PHI2):

      False  nu_alpha = sqrt(N / (L_eff (1 - alpha))) -- threshold_and_overlap.crude_nu_alpha.
             eps enters only through the pedestal in the inversion above.
      True   nu_alpha solves gamma_eff Phi_2(nu_alpha, phi, eps) = 1 - alpha on the full
             K = 4 phi/eps, i.e. eps also enters the threshold itself. This is the same
             equation the crude route already solves, with its flat Phi_2 = 1/nu^2 replaced
             by the real one; it costs one bisection and is what the approximate route uses
             at alpha = 0.

    See misc/test_scripts/x_alpha_eps_check.py for the comparison of the two against the
    numerically inverted approximate route.
    """
    tc = theory_context(ctx, design, sim_row, cfg, measured)
    if tc is None:
        return None
    n = _n_vector(design)
    N = int(n.sum())
    phi = n / N
    tau = np.diag(ctx["T"])
    eps_0 = float(tau[SISTER[0]] / float(np.sum(phi * tau)) - 1.0)
    a, b = SISTER
    phi_t = 2.0 * float(phi[a]) * float(phi[b]) / (float(phi[a]) + float(phi[b]))
    L_eff = tc["L_eff"]

    def fst_of_alpha(alpha):
        if eps_in_phi2:
            nu_a = _nu_alpha_eps(float(alpha), N, L_eff, tc["phi"], tc["eps"])
        else:
            nu_a = crude_nu_alpha(float(alpha), N, L_eff)
        return (nu_a - eps_0) / ((1.0 + eps_0) * N * phi_t)

    return fst_of_alpha


def _nu_alpha_eps(alpha, N, L_eff, phi, eps):
    """
    nu at which the eps-aware crude overlap reaches alpha: gamma_eff Phi_2(nu) = 1 - alpha.

    The flat route takes Phi_2 = 1/nu^2, which is only right when every eps_a is 0; here
    tau/tau_bar spans 0.81-1.17, so Phi_2 differs from 1/nu^2 by tens of percent near the
    pole. Phi_2 falls monotonically from +inf at nu = max(eps) to 0, so the root is unique
    and bisection on a bracket grown outward from the pole needs no starting guess.
    """
    ge = float(N) / float(L_eff)
    target = 1.0 - float(alpha)
    eps = np.asarray(eps, dtype=float)
    e_max = float(eps.max())
    scale = max(abs(e_max), np.sqrt(ge), 1.0)

    lo = e_max + scale * 1e-12
    for _ in range(400):
        if ge * Phi_2(lo, phi, eps) > target:
            break
        lo = e_max + (lo - e_max) / 2.0
    hi = e_max + 2.0 * scale
    for _ in range(400):
        if ge * Phi_2(hi, phi, eps) < target:
            break
        hi = e_max + 2.0 * (hi - e_max)
    return float(brentq(lambda v: ge * Phi_2(v, phi, eps) - target, lo, hi,
                        xtol=1e-13, rtol=1e-12))


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
