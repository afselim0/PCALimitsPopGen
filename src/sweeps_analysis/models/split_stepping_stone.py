"""
Recent spatial structure: a panmictic population A plus a K x K grid B split off t_split
generations ago. PC recovery of B's spatial axis against 1/M.

phi here is the *total* sample fraction from the structured demes B -- not the
fraction-of-demes-sampled phi in models/stepping_stone.py, which is a different sweep
entirely (every deme is always sampled here; phi only trades n_A against the B block's
total).

Policies (see models/__init__.py for why these are explicit rather than defaulted):
    axis location   select_by_overlap=True. The (1,0)/(0,1) pair is NOT reliably the
                    leading eigenvalue of the full A+B system once A carries a large sample
                    share -- at phi=0.5, K=9 it is rank 0-1, but at small phi another axis
                    outranks it -- so it is located by overlap rather than assumed to lead.
    target axis     the equilibrium 2D stepping-stone spatial pair (equilibrium_B_axes),
                    fixed across the whole M sweep
    theory route    flat, with nu from within_B_F_hat's sparse phase-type solve
    replicate loop  pc_recovery

prepare() runs the phase-type solve for this simulation's m, once per array task.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "analysis"))  # src/analysis/
from pc_recovery import analytical_cosine
from split_stepping_stone_theory import equilibrium_B_axes, within_B_F_hat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # src/
from threshold_and_overlap import (approx_nu_c, approx_overlap, exact_nu_c, exact_overlap)

SIM_PARAM = "invM"
X_KEY = "inv_M"


def x_value(sim_row, cfg):
    """1/M, with M = 4 * Ne_ancestral / K^2 * m, matching the existing driver."""
    return 1.0 / _M(sim_row)


def _M(sim_row):
    K = int(sim_row["K"])
    return 4.0 * float(sim_row["Ne_ancestral"]) / (K * K) * float(sim_row["m"])


def prepare(cfg, sim_row):
    """Cache the per-simulation constants; within_B_F_hat depends on n_per_deme_B too, so
    it is called per cell rather than here."""
    return {"K": int(sim_row["K"]),
            "m": float(sim_row["m"]),
            "t_split": float(sim_row["t_split"]),
            "Ne_ancestral": float(sim_row["Ne_ancestral"]),
            "M": _M(sim_row)}


def prepare_cached(cfg, cell_row):
    """
    The summariser's stand-in for prepare().

    prepare() reads simulation columns that live in the parameter table, not in a cell
    row, so calling it from the summariser raises. axes() records each of them in the
    cell's extra precisely so this can rebuild ctx from a CSV row -- theory_overlap
    re-solves within_B_F_hat per M and needs the full set, not just M.
    """
    return {"K": int(cell_row["K"]), "m": float(cell_row["m"]),
            "t_split": float(cell_row.get("t_split", float("nan"))),
            "Ne_ancestral": float(cell_row.get("Ne_ancestral", float("nan"))),
            "M": float(cell_row["M"])}


def spatial_axis(K, m, t_split, Ne_ancestral, n_per_deme_B, n_A):
    """
    The theoretical (1,0)/(0,1) spatial pair, lifted to full [A, B_00, ...] sample space:
    zero on A's block (A carries none of the spatial signal), the population-space pair
    spread per-deme on B's block. Ported from src/analysis/RecentSpatialOverlapVsM.

    The pair is the equilibrium 2D stepping-stone one; only nu stays on the transient
    phase-type solve. See split_stepping_stone_theory.equilibrium_B_axes.
    """
    nu, _ = within_B_F_hat(K, m, t_split, Ne_ancestral, n_per_deme_B)
    leading, _ = equilibrium_B_axes(K, n_per_deme_B)
    n_B = np.full(K * K, n_per_deme_B)
    lifted_B = np.repeat(leading / np.sqrt(n_B)[:, None], n_B, axis=0)
    V_theory = np.vstack([np.zeros((n_A, 2)), lifted_B])
    return V_theory, nu[:2]


def designs(cfg):
    """[(level, design_dict), ...] -- varying either n-per-deme-B or phi."""
    vary = cfg["sampling"]["vary"]
    fixed = cfg["sampling"]["fixed"]
    param, values = vary["param"], vary["values"]
    if param == "n_per_deme_B":
        phi = float(fixed["phi"])
        return [(int(v), {"n_per_deme_B": int(v), "phi": phi}) for v in values]
    if param == "phi":
        n_per = int(fixed["n_per_deme_B"])
        return [(float(v), {"n_per_deme_B": n_per, "phi": float(v)}) for v in values]
    raise ValueError(f"split_stepping_stone: cannot vary {param!r}; expected "
                     f"'n_per_deme_B' or 'phi'")


def axes(ctx, design, sim_row, cfg):
    """The spatial pair, located by overlap rather than by rank."""
    K = ctx["K"]
    K_demes = K * K
    n_per_deme_B = design["n_per_deme_B"]
    phi = design["phi"]

    # phi = B's total share of N; n_A is solved from phi and n_per_deme_B.
    N_B_total = n_per_deme_B * K_demes
    n_A = int(round(N_B_total * (1.0 - phi) / phi))

    avail_A = int(sim_row["n_per_pop_A"])
    avail_B = int(sim_row["n_per_deme_B"])
    if n_A < 1 or n_A > avail_A or n_per_deme_B > avail_B:
        return None          # outside what this simulation can subsample

    V_theory, nu = spatial_axis(K, ctx["m"], ctx["t_split"], ctx["Ne_ancestral"],
                                n_per_deme_B, n_A)
    n = np.concatenate([[n_A], np.full(K_demes, n_per_deme_B)])
    n_available = np.concatenate([[avail_A], np.full(K_demes, avail_B)])

    return dict(V_theory=V_theory, nu=nu, n=n, n_available=n_available,
                indices=[0, 1], select_by_overlap=True, eps=None,
                theory_route="flat", per_replicate=False,
                extra=dict(K=K, M=ctx["M"], m=ctx["m"], t_split=ctx["t_split"],
                           Ne_ancestral=ctx["Ne_ancestral"],
                           n_per_deme_B=n_per_deme_B, n_A=n_A, phi=phi,
                           N=int(n_A + N_B_total), nu_1=float(nu[0])))


def theory_overlap(ctx, design, sim_row, cfg, measured):
    """
    1/M -> predicted cos^2, re-solving the spatial eigenvalue at each M.

    nu comes from within_B_F_hat's sparse phase-type solve, which is what this demography
    needs -- the equilibrium stepping-stone closed form does not apply to a recent split.
    It is genuinely nonlinear in 1/M (the split time sets a ceiling on how much spatial
    structure can accumulate), so the earlier proportional rescaling of a single anchor
    was not just imprecise but the wrong shape. m is recovered from the swept M at fixed
    K and Ne_ancestral, and the solve repeated per point.

    The overlap uses the exact route on the per-locus weights h when available. phi covers
    the panmictic block A plus B's K^2 demes; eps is left at zero, as the flat route
    assumed.
    """
    K = int(ctx["K"])
    K_demes = K * K
    Ne_anc, t_split = float(ctx["Ne_ancestral"]), float(ctx["t_split"])
    n_per_deme_B = int(measured.get("n_per_deme_B", 0))
    n_A = int(measured.get("n_A", 0))
    gamma_raw = float(measured["gamma_raw"])
    if not n_per_deme_B or not n_A or not np.isfinite(Ne_anc) or not np.isfinite(t_split):
        return None

    n = np.concatenate([[n_A], np.full(K_demes, n_per_deme_B)])
    phi = n / n.sum()
    eps = np.zeros(len(n))

    h = measured.get("h")
    use_exact = h is not None and len(np.asarray(h))
    if use_exact:
        h = np.asarray(h, dtype=float)
        nu_c = exact_nu_c(phi, eps, h, gamma_raw)
    else:
        mu2, delta = float(measured["mu2"]), float(measured["delta"])
        if not np.isfinite(mu2) or not np.isfinite(delta):
            return None
        nu_c = approx_nu_c(phi, eps, gamma_raw, mu2, delta)

    def overlap(inv_M):
        # M = 4 * Ne_ancestral / K^2 * m, inverted for m at this point of the sweep.
        m = (1.0 / float(inv_M)) * (K * K) / (4.0 * Ne_anc)
        nu_vals, _ = within_B_F_hat(K, m, t_split, Ne_anc, n_per_deme_B)
        nu = float(np.mean(nu_vals[:2]))       # the degenerate spatial pair
        if use_exact:
            return exact_overlap(nu, phi, eps, h, gamma_raw, nu_c=nu_c)
        return approx_overlap(nu, phi, eps, gamma_raw, mu2, delta, nu_c=nu_c)
    return overlap
