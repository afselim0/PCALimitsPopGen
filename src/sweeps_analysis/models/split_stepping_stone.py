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
# No threshold_and_overlap import: this model publishes no theory (see the note below axes).

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


# --- no theory for this model ---------------------------------------------
#
# theory_overlap, nu_of_x and theory_context are deliberately NOT defined here, which is
# what marks this model theory-free (see models/__init__.py). The summariser tests for the
# attribute and falls back to NaN y_theory / theory_crossing_pred, and writes no theory.csv
# rows for these sweeps.
#
# The reason is the demography, not the machinery: this is a recent split, so the
# equilibrium stepping-stone closed form does not apply, and the sparse phase-type solve
# that would replace it (split_stepping_stone_theory.within_B_F_hat) is not currently
# trusted enough to publish as a prediction. axes() below still uses within_B_F_hat to
# build V_theory -- that is the MEASUREMENT axis the empirical cosine is taken against,
# which is a different thing from predicting what that cosine will be.
