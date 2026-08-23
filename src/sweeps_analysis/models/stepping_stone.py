"""
1D and 2D stepping stone: PC recovery against 1/M, under even or random deme sampling.

One module covers four configs because the demography and the theory are identical across
them; only the sampling differs. `dimension` and the sampling parameter come from the
config.

Policies (see models/__init__.py for why these are explicit rather than defaulted):
    axis location   fixed rank [0] in 1D; [0, 1] in 2D -- the (1,0)/(0,1) pair is exactly
                    degenerate, so no canonical basis exists inside it and only the
                    subspace is well defined. pc_recovery's principal angles compare
                    subspaces, which is what makes the 2D comparison well posed.
    theory route    flat -- analytical_cosine's BBP form, as the existing drivers use.
                    K is comparable to N here, so the Phi_2 criterion behind the exact
                    route does not apply (see threshold_and_overlap's note on
                    spectrum_psi); adopting it would be a change in the science, not a
                    port, so it is left alone.
    replicate loop  pc_recovery for even sampling; cell_replicates for random sampling,
                    where phi selects a fresh deme subset per replicate and the
                    theoretical axis therefore changes per replicate.

x axis. Overlap grows as migration weakens, and the transition fit needs an increasing
curve, so x is 1/M rather than M. That reciprocal is applied here, in the analysis, not in
the plotting layer.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "analysis"))  # src/analysis/
from pc_recovery import (analytical_cosine, effective_L, principal_cosines,
                         procrustes_distance, sample_eigenvectors, subsample)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "simulations"))  # src/simulations/
from stepping_stone_grid import (closed_form_eigenvector_1d, closed_form_eigenvector_2d,
                                 closed_form_spectrum, num_demes)

SIM_PARAM = "invM"
X_KEY = "inv_M"


def x_value(sim_row, cfg):
    """1/M: overlap increases as migration weakens, so the fit needs the reciprocal."""
    return 1.0 / float(sim_row["M"])


def k_lead(cfg):
    """1 in 1D, 2 (the degenerate pair) in 2D."""
    return 1 if int(cfg["fixed"]["dimension"]) == 1 else 2


def sample_demes(K_demes, phi, rng):
    """A random subset of deme indices, without replacement."""
    n_sampled = max(int(round(phi * K_demes)), 1)
    return np.sort(rng.choice(K_demes, n_sampled, replace=False))


def true_axis_restricted(K, dimension, n_per_deme, sampled_demes, K_demes):
    """
    The full-K leading closed-form eigenvector(s), restricted to the sampled demes'
    sample-space rows and renormalized. 1D returns one column; 2D returns the degenerate
    (1,0)/(0,1) pair, each restricted and renormalized independently.

    Deliberately NOT recomputed on the reduced deme set: the axis being tested is the one
    the full lattice defines, and asking how well a sparse sample recovers it is the
    question. Ported from src/analysis/SteppingStoneRandomSampling.
    """
    if dimension == 1:
        cols = [closed_form_eigenvector_1d(1, K, n_per_deme)]
    else:
        cols = [closed_form_eigenvector_2d(1, 0, K, n_per_deme),
                closed_form_eigenvector_2d(0, 1, K, n_per_deme)]

    row_mask = np.zeros(K_demes, dtype=bool)
    row_mask[sampled_demes] = True
    sample_mask = np.repeat(row_mask, n_per_deme)

    out = [u[sample_mask] / np.linalg.norm(u[sample_mask]) for u in cols]
    return np.column_stack(out)


def prepare(cfg, sim_row):
    """Closed-form spectrum for this simulation's M. Free."""
    K = int(sim_row["K"])
    dimension = int(cfg["fixed"]["dimension"])
    return {"K": K, "dimension": dimension, "M": float(sim_row["M"]),
            "K_demes": num_demes(K, dimension)}


def designs(cfg):
    """[(level, design_dict), ...] -- varying either n-per-deme or the deme fraction."""
    vary = cfg["sampling"]["vary"]
    fixed = cfg["sampling"]["fixed"]
    param, values = vary["param"], vary["values"]
    if param == "n_per_deme":
        frac = float(fixed["deme_fraction"])
        return [(int(v), {"n_per_deme": int(v), "deme_fraction": frac}) for v in values]
    if param == "deme_fraction":
        n_per = int(fixed["n_per_deme"])
        return [(float(v), {"n_per_deme": n_per, "deme_fraction": float(v)})
                for v in values]
    raise ValueError(f"stepping_stone: cannot vary {param!r}; expected 'n_per_deme' or "
                     f"'deme_fraction'")


def axes(ctx, design, sim_row, cfg):
    """
    The leading structure axis (1D) or degenerate pair (2D).

    A deme_fraction below 1 takes the per-replicate path: the axis depends on which demes
    are drawn, so it cannot be built once here.
    """
    K, dimension, M = ctx["K"], ctx["dimension"], ctx["M"]
    K_demes = ctx["K_demes"]
    n_per = design["n_per_deme"]
    frac = design["deme_fraction"]
    k = k_lead(cfg)

    _, nu_full, V_full = closed_form_spectrum(K, M, n_per, dimension)
    nu = nu_full[:k]

    if frac >= 1.0:
        n = np.full(K_demes, n_per, dtype=int)
        N = int(n.sum())
        return dict(V_theory=V_full[:, :k], nu=nu, n=n, n_available=None,
                    indices=list(range(k)), select_by_overlap=False, eps=None,
                    theory_route="flat", per_replicate=False,
                    extra=dict(K=K, K_demes=K_demes, dimension=dimension, M=M,
                               m=float(sim_row["m"]), Ne=float(sim_row["Ne"]),
                               n_per_deme=n_per, deme_fraction=frac,
                               n_demes_sampled=K_demes, nu_1=float(nu[0])))

    return dict(V_theory=None, nu=nu, n=None, n_available=None,
                indices=list(range(k)), select_by_overlap=False, eps=None,
                theory_route="flat", per_replicate=True,
                extra=dict(K=K, K_demes=K_demes, dimension=dimension, M=M,
                           m=float(sim_row["m"]), Ne=float(sim_row["Ne"]),
                           n_per_deme=n_per, deme_fraction=frac,
                           phi=frac, nu_1=float(nu[0])))


def cell_replicates(ctx, design, sim_row, cfg, Z, n_replicates, seed):
    """
    The random-deme replicate loop, returning what pc_recovery returns.

    Each replicate draws its own deme subset AND its own individuals within them, so both
    sources of randomness contribute to the spread. The rng call order (default_rng([seed,
    rep]) -> sample_demes -> subsample) is preserved exactly from
    src/analysis/SteppingStoneRandomSampling.measure_cell: it is what makes the two
    pipelines draw identical subsamples at a shared seed, which is how the port is verified.
    """
    K, dimension, M = ctx["K"], ctx["dimension"], ctx["M"]
    K_demes = ctx["K_demes"]
    n_per = design["n_per_deme"]
    phi = design["deme_fraction"]
    k = k_lead(cfg)
    L = cfg["fixed"].get("L")

    _, nu_full, _ = closed_form_spectrum(K, M, n_per, dimension)
    nu = nu_full[:k]

    cos_all, dists, L_effs, L_raws = [], [], [], []
    for rep in range(n_replicates):
        rng = np.random.default_rng([seed, rep])
        demes = sample_demes(K_demes, phi, rng)
        n = np.zeros(K_demes, dtype=int)
        n[demes] = n_per

        sub = subsample(Z, n, L, rng)
        V_theory = true_axis_restricted(K, dimension, n_per, demes, K_demes)
        by_magnitude = np.any(nu < 0)
        V_s = sample_eigenvectors(sub, list(range(k)), by_magnitude=by_magnitude)

        cos_all.append(principal_cosines(V_theory, V_s))
        dists.append(procrustes_distance(V_theory, V_s))
        L_effs.append(effective_L(sub))
        L_raws.append(sub.shape[1])

    cos_all = np.array(cos_all)
    mean_cos = cos_all.mean(axis=1)
    L_eff_med = float(np.median(L_effs))
    L_raw_med = float(np.median(L_raws))
    N_eff = int(n.sum())            # every replicate samples the same total N
    gamma = N_eff / L_eff_med
    _, analytical_mean = analytical_cosine(nu, gamma, N=N_eff)

    return dict(median_cos=float(np.median(mean_cos)),
                se_cos=float(np.std(mean_cos, ddof=1) / np.sqrt(len(mean_cos)))
                if len(mean_cos) > 1 else np.nan,
                analytical_cos=analytical_mean,
                procrustes_d=float(np.median(dists)),
                L_eff=L_eff_med, L_raw=L_raw_med, gamma=gamma,
                gamma_raw=N_eff / L_raw_med, mu2=np.nan, delta=np.nan,
                N=N_eff, K=K_demes, k=k,
                n_demes_sampled=int(len(demes)),
                num_replicates=n_replicates, seed=seed,
                cos_replicates=mean_cos)


def theory_overlap(ctx, design, sim_row, cfg, measured):
    """
    1/M -> predicted cos^2, via the flat BBP form.

    nu_k = N / (M * K_demes * mu_k) is linear in 1/M at fixed design, so the closure
    rescales the recorded nu by the ratio of 1/M values rather than refitting anything.
    """
    nu_ref = float(measured.get("nu_1", np.nan))
    inv_M_ref = 1.0 / float(ctx["M"])
    gamma = float(measured["gamma"])
    N = int(measured["N"])
    if not np.isfinite(nu_ref) or nu_ref <= 0:
        return None

    def overlap(inv_M):
        nu = nu_ref * (float(inv_M) / inv_M_ref)
        _, mean = analytical_cosine(np.atleast_1d(nu), gamma, N=N)
        return float(mean) ** 2
    return overlap
