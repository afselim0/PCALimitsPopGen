"""
1D and 2D stepping stone: PC recovery against 1/M, under even or random deme sampling.

One module covers six configs because the demography and the theory are identical across
them; only the sampling differs. `dimension` and the sampling parameter come from the
config: n-per-deme at full coverage, a random fraction of demes, or `deme_stride`, which
keeps every p-th deme at a FIXED total N (configs 11 and 12).

Uniform dilution (`deme_stride`) is deterministic given (K, p), so unlike the random
fraction it needs no per-replicate redraw: the axis is a property of the cell. Total N is
held fixed across p, so the samples freed by dropping demes are redistributed over the
demes that remain -- the sweep isolates spatial coverage from sample size.

Policies (see models/__init__.py for why these are explicit rather than defaulted):
    axis location   fixed rank [0] in 1D; [0, 1] in 2D -- the (1,0)/(0,1) pair is exactly
                    degenerate, so no canonical basis exists inside it and only the
                    subspace is well defined. pc_recovery's principal angles compare
                    subspaces, which is what makes the 2D comparison well posed.
    theory route    the exact heteroskedastic route on the per-locus weights h, on both
                    paths. K is comparable to N here, so the Phi_2 criterion behind it is
                    not strictly the right one (see threshold_and_overlap's note on
                    spectrum_psi) -- that question is open and separate from which nu is
                    fed in.
    nu              closed form N/(M K_demes mu_k) only when every deme is sampled;
                    under a random fraction the median over random deme subsets, and
                    under uniform dilution the exact numeric value for that stride --
                    see stepping_stone_grid.{subsampled,strided}_structure_eigenvalues.
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
from pc_recovery import (analytical_cosine, effective_L, locus_moments, locus_weights,
                         principal_cosines, procrustes_distance, sample_eigenvectors,
                         subsample)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # src/
from threshold_and_overlap import (approx_nu_c, approx_overlap, exact_nu_c, exact_overlap)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "simulations"))  # src/simulations/
from stepping_stone_grid import (closed_form_eigenvector_1d, closed_form_eigenvector_2d,
                                 closed_form_spectrum, num_demes,
                                 replicate_structure_eigenvalues,
                                 strided_demes, strided_structure_eigenvalues,
                                 subsampled_structure_eigenvalues)

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


def stride_n_per_deme(K, p, dimension, N_total):
    """n-per-deme placing N_total samples evenly over the demes a stride p retains.

    Floor, not round: the cell must never ask for more samples than were simulated per
    deme, and a fixed N is the point of the sweep only up to this integer division --
    the realised N is recorded per cell as `N`.
    """
    n_demes = len(strided_demes(K, p, dimension))
    return max(int(N_total // n_demes), 1)


def designs(cfg):
    """[(level, design_dict), ...] -- varying n-per-deme, the deme fraction, or the stride.

    Under `deme_stride` n-per-deme is not a free parameter: it is whatever spreads the
    config's fixed N_total over the demes that stride retains, so it is derived here
    rather than read from the config. K comes from `fixed.K` because designs() is built
    once for the whole sweep, before any simulation row is in hand.
    """
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
    if param == "deme_stride":
        K = int(cfg["fixed"]["K"])
        dimension = int(cfg["fixed"]["dimension"])
        N_total = int(fixed["N_total"])
        return [(int(v), {"deme_stride": int(v), "N_total": N_total,
                          "n_per_deme": stride_n_per_deme(K, int(v), dimension, N_total)})
                for v in values]
    raise ValueError(f"stepping_stone: cannot vary {param!r}; expected 'n_per_deme', "
                     f"'deme_fraction' or 'deme_stride'")


def _stride_axes(ctx, design, sim_row, cfg):
    """
    The spec for one uniform-dilution cell: every p-th deme, n_per_deme derived from N_total.

    The axis is the full-lattice closed-form eigenvector RESTRICTED to the sampled demes
    (true_axis_restricted), the same convention the random-fraction path uses: the axis
    being tested is the one the full lattice defines, and the question is how well a
    diluted sample recovers it. nu is the exact numeric value for this stride, not the
    even-sampling closed form, which holds only at p = 1.

    2D caveat. The (1,0)/(0,1) pair is degenerate only when the retained rows sit
    symmetrically about the lattice centre. At K = 9 a stride of 3 keeps {0, 3, 6}, which
    does not, and the pair splits (nu ~ 49 against 43 at M = 1); the restricted closed-form
    pair then reaches principal cosines of about (0.99, 0.83) with the numeric leading
    pair rather than (1, 1). The comparison stays a subspace comparison regardless, and
    the reduction of nu to a single number is the same mean used everywhere else.
    """
    K, dimension, M = ctx["K"], ctx["dimension"], ctx["M"]
    K_demes = ctx["K_demes"]
    p = int(design["deme_stride"])
    n_per = int(design["n_per_deme"])
    k = k_lead(cfg)

    demes = strided_demes(K, p, dimension)
    n = np.zeros(K_demes, dtype=int)
    n[demes] = n_per

    nu = strided_structure_eigenvalues(K, M, n_per, p, dimension, Ne=float(sim_row["Ne"]))
    V_theory = true_axis_restricted(K, dimension, n_per, demes, K_demes)
    # principal_cosines is the SVD of V_theory^T V_sample and needs an ORTHONORMAL basis;
    # restricting the closed forms to a deme subset destroys the orthogonality they have
    # on the full lattice. At K=9, p=3 the two restricted columns correlate at 0.10, which
    # would bias the reported cosine, so re-orthonormalize. The span is unchanged, and the
    # comparison was always a subspace comparison. (1D has one column, so QR only fixes
    # its sign convention, which principal_cosines takes the absolute value of anyway.)
    if V_theory.shape[1] > 1:
        V_theory = np.linalg.qr(V_theory)[0]

    return dict(V_theory=V_theory, nu=nu, n=n, n_available=None,
                indices=list(range(k)), select_by_overlap=False, eps=None,
                theory_route="exact_h", per_replicate=False,
                extra=dict(K=K, K_demes=K_demes, dimension=dimension, M=M,
                           m=float(sim_row["m"]), Ne=float(sim_row["Ne"]),
                           n_per_deme=n_per, deme_stride=p,
                           N_total=int(design["N_total"]),
                           deme_fraction=len(demes) / K_demes,
                           n_demes_sampled=int(len(demes)), nu_1=float(nu[0])))


def axes(ctx, design, sim_row, cfg):
    """
    The leading structure axis (1D) or degenerate pair (2D).

    A deme_fraction below 1 takes the per-replicate path: the axis depends on which demes
    are drawn, so it cannot be built once here. A deme_stride does not -- the subset is
    fixed by (K, p) -- so it builds its axis once, like even sampling.
    """
    K, dimension, M = ctx["K"], ctx["dimension"], ctx["M"]
    K_demes = ctx["K_demes"]
    n_per = design["n_per_deme"]
    k = k_lead(cfg)

    if "deme_stride" in design:
        return _stride_axes(ctx, design, sim_row, cfg)

    frac = design["deme_fraction"]

    _, nu_full, V_full = closed_form_spectrum(K, M, n_per, dimension)
    nu = nu_full[:k]

    if frac >= 1.0:
        n = np.full(K_demes, n_per, dtype=int)
        N = int(n.sum())
        return dict(V_theory=V_full[:, :k], nu=nu, n=n, n_available=None,
                    indices=list(range(k)), select_by_overlap=False, eps=None,
                    theory_route="exact_h", per_replicate=False,
                    extra=dict(K=K, K_demes=K_demes, dimension=dimension, M=M,
                               m=float(sim_row["m"]), Ne=float(sim_row["Ne"]),
                               n_per_deme=n_per, deme_fraction=frac,
                               n_demes_sampled=K_demes, nu_1=float(nu[0])))

    # nu here is the even-sampling closed form and is only a placeholder: the real value
    # depends on which demes are drawn, so cell_replicates recomputes it as the median
    # over random subsets and overrides both nu and nu_1.
    nu_sub = subsampled_structure_eigenvalues(K, M, n_per, frac, dimension)
    return dict(V_theory=None, nu=nu_sub, n=None, n_available=None,
                indices=list(range(k)), select_by_overlap=False, eps=None,
                theory_route="exact_h", per_replicate=True,
                extra=dict(K=K, K_demes=K_demes, dimension=dimension, M=M,
                           m=float(sim_row["m"]), Ne=float(sim_row["Ne"]),
                           n_per_deme=n_per, deme_fraction=frac,
                           phi=frac, nu_1=float(nu_sub[0])))


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
    Ne = float(sim_row["Ne"])

    # The closed form presumes every deme is sampled; at phi < 1 nu is the median over
    # random deme subsets instead (see subsampled_structure_eigenvalues).
    if phi >= 1.0:
        _, nu_full, _ = closed_form_spectrum(K, M, n_per, dimension)
        nu = nu_full[:k]
    else:
        nu = subsampled_structure_eigenvalues(K, M, n_per, phi, dimension)

    cos_all, cos_all_std, dists, L_effs, L_raws = [], [], [], [], []
    mu2s, deltas, nu_reps = [], [], []
    lam_all, lam_bulk_all = [], []
    h_rep0 = None
    for rep in range(n_replicates):
        rng = np.random.default_rng([seed, rep])
        demes = sample_demes(K_demes, phi, rng)
        n = np.zeros(K_demes, dtype=int)
        n[demes] = n_per

        # This replicate's OWN structure eigenvalue, for the deme set it actually drew.
        # nu above is the median over 16 unrelated subsets, which describes the sampling
        # scheme rather than this draw; at phi < 1 different subsets of the same size give
        # genuinely different nu, so the per-replicate value is what pairs with the
        # per-replicate measurement. The generic F_hat route is used rather than the
        # even-sampling closed form, which does not apply once demes are missing.
        nu_reps.append(replicate_structure_eigenvalues(K, M, n_per, demes, dimension,
                                                       k, Ne=Ne))

        # L_polymorphic must be honoured here as it is on the pc_recovery path: without
        # it L is a pre-filter draw count and the realised width drifts with N, so the
        # replicates being medianed would not share a gamma (measured 5268 to 10000
        # across this config's cells at a nominal L = 10000).
        sub = subsample(Z, n, L, rng,
                        L_polymorphic=cfg["fixed"].get("L_polymorphic", False))
        V_theory = true_axis_restricted(K, dimension, n_per, demes, K_demes)
        by_magnitude = np.any(nu < 0)
        # Only the sampled demes carry samples, so the number of subpopulations here is
        # len(demes): its K-1 structure eigenvalues sit at ranks 0..K-2 and rank K-1 is the
        # first bulk level.
        V_s, lam, lam_bulk = sample_eigenvectors(
            sub, list(range(k)), by_magnitude=by_magnitude,
            return_eigenvalues=True, bulk_index=max(len(demes) - 1, 1))

        cos_all.append(principal_cosines(V_theory, V_s))
        dists.append(procrustes_distance(V_theory, V_s))
        lam_all.append(float(np.mean(lam)))
        lam_bulk_all.append(lam_bulk)

        # Same subsample, same axis, every locus weighted equally (see pc_recovery).
        V_s_std = sample_eigenvectors(sub, list(range(k)), by_magnitude=by_magnitude,
                                      standardize=True)
        cos_all_std.append(principal_cosines(V_theory, V_s_std))

        L_effs.append(effective_L(sub))
        L_raws.append(sub.shape[1])
        # The locus heteroskedasticity the exact threshold/overlap route needs. Recorded
        # here as pc_recovery does on the even-sampling path; without it this path could
        # only ever reach the flat fallback.
        m2, _, dl = locus_moments(sub)
        mu2s.append(m2)
        deltas.append(dl)
        if rep == 0:
            h_rep0 = locus_weights(sub)

    cos_all = np.array(cos_all)
    mean_cos = cos_all.mean(axis=1)
    cos_all_std = np.array(cos_all_std)
    mean_cos_std = cos_all_std.mean(axis=1)
    # One nu per replicate: the degenerate pair is averaged the way the measurement
    # averages its two axes, so nu and cos describe the same subspace.
    nu_per_replicate = np.array([float(np.mean(v)) for v in nu_reps])
    lam_replicates = np.array(lam_all, dtype=float)
    lam_bulk_replicates = np.array(lam_bulk_all, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_hat_replicates = lam_replicates / lam_bulk_replicates - 1.0
    L_eff_med = float(np.median(L_effs))
    L_raw_med = float(np.median(L_raws))
    N_eff = int(n.sum())            # every replicate samples the same total N
    gamma = N_eff / L_eff_med
    gamma_raw = N_eff / L_raw_med
    mu2_med = float(np.median(mu2s))
    delta_med = float(np.median(deltas))

    # phi_a is flat over the sampled demes: only they carry samples, and equally many.
    # eps vanishes -- homogeneous migration and equal deme sizes make every within-deme
    # coalescent time equal.
    n_sampled = int(len(demes))
    phi_a = np.full(n_sampled, 1.0 / n_sampled)
    _, analytical_mean = analytical_cosine(
        nu, gamma, N=N_eff, phi=phi_a, eps=np.zeros(n_sampled),
        gamma_raw=gamma_raw, mu2=mu2_med, delta=delta_med, h=h_rep0)

    return dict(nu=float(np.mean(nu)),
                median_cos=float(np.median(mean_cos)),
                se_cos=float(np.std(mean_cos, ddof=1) / np.sqrt(len(mean_cos)))
                if len(mean_cos) > 1 else np.nan,
                analytical_cos=analytical_mean,
                procrustes_d=float(np.median(dists)),
                L_eff=L_eff_med, L_raw=L_raw_med, gamma=gamma,
                gamma_raw=gamma_raw, mu2=mu2_med, delta=delta_med, h=h_rep0,
                N=N_eff, K=K_demes, k=k,
                n_demes_sampled=int(len(demes)),
                num_replicates=n_replicates, seed=seed,
                cos_replicates=mean_cos,
                median_cos_std=float(np.median(mean_cos_std)),
                se_cos_std=float(np.std(mean_cos_std, ddof=1) / np.sqrt(len(mean_cos_std)))
                if len(mean_cos_std) > 1 else np.nan,
                cos_replicates_std=mean_cos_std,
                # Per-replicate theory: each replicate drew its own deme subset, so nu is
                # a property of the replicate here, not of the cell.
                nu_replicates=nu_per_replicate,
                lam_replicates=lam_replicates,
                lam_bulk_replicates=lam_bulk_replicates,
                nu_hat_replicates=nu_hat_replicates,
                median_lam=float(np.median(lam_replicates)),
                median_lam_bulk=float(np.median(lam_bulk_replicates)),
                median_nu_hat=float(np.median(nu_hat_replicates)))


NU_FORM = "closed_form_or_numeric"


def theory_context(ctx, design, sim_row, cfg, measured):
    """
    The RMT inputs for this cell, with no nu in them.

    Only the sampled demes carry samples, and each carries the same number, so phi is flat
    over however many demes the design actually covers. eps is zero throughout: homogeneous
    migration and equal deme sizes make every within-deme coalescent time equal (see
    stepping_stone_grid.full_spectrum).
    """
    n_demes = int(measured.get("n_demes_sampled", ctx["K_demes"]))
    try:
        gamma_raw = float(measured["gamma_raw"])
        if not np.isfinite(gamma_raw) or gamma_raw <= 0:
            return None
        return dict(phi=np.full(n_demes, 1.0 / n_demes), eps=np.zeros(n_demes),
                    h=measured.get("h"), gamma_raw=gamma_raw,
                    mu2=float(measured.get("mu2", np.nan)),
                    delta=float(measured.get("delta", np.nan)),
                    N=int(measured["N"]), L_eff=float(measured["L_eff"]))
    except (KeyError, TypeError, ValueError):
        return None


def nu_of_x(ctx, design, sim_row, cfg, measured):
    """
    1/M -> the structure eigenvalue, by whichever of the three routes the design calls for.

    Under EVEN sampling nu_k = N / (M * K_demes * mu_k) with the exact
    mu_k = 4 sin^2(pi k / 2K) (1D) or mu_k + mu_l (2D), recomputed at every 1/M by
    closed_form_spectrum rather than rescaling a single recorded anchor. The two agree
    here -- nu really is proportional to 1/M with no pedestal -- but recomputing keeps the
    exact mu_k in the loop instead of inheriting whatever the anchor cell held.

    Under uniform dilution nu comes from strided_structure_eigenvalues, the exact numeric
    value for the deme set that stride names -- no median, because the set is not random.

    That closed form presumes every deme is sampled and is WRONG at deme_fraction < 1,
    where it overstates nu badly (1D, K=25, phi=0.2: 101 against an actual ~18). A fraction
    does not name a scheme, so nu is taken from subsampled_structure_eigenvalues, which
    defines it as the median over 16 random deme subsets.
    """
    K, dimension = ctx["K"], ctx["dimension"]
    n_per = int(measured.get("n_per_deme", design["n_per_deme"]
                             if isinstance(design, dict) else 0))
    if not n_per:
        return None
    k = k_lead(cfg)
    n_demes = int(measured.get("n_demes_sampled", ctx["K_demes"]))
    frac = float(measured.get("deme_fraction", n_demes / ctx["K_demes"]))
    # A stride names an exact deme set, so nu is the numeric value for that set rather
    # than the median over random subsets of the same size.
    stride = measured.get("deme_stride")
    stride = int(stride) if stride is not None and np.isfinite(stride) else None
    Ne = float(measured.get("Ne", 100.0))

    def nu_at(inv_M):
        M = 1.0 / float(inv_M)
        if stride is not None:
            nu_full = strided_structure_eigenvalues(K, M, n_per, stride, dimension, Ne=Ne)
        elif frac >= 1.0:
            _, nu_full, _ = closed_form_spectrum(K, M, n_per, dimension)
        else:
            nu_full = subsampled_structure_eigenvalues(K, M, n_per, frac, dimension)
        return float(np.mean(nu_full[:k]))   # the degenerate pair shares one value in 2D
    return nu_at


def theory_overlap(ctx, design, sim_row, cfg, measured):
    """
    1/M -> predicted cos^2, from the structure eigenvalue at each M.

    Under EVEN sampling nu_k = N / (M * K_demes * mu_k) with the exact
    mu_k = 4 sin^2(pi k / 2K) (1D) or mu_k + mu_l (2D), recomputed at every 1/M by
    closed_form_spectrum rather than rescaling a single recorded anchor. The two agree
    here -- nu really is proportional to 1/M with no pedestal -- but recomputing keeps
    the exact mu_k in the loop instead of inheriting whatever the anchor cell held.

    Under uniform dilution nu comes from strided_structure_eigenvalues, the exact numeric
    value for the deme set that stride names -- no median, because the set is not random.

    That closed form presumes every deme is sampled and is WRONG at deme_fraction < 1,
    where it overstates nu badly (1D, K=25, phi=0.2: 101 against an actual ~18). A
    fraction does not name a scheme, so nu is taken from
    subsampled_structure_eigenvalues, which defines it as the median over 16 random deme
    subsets -- the same redraw the measurement performs per replicate.

    The overlap goes through the exact heteroskedastic route on the per-locus weights h
    when the sidecar supplies them, falling back to the two-moment route otherwise. eps
    is zero throughout: homogeneous migration and equal deme sizes make every
    within-deme coalescent time equal (see stepping_stone_grid.full_spectrum).

    Caveat carried from the module docstring: K is comparable to N here, so the Phi_2
    criterion behind this route is not the strictly correct one -- spectrum_psi is (see
    threshold_and_overlap). This uses the exact route as asked; the criterion question is
    separate.
    """
    tc = theory_context(ctx, design, sim_row, cfg, measured)
    nu_fn = nu_of_x(ctx, design, sim_row, cfg, measured)
    if tc is None or nu_fn is None:
        return None
    phi, eps, gamma_raw = tc["phi"], tc["eps"], tc["gamma_raw"]

    h = tc.get("h")
    use_exact = h is not None and len(np.asarray(h))
    if use_exact:
        h = np.asarray(h, dtype=float)
        nu_c = exact_nu_c(phi, eps, h, gamma_raw)
    else:
        mu2, delta = tc["mu2"], tc["delta"]
        if not np.isfinite(mu2) or not np.isfinite(delta):
            # The random-sampling path records mu2/delta as NaN, so without h there is
            # nothing to build an exact prediction from.
            return None
        nu_c = approx_nu_c(phi, eps, gamma_raw, mu2, delta)

    def overlap(inv_M):
        nu = nu_fn(inv_M)
        if use_exact:
            return exact_overlap(nu, phi, eps, h, gamma_raw, nu_c=nu_c)
        return approx_overlap(nu, phi, eps, gamma_raw, mu2, delta, nu_c=nu_c)
    return overlap
