#%%
"""
How well PCA recovers the theoretical structure axes of a simulated genotype matrix.

One measurement, applied uniformly across demographic scenarios: subsample a genotype
matrix, compute its PCs, and compare them against theoretical eigenvectors by principal
angles. The comparison is between *subspaces*, so a degenerate multiplet -- routine in
symmetric spatial demographies -- gets a single verdict rather than an arbitrary basis
choice inside its eigenspace.

    load_simulations         .npy sweep in a directory -> parameters + mmap'd arrays
    subsample                draw n_a samples per subpopulation, L SNPs, drop monomorphic
    theoretical_eigenvectors sample-space structure axes from a coalescent time matrix
    pc_recovery              the measurement: median cosine similarity over replicates

Metric. For k axes the SVD of V_theory^T @ V_sample gives the cosines of the principal
angles; their mean is the primary number, in [0, 1] for any k. At k = 1 Procrustes
alignment is a sign flip, so this is exactly |cos| -- the k = 1 and k > 1 cases are on the
same scale and sweeps are comparable across K. The Procrustes distance is reported
alongside for the cases where the individual axes, not just the subspace, are of interest.
"""
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from utils import compute_sample_covariance, compute_F_hat
from threshold_and_overlap import (eta_moments, approx_nu_c, approx_overlap,
                                   exact_nu_c, exact_overlap)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMULATION_ROOT = os.path.join(_PROJECT_ROOT, "results", "simulations")


def _parse_simID(simID):
    """
    Parameters encoded in a filename stem, as {name: float}.

    Covers every convention in results/simulations: Fst<v>, invM<v> (the stepping-stone
    kinds, whose swept parameter is 1/M), and t<v>_Fst<v>. A stem
    that matches nothing yields an empty dict rather than an error, so an unrecognized
    naming scheme degrades to "no parsed parameters" instead of failing the load.
    """
    return {name: float(value)
            for name, value in re.findall(r"([A-Za-z_]+)([\d.eE+-]+)", str(simID))}


def load_simulations(simdir, mmap=True):
    """
    Genotype matrices in a simulation directory, with their parameters.

    Prefers a *simulation_parameters.csv, whose simID column is the filename stem; falls
    back to globbing *.npy and parsing the stems when no CSV is present (single_split has
    none, and SS_2D/K11 is an incomplete run whose CSV was never written).

    Arrays are memory-mapped by default: a full sweep is tens of gigabytes, and subsample()
    touches only a slice of each.

    Returns:
        (params, arrays): a DataFrame with one row per simulation (always including simID,
        and any parameters recovered from the CSV or the filename), and the matching list
        of (N, L) arrays. Both are sorted by the first parsed parameter.
    """
    if not os.path.isdir(simdir):
        raise FileNotFoundError(f"no such simulation directory: {simdir}")

    csvs = glob.glob(os.path.join(simdir, "*simulation_parameters.csv"))
    paths = sorted(glob.glob(os.path.join(simdir, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"no .npy simulations in {simdir}")

    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    params = pd.DataFrame({"simID": stems, "path": paths})

    if csvs:
        meta = pd.read_csv(csvs[0])
        if "simID" in meta.columns:
            meta["simID"] = meta["simID"].astype(str)
            params = params.merge(meta, on="simID", how="left")

    # Fill in anything the CSV did not provide (or that has no CSV at all) from the stem.
    parsed = pd.DataFrame([_parse_simID(s) for s in stems])
    for col in parsed.columns:
        if col not in params.columns:
            params[col] = parsed[col].values
        else:
            params[col] = params[col].fillna(parsed[col])

    sort_cols = [c for c in params.columns if c not in ("simID", "path")
                 and pd.api.types.is_numeric_dtype(params[c]) and params[c].notna().any()]
    if sort_cols:
        params = params.sort_values(sort_cols[0]).reset_index(drop=True)

    arrays = [np.load(p, mmap_mode="r" if mmap else None) for p in params["path"]]
    return params, arrays


def subsample(Z, n, L=None, rng=None, n_available=None, L_polymorphic=False):
    """
    Draw n_a samples from each subpopulation and (optionally) L SNP columns.

    Rows of Z are block-ordered by subpopulation. By default each block is assumed the
    same size, Z.shape[0] // K, which is how every equal-sampling simulation in
    src/simulations writes them. Pass n_available (length K, block sizes summing to
    Z.shape[0]) for demographies with unequal per-subpopulation sample sizes, e.g.
    split_stepping_stone's panmictic A block vs. its many smaller per-deme B blocks.

    Monomorphic loci are dropped *after* the rows are subset, not before: a locus that
    segregates in the full sample may be fixed in the subsample, and leaving those columns
    in would inflate the raw SNP count with columns that carry no signal.

    That ordering means L is a pre-filter draw count by default, so the surviving width
    falls short of L by however many drawn loci turned out monomorphic -- and by more at
    small n, which is exactly when a sweep wants L held fixed. L_polymorphic=True instead
    filters first and then draws exactly L of the polymorphic columns, so every cell comes
    back the same width; it raises if fewer than L are available.

    Args:
        Z: (N_full, L_full) genotype matrix.
        n: per-subpopulation sample sizes, length K.
        L: number of SNP columns. Drawn before the polymorphic filter by default, so the
            result is narrower than L; with L_polymorphic=True it is the exact output
            width. None keeps all.
        rng: numpy Generator.
        n_available: per-subpopulation block sizes actually present in Z, length K. None
            assumes every block has Z.shape[0] // K rows (the equal-sampling default).
        L_polymorphic: draw L from the polymorphic columns rather than from all columns,
            making L the exact number of SNPs returned.

    Returns:
        (sum(n), L_polymorphic) float32 array.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = np.asarray(n, dtype=int)
    K = len(n)
    if n_available is None:
        if Z.shape[0] % K:
            raise ValueError(f"{Z.shape[0]} rows do not divide evenly into K={K} subpopulations")
        n_available = np.full(K, Z.shape[0] // K, dtype=int)
    else:
        n_available = np.asarray(n_available, dtype=int)
        if len(n_available) != K:
            raise ValueError(f"n_available has {len(n_available)} entries but n has {K}")
        if n_available.sum() != Z.shape[0]:
            raise ValueError(f"n_available sums to {n_available.sum()} but Z has "
                             f"{Z.shape[0]} rows")
    if np.any(n > n_available):
        raise ValueError(f"requested {n.max()} samples but only {n_available.min()} are "
                         f"available in the smallest subpopulation")

    offsets = np.concatenate([[0], np.cumsum(n_available)[:-1]])
    idx = np.sort(np.concatenate([off + rng.choice(avail, n_a, replace=False)
                                  for off, avail, n_a in zip(offsets, n_available, n)]))
    if L_polymorphic and L is not None:
        # Every column has to be read before the polymorphic ones can be counted, so this
        # path costs a full-width row slice; that is the price of an exact output width.
        sub = np.asarray(Z[idx, :], dtype=np.float32)
        freq = sub.mean(axis=0)
        poly = np.flatnonzero((freq > 0) & (freq < 1))
        if len(poly) < L:
            raise ValueError(f"requested L={L} polymorphic SNPs but only {len(poly)} are "
                             f"polymorphic in this subsample of {int(n.sum())} samples")
        return sub[:, np.sort(rng.choice(poly, L, replace=False))]

    cols = np.sort(rng.choice(Z.shape[1], L, replace=False)) \
        if L is not None and L < Z.shape[1] else slice(None)
    # A single np.ix_ index avoids materializing the full-width row slice that
    # Z[idx][:, cols] would read off the memory-mapped file; sorted indices keep the
    # mmap reads close to sequential instead of scattering across the file.
    sub = np.asarray(Z[np.ix_(idx, cols)] if isinstance(cols, np.ndarray) else Z[idx][:, cols],
                      dtype=np.float32)
    freq = sub.mean(axis=0)
    return sub[:, (freq > 0) & (freq < 1)]


def effective_L(Z):
    """
    Variance-effective SNP count, the participation ratio (sum v)^2 / sum v^2, v = p(1-p).

    The GRM is built from unnormalized genotypes, so column l enters with weight v_l and
    the aspect ratio that governs the bulk is N / L_eff rather than N / L.
    """
    p = np.asarray(Z, dtype=np.float64).mean(axis=0)
    v = p * (1.0 - p)
    return float(v.sum() ** 2 / (v ** 2).sum())


def locus_moments(Z):
    """(mu2, mu3, delta) of the per-locus weights v = p(1-p), over polymorphic columns."""
    return eta_moments(locus_weights(Z))


def locus_weights(Z):
    """The per-locus weights h = p(1-p) over polymorphic columns.

    The exact route of src/threshold_and_overlap needs the whole vector, not the two
    moments locus_moments reduces it to, so this is kept alongside rather than derived
    from it.
    """
    p = np.asarray(Z, dtype=np.float64).mean(axis=0)
    v = p * (1.0 - p)
    return v[v > 0]


def theoretical_eigenvectors(T, n, indices=None, rtol=1e-8):
    """
    Sample-space structure eigenvectors implied by a population-level coalescent time
    matrix T (K x K, diag = tau) and per-population sample sizes n.

    F_hat (utils.compute_F_hat) is the K x K matrix whose non-trivial eigenvalues are the
    structure eigenvalues. Its eigenvectors live in population space; each is lifted to
    sample space by giving every sample of population a the value w_a / sqrt(n_a), which
    keeps the lifted vectors orthonormal.

    The trivial eigenvector psi = sqrt(phi) is dropped: it is the direction removed by
    mean-centering and carries no structure.

    Degeneracy. Symmetric spatial demographies routinely give exactly degenerate structure
    eigenvalues (a 2D stepping-stone grid has degenerate pairs), and inside a degenerate
    eigenspace no canonical basis exists -- only the subspace is well defined. Selecting
    `indices` that cut through such a multiplet therefore asks for something the theory does
    not supply, and warns. Principal angles compare subspaces, so passing the whole
    multiplet gives a well-posed answer.

    Args:
        T: (K, K) mean pairwise coalescent times between populations.
        n: per-population sample sizes, length K.
        indices: which structure axes to return, 0 = largest eigenvalue. None returns all
            K-1, ordered by descending eigenvalue.
        rtol: relative tolerance for the degeneracy check.

    Returns:
        (V, nu): the (N, k) orthonormal sample-space axes and their structure eigenvalues.
    """
    n = np.asarray(n, dtype=int)
    K = len(n)
    N = int(n.sum())
    phi = n / N
    psi = np.sqrt(phi)

    w, V = np.linalg.eigh(compute_F_hat(T, n))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    # Drop the trivial psi direction rather than a fixed position in the ordering: with
    # heterogeneous tau its eigenvalue is not always the smallest.
    trivial = int(np.argmax(np.abs(psi @ V)))
    keep = [j for j in range(K) if j != trivial]
    w, V = w[keep], V[:, keep]

    if indices is not None:
        indices = np.atleast_1d(indices)
        for i in indices:
            same = np.abs(w - w[i]) <= rtol * max(abs(w[i]), 1.0)
            if same.sum() > 1 and not np.all(np.isin(np.flatnonzero(same), indices)):
                warnings.warn(
                    f"index {i} lies in a {same.sum()}-fold degenerate eigenspace "
                    f"(nu={w[i]:.6g}) but not every member was requested; no canonical "
                    f"basis exists inside a degenerate multiplet, so pass all of "
                    f"{np.flatnonzero(same).tolist()} to get a well-posed subspace.")
        w, V = w[indices], V[:, indices]

    # Lift population space -> sample space; the 1/sqrt(n_a) keeps the columns orthonormal.
    lift = np.repeat(V / np.sqrt(n)[:, None], n, axis=0)
    return lift, w


def principal_cosines(V_theory, V_sample):
    """
    Cosines of the principal angles between two orthonormal bases, descending.

    The SVD of V_theory^T @ V_sample; singular values are the cosines. At k = 1 this is
    |cos| between two vectors, so the metric is continuous from one axis to many.
    """
    s = np.linalg.svd(V_theory.T @ V_sample, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def procrustes_distance(V_theory, V_sample):
    """
    ||V_theory - V_sample @ R||_F / sqrt(k) after the optimal rotation R.

    Normalized by sqrt(k) so the value does not grow simply because more axes are compared.
    Unlike the principal cosines this is sensitive to the individual axes rather than only
    the subspace they span.
    """
    R = orthogonal_procrustes(V_sample, V_theory)[0]
    return float(np.linalg.norm(V_theory - V_sample @ R) / np.sqrt(V_theory.shape[1]))


def sample_eigenvectors(Z, indices, by_magnitude=False):
    """
    Eigenvectors of the sample GRM at `indices`, 0 = largest eigenvalue.

    by_magnitude ranks by |lambda| instead. Structure eigenvalues can be negative when the
    within-population coalescent times are heterogeneous (eps_a far from 0), and a negative
    structure eigenvalue sits *below* the floor in signed order -- so ranking by descending
    lambda would return a floor eigenvector in its place. Rank by magnitude whenever any
    theoretical nu is negative; theoretical_eigenvectors reports the nu so the caller can
    tell, and pc_recovery switches automatically.
    """
    G = compute_sample_covariance(Z)
    w, V = np.linalg.eigh(G)
    order = np.argsort(np.abs(w))[::-1] if by_magnitude else np.argsort(w)[::-1]
    if by_magnitude:
        # Drop the mean-centering zero, which has the smallest magnitude of all.
        order = order[np.abs(w[order]) > 1e-8 * max(np.abs(w).max(), 1.0)]
    return V[:, order[np.atleast_1d(indices)]]


def eigenvectors_by_overlap(Z, V_theory):
    """
    The sample eigenvector best matching each column of V_theory, located by overlap.

    Rank-based selection assumes the axis of interest is a *leading* PC. That fails whenever
    a deeper structure axis outranks it: on OOA3 the CEU/CHB/YRI splits carry eigenvalues of
    65, 21 and 11 while the recent YRI/YRI2 split sits at ~2, so the sister axis is rank 2
    and asking for rank 0 returns a near-orthogonal vector (overlap 2e-4 against 0.97).
    Use this only when the axis's rank genuinely isn't known ahead of time -- if it is
    (e.g. deterministically rank 2, as measured for the OOA3 sister axis in
    SisterEigenvalue.py/SisterPCAlignment.py), read that rank directly via
    sample_eigenvectors instead: searching the full spectrum for a known-fixed rank only
    inflates the no-signal baseline (see support_overlap_floor) without changing which
    eigenvector gets picked.

    Each theoretical axis is matched to its own argmax, so a permuted or non-contiguous set
    of ranks is handled. Matches are taken greedily and without replacement, which keeps two
    theoretical axes from collapsing onto the same sample eigenvector.
    """
    G = compute_sample_covariance(Z)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    V = V[:, order]

    V_theory = np.atleast_2d(np.asarray(V_theory, dtype=float))
    overlaps = np.abs(V.T @ V_theory)               # (n_candidates, k)

    picked, out = set(), []
    for j in range(V_theory.shape[1]):
        col = overlaps[:, j].copy()
        col[list(picked)] = -1.0
        k = int(np.argmax(col))
        picked.add(k)
        out.append(V[:, k])
    return np.column_stack(out)


def support_overlap_floor(V_theory, n, num_candidates=None, num_draws=20000, seed=0):
    """
    The no-recovery floor for an axis supported on only part of the sample.

    random_overlap_floor(N) is the mean |cos| between a *uniformly random* direction in R^N
    and a fixed axis. Two things break that here:

      * **Support.** The sister axis is exactly zero outside the sister rows, so only those
        rows contribute. The ambient dimension is the size of the axis's support, not N.
        Measured (under the old full-spectrum eigenvectors_by_overlap search): at fixed
        n_1+n_2 = 200 the floor is flat (0.144-0.169) however the 200 are split, so the
        support size is what matters and not any balance-weighted dimension.
      * **Selection.** eigenvectors_by_overlap returns the *best-matching* eigenvector out of
        `num_candidates` candidates, and a maximum over draws sits well above the mean of
        one draw. Pass num_candidates matching how many candidates the caller actually
        searches over -- for a single fixed-rank read (no search at all) that is 1, which
        reduces this to the mean of one draw in the support subspace.

    Both are estimated by direct simulation rather than a closed form: sqrt(2 ln d / d), the
    usual max-of-Gaussians heuristic, is off by 0.54-0.81 across this range.

    Returns the expected max |cos| over num_candidates random directions in the support.
    """
    V_theory = np.atleast_2d(np.asarray(V_theory, dtype=float))
    support = int(np.count_nonzero(np.abs(V_theory).sum(axis=1) > 0))
    if support < 2:
        return 0.0
    d = support if num_candidates is None else int(num_candidates)

    rng = np.random.default_rng(seed)
    V = rng.standard_normal((num_draws, support))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    c = np.abs(V[:, 0])                              # fixed axis, wlog e_1 by isotropy
    usable = (len(c) // d) * d
    return float(c[:usable].reshape(-1, d).max(axis=1).mean())


def random_overlap_floor(N):
    """
    Expected |cos| between a random unit vector and a fixed one in N dimensions.

    E|cos| = sqrt(2 / (pi * N)) to leading order: the coordinates of a uniformly random
    direction are asymptotically Gaussian, so the projection onto a fixed axis is
    N(0, 1/N) and its absolute value has mean sqrt(2/(pi*N)).
    """
    return float(np.sqrt(2.0 / (np.pi * N)))


def analytical_cosine(nu, gamma, N=None, phi=None, eps=None,
                      gamma_raw=None, mu2=None, delta=None, h=None):
    """
    The limiting cosine similarity between a sample PC and its theoretical axis.

    Returned as |cos| = sqrt(overlap^2), matching the scale of the measured cosines.

    Two routes, chosen by what is supplied:

      * Exact (preferred): when phi, gamma_raw and the per-locus weights h are given,
        the overlap comes from src/threshold_and_overlap's exact_nu_c/exact_overlap,
        which sum over every locus individually rather than through the two-moment
        expansion. eps=None is read as eps_a = 0 for all a, which is the equal-tau
        special case of the same path.
      * Two-moment (gamma_eff): when h is absent but mu2/delta are given. Within ~0.5%
        of exact for nu_c at gamma <= 0.05, but degrading to ~10% near threshold at
        large gamma, so it is a fallback rather than the default.
      * Flat fallback: (1 - gamma/nu^2)/(1 + gamma/nu) for nu > sqrt(gamma), else 0.
        This is the eps == 0, eta == 1 limit and is kept only for callers that have not
        been given the extra quantities.

    Note gamma vs gamma_raw. gamma = N/L_eff already carries the mu2 part of the locus
    heteroskedasticity, since L_eff = L_raw/mu2 identically; the theory module applies
    mu2 itself, so it must be handed the raw N/L_raw instead. Passing gamma there would
    apply mu2 twice.

    Below threshold the limiting overlap is 0, but that is an N -> infinity statement: at
    finite N an uninformative PC still has expected |cos| = sqrt(2/(pi*N)) against any
    fixed axis simply because two vectors in finite dimension are never exactly orthogonal.
    Passing N floors the prediction there, so it is comparable with a measurement that can
    never go below it.

    Per-axis values are only individually meaningful when the axes are non-degenerate:
    inside a degenerate multiplet every member shares the same nu, so the predictions are
    identical while the measured per-axis cosines depend on an arbitrary basis choice.
    The mean over axes is the well-posed comparison there.

    Args:
        nu: structure eigenvalue(s), scalar or array.
        gamma: aspect ratio N / L_eff (used by the flat fallback).
        N: sample size. When given, the prediction is floored at the finite-N random
            overlap sqrt(2/(pi*N)); when None the asymptotic 0 is kept.
        phi: sampling proportions n_a/N. Required for the exact route.
        eps: within-population deviations tau_a/tau_bar - 1. None means all zero.
        gamma_raw: aspect ratio N / L_raw. Required for the exact route.
        mu2, delta: locus-weight moments from locus_moments. Used by the two-moment route.
        h: per-locus weights p(1-p) from locus_weights. Selects the exact route.

    Returns:
        (per_axis, mean): the |cos| for each nu, and their mean.
    """
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    have_phi = phi is not None and gamma_raw is not None
    exact = have_phi and h is not None and len(np.asarray(h)) > 0
    two_moment = have_phi and mu2 is not None and delta is not None

    if exact:
        phi = np.asarray(phi, dtype=float)
        eps_a = np.zeros_like(phi) if eps is None else np.asarray(eps, dtype=float)
        h = np.asarray(h, dtype=float)
        nu_c = exact_nu_c(phi, eps_a, h, gamma_raw)
        ov = np.array([exact_overlap(float(v), phi, eps_a, h, gamma_raw, nu_c=nu_c)
                       for v in nu])
    elif two_moment:
        phi = np.asarray(phi, dtype=float)
        eps_a = np.zeros_like(phi) if eps is None else np.asarray(eps, dtype=float)
        nu_c = approx_nu_c(phi, eps_a, gamma_raw, mu2, delta)
        ov = np.array([approx_overlap(float(v), phi, eps_a, gamma_raw, mu2, delta, nu_c=nu_c)
                       for v in nu])
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ov = np.where(nu > np.sqrt(gamma), (1 - gamma / nu ** 2) / (1 + gamma / nu), 0.0)

    per_axis = np.sqrt(np.clip(ov, 0.0, 1.0))
    if N is not None:
        per_axis = np.maximum(per_axis, random_overlap_floor(N))
    return per_axis, float(per_axis.mean())


def _median_se(x):
    """
    Standard error of a median, via 1.4826 * MAD / sqrt(n).

    The sample standard deviation is the spread of the *mean*; using it for a median would
    misstate the error bar, so the MAD-based scale estimate is used instead.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad / np.sqrt(x.size))


def pc_recovery(Z, n, V_theory, indices, L=None, num_replicates=20, seed=0, nu=None, eps=None,
                select_by_overlap=False, n_available=None, L_polymorphic=False):
    """
    How well the sample PCs at `indices` recover the theoretical axes V_theory.

    Each replicate independently subsamples n_a samples per subpopulation (and L SNPs, if
    given), keeps the polymorphic loci, builds the GRM, and compares its PCs against
    V_theory by principal angles.

    Args:
        Z: (N_full, L_full) genotype matrix, rows block-ordered by subpopulation.
        n: per-subpopulation sample sizes, length K.
        V_theory: (sum(n), k) orthonormal theoretical axes, e.g. from
            theoretical_eigenvectors().
        indices: which sample PCs to compare against them, 0 = largest eigenvalue.
        L: SNP columns to subsample per replicate. None keeps all polymorphic loci.
        L_polymorphic: make L the exact polymorphic width of each replicate rather than a
            pre-filter draw count -- see subsample(). Keeps L identical across cells whose
            sample sizes differ, which a sweep over n needs.
        num_replicates: independent subsamples.
        seed: base seed; replicate r uses [seed, r].
        nu: the theoretical structure eigenvalues for V_theory, as returned by
            theoretical_eigenvectors. Used for the analytical_cos prediction, and to detect
            negative structure eigenvalues, which force ranking the sample PCs by |lambda|
            rather than by lambda (see sample_eigenvectors). Omitting it leaves
            analytical_cos as NaN.
        select_by_overlap: locate the sample eigenvectors by maximum overlap with V_theory
            instead of by rank, ignoring `indices`. Required whenever the axis of interest
            is not a leading PC -- see eigenvectors_by_overlap.
        n_available: per-subpopulation block sizes actually present in Z, length K. None
            assumes every block has the same size (see subsample()); pass this for
            demographies with unequal per-subpopulation sample sizes.

    Returns:
        dict with, for the subspace as a whole, median_cos / se_cos and the matching
        zero-parameter prediction analytical_cos; the same quantities per axis in
        cos_per_axis / se_cos_per_axis / analytical_cos_per_axis; the median Procrustes
        distance; and the measured L_eff, L_raw and gamma = N / L_eff.

        cos_replicates holds the individual replicate cosines that median_cos reduces
        (cos_replicates_per_axis the per-axis matrix), so a caller can record the whole
        cloud and fit a transition curve through it -- see crossing_fit.
    """
    n = np.asarray(n, dtype=int)
    indices = np.atleast_1d(indices)
    V_theory = np.asarray(V_theory, dtype=float)
    if V_theory.shape[0] != n.sum():
        raise ValueError(f"V_theory has {V_theory.shape[0]} rows but n sums to {n.sum()}")
    if V_theory.shape[1] != len(indices):
        raise ValueError(f"V_theory has {V_theory.shape[1]} columns but {len(indices)} "
                         f"indices were requested")

    by_magnitude = nu is not None and np.any(np.asarray(nu) < 0)

    cos_all, dists, L_effs, L_raws = [], [], [], []
    mu2s, deltas = [], []
    for rep in range(num_replicates):
        rng = np.random.default_rng([seed, rep])
        sub = subsample(Z, n, L, rng, n_available=n_available,
                        L_polymorphic=L_polymorphic)
        V_s = (eigenvectors_by_overlap(sub, V_theory) if select_by_overlap
               else sample_eigenvectors(sub, indices, by_magnitude=by_magnitude))
        cos_all.append(principal_cosines(V_theory, V_s))
        dists.append(procrustes_distance(V_theory, V_s))
        L_effs.append(effective_L(sub))
        L_raws.append(sub.shape[1])
        m2, _, dl = locus_moments(sub)
        mu2s.append(m2)
        deltas.append(dl)
        if rep == 0:
            # The exact route needs the whole weight vector. One replicate's is
            # representative (mu2/delta vary by <1% across replicates) and keeping all of
            # them would be num_replicates x L floats per cell.
            h_rep0 = locus_weights(sub)

    cos_all = np.array(cos_all)                 # (num_replicates, k)
    mean_cos = cos_all.mean(axis=1)             # subspace similarity per replicate

    L_eff_med = float(np.median(L_effs))
    L_raw_med = float(np.median(L_raws))
    gamma = n.sum() / L_eff_med
    # gamma_raw is the aspect ratio the theory module expects: it applies mu2 itself, and
    # gamma = N/L_eff already has mu2 folded in (L_eff == L_raw/mu2 exactly), so passing
    # gamma there would square the correction.
    mu2_med = float(np.median(mu2s))
    delta_med = float(np.median(deltas))
    gamma_raw = n.sum() / L_raw_med

    # The zero-parameter prediction alongside the measurement: gamma comes from the
    # measured L_eff and nu from the demography, so nothing here is fitted.
    if nu is None:
        analytical_per_axis = np.full(len(indices), np.nan)
        analytical_mean = np.nan
    else:
        analytical_per_axis, analytical_mean = analytical_cosine(
            nu, gamma, N=n.sum(), phi=n / n.sum(), eps=eps,
            gamma_raw=gamma_raw, mu2=mu2_med, delta=delta_med, h=h_rep0)

    return dict(
        median_cos=float(np.median(mean_cos)),
        se_cos=_median_se(mean_cos),
        # Per-replicate values, kept so callers can fit a curve through the whole cloud
        # rather than through the medians: cos_replicates is the subspace similarity of
        # each replicate, cos_replicates_per_axis the (num_replicates, k) matrix behind it.
        cos_replicates=mean_cos,
        cos_replicates_per_axis=cos_all,
        procrustes_replicates=np.asarray(dists, dtype=float),
        cos_per_axis=np.median(cos_all, axis=0),
        se_cos_per_axis=np.array([_median_se(cos_all[:, j]) for j in range(cos_all.shape[1])]),
        analytical_cos=analytical_mean,
        analytical_cos_per_axis=np.asarray(analytical_per_axis, dtype=float),
        procrustes_d=float(np.median(dists)),
        se_procrustes=_median_se(dists),
        L_eff=L_eff_med,
        L_raw=L_raw_med,
        gamma=gamma,
        gamma_raw=gamma_raw,
        mu2=mu2_med,
        delta=delta_med,
        h=h_rep0,
        nu=np.atleast_1d(np.asarray(nu, dtype=float)) if nu is not None else None,
        N=int(n.sum()), K=len(n), k=len(indices),
        num_replicates=num_replicates, seed=seed)

# %%
