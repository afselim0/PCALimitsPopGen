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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMULATION_ROOT = os.path.join(_PROJECT_ROOT, "results", "simulations")


def _parse_simID(simID):
    """
    Parameters encoded in a filename stem, as {name: float}.

    Covers every convention in results/simulations: Fst<v>, m<v>, and t<v>_Fst<v>. A stem
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


def subsample(Z, n, L=None, rng=None):
    """
    Draw n_a samples from each subpopulation and (optionally) L SNP columns.

    Rows of Z are assumed block-ordered with Z.shape[0] // K samples per subpopulation,
    which is how every simulation in src/simulations writes them.

    Monomorphic loci are dropped *after* the rows are subset, not before: a locus that
    segregates in the full sample may be fixed in the subsample, and leaving those columns
    in would inflate the raw SNP count with columns that carry no signal.

    Args:
        Z: (N_full, L_full) genotype matrix.
        n: per-subpopulation sample sizes, length K.
        L: number of SNP columns to draw before the polymorphic filter. None keeps all.
        rng: numpy Generator.

    Returns:
        (sum(n), L_polymorphic) float32 array.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = np.asarray(n, dtype=int)
    K = len(n)
    n_per_pop = Z.shape[0] // K
    if Z.shape[0] % K:
        raise ValueError(f"{Z.shape[0]} rows do not divide evenly into K={K} subpopulations")
    if np.any(n > n_per_pop):
        raise ValueError(f"requested {n.max()} samples but only {n_per_pop} are available "
                         f"per subpopulation")

    idx = np.sort(np.concatenate([a * n_per_pop + rng.choice(n_per_pop, n_a, replace=False)
                                  for a, n_a in enumerate(n)]))
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

    Each theoretical axis is matched to its own argmax, so a permuted or non-contiguous set
    of ranks is handled. Matches are taken greedily and without replacement, which keeps two
    theoretical axes from collapsing onto the same sample eigenvector.
    """
    G = compute_sample_covariance(Z)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    V = V[:, order]

    V_theory = np.atleast_2d(np.asarray(V_theory, dtype=float))
    overlaps = np.abs(V.T @ V_theory)               # (N, k)

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
        Measured: at fixed n_1+n_2 = 200 the floor is flat (0.144-0.169) however the 200 are
        split, so the support size is what matters and not any balance-weighted dimension.
      * **Selection.** eigenvectors_by_overlap returns the *best-matching* eigenvector out of
        many candidates, and a maximum over draws sits well above the mean of one draw.

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


def analytical_cosine(nu, gamma, N=None):
    """
    The BBP limiting cosine similarity between a sample PC and its theoretical axis.

        overlap^2 = (1 - gamma/nu^2) / (1 + gamma/nu)    for nu > sqrt(gamma), else 0

    Returned as |cos| = sqrt(overlap^2), matching the scale of the measured cosines.

    Below threshold the limiting overlap is 0, but that is an N -> infinity statement: at
    finite N an uninformative PC still has expected |cos| = sqrt(2/(pi*N)) against any
    fixed axis simply because two vectors in finite dimension are never exactly orthogonal.
    Passing N floors the prediction there, so it is comparable with a measurement that can
    never go below it. The floor is applied as a maximum rather than only where the BBP
    value is 0, which keeps the prediction continuous; in practice BBP already exceeds the
    floor at its first positive point, so the two agree.

    Per-axis values are only individually meaningful when the axes are non-degenerate:
    inside a degenerate multiplet every member shares the same nu, so the predictions are
    identical while the measured per-axis cosines depend on an arbitrary basis choice.
    The mean over axes is the well-posed comparison there.

    Args:
        nu: structure eigenvalue(s), scalar or array.
        gamma: aspect ratio N / L_eff.
        N: sample size. When given, the prediction is floored at the finite-N random
            overlap sqrt(2/(pi*N)); when None the asymptotic 0 is kept.

    Returns:
        (per_axis, mean): the |cos| for each nu, and their mean.
    """
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
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


def pc_recovery(Z, n, V_theory, indices, L=None, num_replicates=20, seed=0, nu=None,
                select_by_overlap=False):
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

    Returns:
        dict with, for the subspace as a whole, median_cos / se_cos and the matching
        zero-parameter prediction analytical_cos; the same quantities per axis in
        cos_per_axis / se_cos_per_axis / analytical_cos_per_axis; the median Procrustes
        distance; and the measured L_eff, L_raw and gamma = N / L_eff.
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
    for rep in range(num_replicates):
        rng = np.random.default_rng([seed, rep])
        sub = subsample(Z, n, L, rng)
        V_s = (eigenvectors_by_overlap(sub, V_theory) if select_by_overlap
               else sample_eigenvectors(sub, indices, by_magnitude=by_magnitude))
        cos_all.append(principal_cosines(V_theory, V_s))
        dists.append(procrustes_distance(V_theory, V_s))
        L_effs.append(effective_L(sub))
        L_raws.append(sub.shape[1])

    cos_all = np.array(cos_all)                 # (num_replicates, k)
    mean_cos = cos_all.mean(axis=1)             # subspace similarity per replicate

    L_eff_med = float(np.median(L_effs))
    gamma = n.sum() / L_eff_med

    # The zero-parameter prediction alongside the measurement: gamma comes from the
    # measured L_eff and nu from the demography, so nothing here is fitted.
    if nu is None:
        analytical_per_axis = np.full(len(indices), np.nan)
        analytical_mean = np.nan
    else:
        analytical_per_axis, analytical_mean = analytical_cosine(nu, gamma, N=n.sum())

    return dict(
        median_cos=float(np.median(mean_cos)),
        se_cos=_median_se(mean_cos),
        cos_per_axis=np.median(cos_all, axis=0),
        se_cos_per_axis=np.array([_median_se(cos_all[:, j]) for j in range(cos_all.shape[1])]),
        analytical_cos=analytical_mean,
        analytical_cos_per_axis=np.asarray(analytical_per_axis, dtype=float),
        procrustes_d=float(np.median(dists)),
        se_procrustes=_median_se(dists),
        L_eff=L_eff_med,
        L_raw=float(np.median(L_raws)),
        gamma=gamma,
        nu=np.atleast_1d(np.asarray(nu, dtype=float)) if nu is not None else None,
        N=int(n.sum()), K=len(n), k=len(indices),
        num_replicates=num_replicates, seed=seed)

# %%
