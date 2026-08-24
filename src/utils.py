#%%
import re
from concurrent.futures import ProcessPoolExecutor
import msprime
from tqdm import tqdm
import numpy as np
import pandas as pd

def center_matrix(X):
    """Double-center a matrix: H @ X @ H, with H = I - (1/n) * 11^T."""
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return -H @ X @ H

def _off_diagonal_mean(M):
    """Mean of M over off-diagonal entries only (excludes the zero diagonal)."""
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)].mean()


def compute_sample_covariance(Z, mean_normalize=True):
    """
    Sample GRM from a binary genotype matrix Z, shape (n, L).

    Delta_ij = mean squared Hamming distance per locus between samples i, j.
    G = -H @ Delta @ H, optionally rescaled by the off-diagonal mean distance.
    """
    n, L = Z.shape
    sq_norms = np.sum(Z ** 2, axis=1)
    Delta = (sq_norms[:, None] + sq_norms[None, :] - 2 * Z @ Z.T) / L

    G = center_matrix(Delta)
    if mean_normalize:
        G /= (_off_diagonal_mean(Delta) / 2)
    return G


def compute_expected_covariance(T, mean_normalize=True):
    """
    E[GRM] from a sample-level mean pairwise coalescent time matrix T
    (diagonal zero). Same convention as compute_sample_covariance.
    """
    G = center_matrix(T)
    if mean_normalize:
        G /= (_off_diagonal_mean(T) / 2)
    return G

def _sampling_populations(demography):
    """Sampleable population names from a demography, in spatial/numeric order."""
    pops = [pop.name for pop in demography.populations
            if pop.default_sampling_time in (0, None)]
    pops.sort(key=lambda name: [int(tok) if tok.isdigit() else tok
                                 for tok in re.split(r'(\d+)', name)])
    return pops


def _simulate_haplotype_chunk(demography, samples_dict, L, random_seed, show_progress):
    """Simulate L independent biallelic SNPs for one chunk (one process/seed)."""
    ts_samples = msprime.sim_ancestry(demography=demography, samples=samples_dict,
                                    random_seed=random_seed, num_replicates=L,
                                    sequence_length=1, ploidy=2)
    n_tot = 2*sum(samples_dict.values())
    haplotypes = np.zeros((n_tot, L), dtype=np.bool_)
    rng = np.random.default_rng(random_seed)
    for i, ts in tqdm(enumerate(ts_samples), total=L, disable=not show_progress):
        tree = ts.first()
        nodes, ancestral_branch_lengths = [], []
        for u in tree.nodes():
            bl = tree.branch_length(u)
            if bl > 0:
                nodes.append(u)
                ancestral_branch_lengths.append(bl)
        ancestral_branch_lengths = np.array(ancestral_branch_lengths)
        probabilities = ancestral_branch_lengths / np.sum(ancestral_branch_lengths)
        mutation_node = rng.choice(nodes, p=probabilities)
        haplotypes[np.array(list(tree.leaves(mutation_node))), i] = True
    return haplotypes


def simulate_haplotypes(demography, n=100, L=int(1e6), random_seed=42, n_workers=1,
                         show_progress=True):
    """Simulate L independent biallelic SNPs by placing a random mutation on a random tree generated from a specified demography

    Args:
        n (int or dict, optional): number of diploid individuals per population. A scalar
            applies the same size to every population; a dict {pop_name: n_a} allows
            per-population sizes. Defaults to 100.
        L (int, optional): Number of SNPs to simulate. Defaults to int(1e5).
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_workers (int, optional): number of processes to split the L replicates across. Each
            replicate is an independent coalescent simulation, so this parallelizes with no
            cross-worker communication; workers use random_seed + worker index. Defaults to 1
            (serial, same behavior/output as before).
        show_progress (bool, optional): show a tqdm bar over the L replicates. Defaults to True;
            set False when the caller already tracks progress at a coarser level.
    """
    if isinstance(n, dict):
        samples_dict = {pop: int(n[pop]) for pop in _sampling_populations(demography)}
    else:
        samples_dict = {pop: int(n) for pop in _sampling_populations(demography)}

    if n_workers <= 1:
        return _simulate_haplotype_chunk(demography, samples_dict, L, random_seed, show_progress)

    n_workers = min(n_workers, L)
    chunk_sizes = np.diff(np.linspace(0, L, n_workers + 1).astype(int))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(_simulate_haplotype_chunk, demography, samples_dict,
                      int(chunk_L), random_seed + w, show_progress and w == 0)
            for w, chunk_L in enumerate(chunk_sizes) if chunk_L > 0
        ]
        chunks = [f.result() for f in futures]
    return np.concatenate(chunks, axis=1)

def _simulate_counts_chunk(demography, samples_dict, L, random_seed, show_progress):
    """Derived allele *count* for L independent SNPs, without building the genotype matrix."""
    ts_samples = msprime.sim_ancestry(demography=demography, samples=samples_dict,
                                      random_seed=random_seed, num_replicates=L,
                                      sequence_length=1, ploidy=2)
    counts = np.zeros(L, dtype=np.int32)
    rng = np.random.default_rng(random_seed)
    for i, ts in tqdm(enumerate(ts_samples), total=L, disable=not show_progress):
        tree = ts.first()
        nodes, ancestral_branch_lengths = [], []
        for u in tree.nodes():
            bl = tree.branch_length(u)
            if bl > 0:
                nodes.append(u)
                ancestral_branch_lengths.append(bl)
        ancestral_branch_lengths = np.array(ancestral_branch_lengths)
        probabilities = ancestral_branch_lengths / np.sum(ancestral_branch_lengths)
        mutation_node = rng.choice(nodes, p=probabilities)
        # The genotype column is the indicator of this node's leaf set, so its derived
        # allele count is just how many leaves it has -- no need to materialize the column.
        counts[i] = tree.num_samples(mutation_node)
    return counts


def simulate_allele_counts(demography, n=100, L=int(1e6), random_seed=42, n_workers=1,
                           show_progress=True):
    """
    Derived allele counts for L independent biallelic SNPs, one integer per SNP.

    Same model as simulate_haplotypes -- one mutation placed on a random ancestral branch
    of each independent tree -- but records only how many samples carry the derived allele
    rather than which ones. For any quantity that depends on the genotype matrix solely
    through its column allele frequencies (L_eff and the site frequency spectrum, for
    instance) this is exactly equivalent and costs O(L) storage instead of O(N * L): at
    N = 1e4 and L = 1e6 that is 4 MB rather than 10 GB.

    Args:
        n: diploid individuals per population; each contributes 2 haploid samples.
        L: number of independent SNPs.
        n_workers: processes to split the L replicates across.

    Returns:
        (counts, n_haploid): the length-L int32 array of derived allele counts, and the
        total number of haploid samples they are counts out of.
    """
    samples_dict = {pop: int(n) for pop in _sampling_populations(demography)}
    n_haploid = 2 * sum(samples_dict.values())

    if n_workers <= 1:
        return _simulate_counts_chunk(demography, samples_dict, L, random_seed,
                                      show_progress), n_haploid

    n_workers = min(n_workers, L)
    chunk_sizes = np.diff(np.linspace(0, L, n_workers + 1).astype(int))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(_simulate_counts_chunk, demography, samples_dict,
                      int(chunk_L), random_seed + w, show_progress and w == 0)
            for w, chunk_L in enumerate(chunk_sizes) if chunk_L > 0
        ]
        chunks = [f.result() for f in futures]
    return np.concatenate(chunks), n_haploid


def subsample_allele_counts(counts, n_full, n, rng):
    """
    Down-sample derived allele counts from n_full haploid samples to n, hypergeometrically.

    Drawing n of the n_full samples without replacement and counting how many carry the
    derived allele is exactly a hypergeometric draw, so a subsample's frequency spectrum
    can be obtained from the full one without ever storing the genotype matrix.

    Returns the counts out of n, including the zeros and n's -- loci that are monomorphic
    in the subsample. Callers filter those out as usual.
    """
    return rng.hypergeometric(counts, n_full - counts, n)


def compute_pop_coal_times(demography, pops):
    """Mean pairwise coalescence time between (and within) each population."""
    dbg = demography.debug()
    num_pops = len(pops)
    pop_coal_times = pd.DataFrame(np.zeros((num_pops, num_pops)), index=pops, columns=pops)
    for i, pop1 in enumerate(pops):
        for j, pop2 in enumerate(pops):
            if i > j:
                continue
            elif i == j:
                lineages = {pop1: 2}
            else:
                lineages = {pop1: 1, pop2: 1}
            mean_coal_time = dbg.mean_coalescence_time(lineages=lineages)
            pop_coal_times.iloc[i, j] = mean_coal_time
            pop_coal_times.iloc[j, i] = mean_coal_time
    return pop_coal_times

def expand_pop_coal_times(pop_coal_times, pops, samples_dict):
    """Expand the within- and between-population coalescent times to between individuals

    Args:
        pop_coal_times (pd.DataFrame): KxK matrix of mean pairwise coalescent times between populations, indexed/columned by population name.
        pops (list): population names, defining the order in which samples are laid out.
        samples_dict (dict): A dictionary where keys are population names and values are the number of diploid individuals from that population. (each individual contributes 2 haploid samples)

    Returns:
        pd.DataFrame: a matrix of mean pairwise coalescent times between samples, where the order of samples is determined by the order of populations in pops and the number of samples specified in samples_dict.
    """
    assert all(pop in pops for pop in samples_dict.keys()), "All populations in samples_dict must be present in pops"
    assert all(isinstance(samples_dict[pop], int) and samples_dict[pop] > 0 for pop in samples_dict.keys()), "All values in samples_dict must be positive integers"
    n_total = sum(samples_dict[pop] * 2 for pop in pops)  # Total number of haploid samples
    samples_dict_haploid = {pop: samples_dict[pop] * 2 for pop in pops}  # Convert to haploid samples
    full_matrix = np.zeros((n_total, n_total))

    ## Fill in the full matrix with the appropriate coalescent times
    idx_start_values, idx_start = {}, 0
    for pop in pops:
        idx_start_values[pop] = idx_start
        idx_start += samples_dict_haploid[pop]
    for i, pop1 in enumerate(pops):
        i_start = idx_start_values[pop1]
        i_end = i_start + samples_dict_haploid[pop1]
        for j, pop2 in enumerate(pops):
            j_start = idx_start_values[pop2]
            j_end = j_start + samples_dict_haploid[pop2]
            full_matrix[i_start:i_end, j_start:j_end] = pop_coal_times.iloc[i, j]
    # A haploid sample coalesces with itself at time 0
    np.fill_diagonal(full_matrix, 0)

    sample_labels = [f"{pop}_{k}" for pop in pops for k in range(samples_dict_haploid[pop])]
    return pd.DataFrame(full_matrix, index=sample_labels, columns=sample_labels)

def center_and_normalize(X):
    """
    Double-center and normalize: G = -(H @ X @ H) / X_bar, with H = I - (1/n) * 11^T
    and X_bar the mean of the off-diagonal entries of X.

    X_bar is taken from the raw input X (not from the centered, negated matrix), per
    G = -(H T H)/T_bar and G_hat = (H Delta H)/(L * Delta_bar); normalizing by the
    off-diagonal mean of -(H X H) instead would flip the sign of the whole spectrum.
    """
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    off_diag_mean = X[~np.eye(n, dtype=bool)].mean()
    return -(H @ X @ H) / off_diag_mean

def expand_coal_times(T, sample_sizes):
    """
    Expand a population-level (K x K) mean pairwise coalescent time matrix T to the
    sample-level (N x N) matrix. Each entry T_ij fills a block of size (n_i, n_j),
    where n_i and n_j are the sample sizes for populations i and j.

    The sample-level diagonal is zero: a sample's coalescent time with itself is 0,
    while two *distinct* samples from population a coalesce at tau_a = T[a, a]. Leaving
    tau_a on the diagonal would make each diagonal block constant, collapsing the
    N - K "floor" eigenvalues of the resulting covariance matrix.
    """
    num_pops = len(sample_sizes)
    expanded_T = np.zeros((sum(sample_sizes), sum(sample_sizes)))

    start_i = 0
    for i in range(num_pops):
        end_i = start_i + sample_sizes[i]
        start_j = 0
        for j in range(num_pops):
            end_j = start_j + sample_sizes[j]
            expanded_T[start_i:end_i, start_j:end_j] = T[i, j]
            start_j = end_j
        start_i = end_i

    np.fill_diagonal(expanded_T, 0.0)
    return expanded_T

def compute_F_hat(T, sample_sizes):
    """
    Compute the F_hat matrix from a population-level mean pairwise coalescent time
    matrix T (K x K, with diag(T) = tau the within-population mean coalescent times)
    and the sample sizes n_a for each population.

    phi_a   = n_a / N,                          psi_a = sqrt(phi_a)
    eps_a   = tau_a / tau_bar - 1,              tau_bar = sum_a phi_a * tau_a
    (F_2)_ab = T_ab - (tau_a + tau_b) / 2
    F_ab    = -N / tau_bar * sqrt(phi_a * phi_b) * (F_2)_ab   (zero diagonal)
    P_hat   = I_K - psi psi^T
    F_hat   = P_hat @ (diag(eps) + F) @ P_hat
    """
    T = np.asarray(T, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    K = T.shape[0]
    N = n.sum()

    phi = n / N
    psi = np.sqrt(phi)

    tau = np.diag(T)
    tau_bar = phi @ tau
    eps = tau / tau_bar - 1

    F_2 = T - (tau[:, None] + tau[None, :]) / 2
    F = -N / tau_bar * np.outer(psi, psi) * F_2
    np.fill_diagonal(F, 0.0)

    P_hat = np.eye(K) - np.outer(psi, psi)

    return P_hat @ (np.diag(eps) + F) @ P_hat

def K2_structure_eigenvalue(sample_sizes, T_within, T_between):
    """
    Compute the single eigenvalue of a K = 2 demogrpahy with two populations and a single split time, given the sample sizes and mean coalescent times.

    Args:
        sample_sizes: list or array of length 2, the sample sizes for the two populations
        T_within: list or array of length 2, the mean coalescent times within each population
        T_between: float, the mean coalescent time between the two populations
    """
    assert len(sample_sizes) == 2, "sample_sizes must have length 2"
    assert len(T_within) == 2, "T_within must have length 2"
    n1, n2 = sample_sizes
    tau_1, tau_2 = T_within
    N = n1 + n2
    phi_1 = n1 / N
    phi_2 = 1 - phi_1
    tau_bar = phi_1 * tau_1 + phi_2 * tau_2
    F2 = T_between - (tau_1 + tau_2) / 2
    return 2*N*phi_1*phi_2*F2/tau_bar - (phi_1 - phi_2)*(tau_1 - tau_2)/tau_bar

def K3_structure_eigenvalues(T, n):
    """Analytically computes the two structure eigenvalues of a K=3 demography given a generic matrix of pairwise coalescent times and a generic number of samples

    Args:
        T (np.ndarray): A 3x3 matrix of pairwise coalescent times.
        n (list): A list or array of length 3, the sample sizes for the three populations.

    Returns:
        np.ndarray: An array of length 2, the two structure eigenvalues.
    """
    T = np.asarray(T, dtype=float)
    n = np.asarray(n, dtype=float)
    assert T.shape == (3, 3), "T must be a 3x3 matrix"
    assert len(n) == 3, "n must have length 3"
    N = n.sum()
    phi = n / N
    phi1, phi2, phi3 = phi
    s = phi1 + phi2

    tau = np.diag(T)
    tau1, tau2, tau3 = tau
    tau_bar = phi @ tau
    eps = tau / tau_bar - 1.0
    eps1, eps2, eps3 = eps

    def F2(a, b):
        return T[a, b] - (tau[a] + tau[b]) / 2.0

    g12 = -N * np.sqrt(phi1 * phi2) * F2(0, 1) / tau_bar
    g13 = -N * np.sqrt(phi1 * phi3) * F2(0, 2) / tau_bar
    g23 = -N * np.sqrt(phi2 * phi3) * F2(1, 2) / tau_bar

    M11 = (phi2 * eps1 + phi1 * eps2 - 2 * np.sqrt(phi1 * phi2) * g12) / s
    M22 = (phi1 * phi3 * eps1 + phi2 * phi3 * eps2 + s**2 * eps3
           + 2 * np.sqrt(phi1 * phi3 * phi2 * phi3) * g12
           - 2 * np.sqrt(phi1 * phi3) * s * g13
           - 2 * np.sqrt(phi2 * phi3) * s * g23) / s
    M12 = (np.sqrt(phi1 * phi2 * phi3) * (eps1 - eps2)
           + np.sqrt(phi3) * (phi2 - phi1) * g12
           - np.sqrt(phi2) * s * g13
           + np.sqrt(phi1) * s * g23) / s

    trace = M11 + M22
    disc = np.sqrt(((M11 - M22) / 2.0) ** 2 + M12 ** 2)
    return np.array([trace / 2.0 + disc, trace / 2.0 - disc])
# %%
