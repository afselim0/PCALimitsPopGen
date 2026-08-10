#%%
import numpy as np
import pandas as pd
import msprime
from tqdm import tqdm
#%%
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
 

class DemographySpectrum:
    def __init__(self, demography, haplotypes = None):
        self.demography = demography
        self.pops = [pop.name for pop in demography.populations if pop.default_sampling_time == 0]
        self.pops.sort()
        self.num_pops = len(self.pops)
        self.pop_coal_times = None
        self.haplotypes = haplotypes
    
    def compute_pop_coal_times(self):
        dbg = self.demography.debug()
        self.pop_coal_times = pd.DataFrame(np.zeros((self.num_pops, self.num_pops)),
                                           index=self.pops, columns=self.pops)
        for i,pop1 in enumerate(self.pops):
            for j,pop2 in enumerate(self.pops):
                if i>j:
                    continue
                elif i == j:
                    lineags = {pop1: 2}
                elif i < j:
                    lineags = {pop1: 1, pop2: 1}
                mean_coal_time = dbg.mean_coalescence_time(lineages=lineags)
                self.pop_coal_times.iloc[i, j] = mean_coal_time
                self.pop_coal_times.iloc[j, i] = mean_coal_time
        return self.pop_coal_times

    def expand_pop_coal_times(self,samples_dict):  
        """Expand the within- and between-population coalescent times to between individuals

        Args:
            samples_dict (dict): A dictionary where keys are population names and values are the number of diploid individuals from that population. (each individual contributes 2 haploid samples)

        Returns:
            np.ndarray: a matrix of mean pairwise coalescent times between samples, where the order of samples is determined by the order of populations in self.pops and the number of samples specified in samples_dict.
        """
        # Create a full matrix of coalescent times
        if self.pop_coal_times is None:
            self.compute_pop_coal_times()
        pop_coal_times = self.pop_coal_times
        assert all(pop in self.pops for pop in samples_dict.keys()), "All populations in samples_dict must be present in self.pops"
        assert all(isinstance(samples_dict[pop], int) and samples_dict[pop] > 0 for pop in samples_dict.keys()), "All values in samples_dict must be positive integers"
        n_total = sum(samples_dict[pop] * 2 for pop in self.pops)  # Total number of haploid samples
        samples_dict_haploid = {pop: samples_dict[pop] * 2 for pop in self.pops}  # Convert to haploid samples
        full_matrix = np.zeros((n_total, n_total))

        ## Fill in the full matrix with the appropriate coalescent times
        idx_start_values, idx_start = {}, 0
        for pop in self.pops:
            idx_start_values[pop] = idx_start
            idx_start += samples_dict_haploid[pop]
        for i, pop1 in enumerate(self.pops):
            i_start = idx_start_values[pop1]
            i_end = i_start + samples_dict_haploid[pop1]
            for j, pop2 in enumerate(self.pops):
                j_start = idx_start_values[pop2]
                j_end = j_start + samples_dict_haploid[pop2]
                full_matrix[i_start:i_end, j_start:j_end] = pop_coal_times.iloc[i, j]
        # A haploid sample coalesces with itself at time 0
        np.fill_diagonal(full_matrix, 0)

        sample_labels = [f"{pop}_{k}" for pop in self.pops for k in range(samples_dict_haploid[pop])]
        return pd.DataFrame(full_matrix, index=sample_labels, columns=sample_labels)

    def simulate_haplotypes(self, n=100, L=int(1e6),random_seed=42):
        """Simulate L independent biallelic SNPs by placing a random mutation on a random tree generated from a specified demography

        Args:
            n (int, optional): number of diploid individuals from each population. Defaults to 100. (half of the number of per-population haploid sample size)
            L (int, optional): Number of SNPs to simulate. Defaults to int(1e5).
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        samples_dict = {pop: int(n) for pop in self.pops}
        ts_samples = msprime.sim_ancestry(demography=self.demography, samples=samples_dict,
                                       random_seed=random_seed, num_replicates=L,
                                       sequence_length=1, ploidy=2)
        n_tot = 2*sum(samples_dict.values())
        haplotypes = np.zeros((n_tot, L))
        for i, ts in tqdm(enumerate(ts_samples)):
            tree = ts.first()
            nodes, ancestral_branch_lengths = [], []
            for u in tree.nodes():
                bl = tree.branch_length(u)
                if bl > 0:
                    nodes.append(u)
                    ancestral_branch_lengths.append(bl)
            ancestral_branch_lengths = np.array(ancestral_branch_lengths)
            probabilities = ancestral_branch_lengths / np.sum(ancestral_branch_lengths)
            mutation_node = np.random.choice(nodes, p=probabilities)
            haplotypes[np.array(list(tree.leaves(mutation_node))), i] = 1
        self.haplotypes = haplotypes    
        return haplotypes

    def subsample_haplotypes(self, samples_dict, L=int(1e5), random_seed=42):
        """Subsample haplotypes from the simulated haplotype matrix based on the specified number of diploid individuals per population.

        Args:
            samples_dict (dict): A dictionary where keys are population names and values are the number of diploid individuals to sample from that population.
            L (int, optional): Number of SNPs to subsample. Defaults to int(1e5).
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        np.random.seed(random_seed)
        if self.haplotypes is None:
            raise ValueError("Haplotypes have not been simulated yet. Please run simulate_haplotypes() first.")
        n_total = sum(samples_dict[pop] * 2 for pop in self.pops)  # Total number of haploids to sample
        n_per_pop_full = self.haplotypes.shape[0] // len(self.pops)  # number of samples in the full simulated matrix of haplotypes 
        n_total_full = self.haplotypes.shape[0]
        idx_start_values = dict(zip(self.pops, np.arange(0, n_total_full, n_per_pop_full)))
        subsampled_indices = []
        for pop in self.pops:
            n_samples_pop = samples_dict[pop] * 2  # number of haploids to sample from this population
            idx_start = idx_start_values[pop]
            idx_end = idx_start + n_per_pop_full
            pop_indices = np.arange(idx_start, idx_end)
            sampled_indices = np.random.choice(pop_indices, size=n_samples_pop, replace=False)
            subsampled_indices.extend(sampled_indices)
        subsampled_indices = np.array(subsampled_indices)
        subsampled_haplotypes = self.haplotypes[subsampled_indices, :]
        is_polymorphic = subsampled_haplotypes.sum(axis=0) > 0
        subsampled_haplotypes = subsampled_haplotypes[:, is_polymorphic]
        num_polymorphic_snps = subsampled_haplotypes.shape[1]
        if num_polymorphic_snps < L:
            raise ValueError(f"Not enough polymorphic SNPs ({num_polymorphic_snps}) to subsample {L} SNPs. Please reduce L or simulate more haplotypes.")
        selected_snp_indices = np.random.choice(num_polymorphic_snps, size=L, replace=False)
        subsampled_haplotypes = subsampled_haplotypes[:, selected_snp_indices]
        return subsampled_haplotypes