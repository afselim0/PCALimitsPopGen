"""
Genotype preprocessing applied before PCA.

Defaults: diploid individuals, mean-centered genotypes, no frequency filter. A sweep config
overrides them with a `preprocess` block, which runner.run_one_task passes to configure():

    preprocess: {standardize: true, maf_min: 0.05}
    preprocess: {diploid: false}

diploid      The simulations hold haplotypes. Consecutive haplotypes of each population are
             paired into diploid individuals: utils.compute_sample_covariance averages the
             haploid GRM over each pair, which is the GRM of 0/1/2 genotypes written in
             haplotype coordinates (each individual occupies two identical rows). Every
             vector therefore keeps length N = number of haplotypes, the structure
             eigenvalues nu and the floor levels are those of the haploid theory, and only
             the number of noise dimensions halves: gamma = (N/2)/L. Sample sizes per
             population must be even.
standardize  Every SNP column scaled to unit variance; the locus weights are then flat.
maf_min      Drops SNPs whose minor allele frequency in the analysed subsample is below it.

L_eff and L_raw are reported on the haploid-equivalent scale (multiplied by the ploidy), so
that gamma = N/L_eff holds with N the number of haplotypes and no formula downstream needs
to know the ploidy. Divide by ploidy() for the number of SNPs.
"""
import numpy as np

DIPLOID = True
STANDARDIZE = False
MAF_MIN = 0.0


def configure(standardize=False, maf_min=0.0, diploid=True):
    global STANDARDIZE, MAF_MIN, DIPLOID
    STANDARDIZE, MAF_MIN, DIPLOID = bool(standardize), float(maf_min), bool(diploid)


def ploidy():
    return 2 if DIPLOID else 1


def keep(freq):
    """Columns kept: polymorphic, with minor allele frequency >= MAF_MIN."""
    maf = np.minimum(freq, 1.0 - freq)
    return (maf > 0) & (maf >= MAF_MIN - 1e-6)


def even(n):
    """Sample sizes usable under the current ploidy: odd counts are lowered by one."""
    n = np.asarray(n)
    return n - (n % 2) if DIPLOID else n


def nearest_usable(x):
    """The sample size closest to the real number x that the current ploidy allows."""
    return max(2 * int(round(x / 2.0)), 2) if DIPLOID else max(int(x), 1)


def rank(N):
    """Number of non-zero eigenvalues of the centered GRM of N haplotypes."""
    return N // ploidy() - 1


def individuals(table):
    """One row per individual from a per-haplotype table (rows in pairing order)."""
    return table.iloc[::2].reset_index(drop=True) if DIPLOID else table
