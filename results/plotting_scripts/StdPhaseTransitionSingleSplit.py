"""
Supplement: the single-split figure (PhaseTransitionSingleSplit) redone on standardized
genotypes with MAF >= 0.05. Theory is plain BBP at gamma = N/L, L the number of SNPs kept.

a  cosine similarity vs Fst at five sample sizes N.
b  cosine similarity vs Fst at five sampling fractions phi.

Reads results/sweep_summaries/single_split_std/{curves,crossings,points,theory}.csv
Writes results/figures/supplement/StdPhaseTransitionSingleSplit.pdf (single column)
"""
import os

import genetics_style as gs
from standardized_figure import make_figure

SUMMARY_DIR = os.path.join(gs._ROOT, "results", "sweep_summaries", "single_split_std")
SWEEPS = [("split_vary_N_std", r"$N$", "N"), ("split_vary_phi_std", r"$\phi$", "phi")]

if __name__ == "__main__":
    make_figure("StdPhaseTransitionSingleSplit", SUMMARY_DIR, SWEEPS, r"$F_{ST}$")
