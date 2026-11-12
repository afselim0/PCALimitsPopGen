"""
Supplement: panels d-e of PhaseTransitionAfricanSplit redone on standardized genotypes
with MAF >= 0.05. The theory keeps the coalescent-time eps_a and nu at gamma = N/L, and
that is what fails here: with deep structure and a pooled-frequency filter, the
per-population noise levels of the standardized matrix are no longer tau_a/tau_bar, and
the miss grows as the sister populations become a smaller share of the sample (panel b).

a  cosine similarity vs Fst at five sample sizes N.
b  cosine similarity vs Fst at six harmonic-mean sister fractions.

Reads results/sweep_summaries/african_split_std/{curves,crossings,points,theory}.csv
Writes results/figures/supplement/StdPhaseTransitionAfricanSplit.pdf (single column)
"""
import os

import genetics_style as gs
from standardized_figure import make_figure

SUMMARY_DIR = os.path.join(gs._ROOT, "results", "sweep_summaries", "african_split_std")
SWEEPS = [("african_split_vary_N_std", r"$N$", "N"),
          ("african_split_vary_harmonic_phi_std", r"$\tilde{\phi}$", "H")]

if __name__ == "__main__":
    make_figure("StdPhaseTransitionAfricanSplit", SUMMARY_DIR, SWEEPS, r"$F_{ST}$")
