"""
Supplement: the stepping-stone figure (PhaseTransitionSS) redone on standardized genotypes
with MAF >= 0.05. Theory at gamma = N/L, L the number of SNPs kept.

a  1D (K = 25 demes), five values of n per deme, every deme sampled.
b  2D (9 x 9 demes), same n grid.

Reads results/sweep_summaries/stepping_stone_even_std/{curves,crossings,points,theory}.csv
Writes results/figures/supplement/StdPhaseTransitionSS.pdf (single column)
"""
import os

import genetics_style as gs
from standardized_figure import make_figure

SUMMARY_DIR = os.path.join(gs._ROOT, "results", "sweep_summaries", "stepping_stone_even_std")
SWEEPS = [("ss1d_vary_n_std", r"$n$ per deme", "n (1D)"), ("ss2d_vary_n_std", r"$n$ per deme", "n (2D)")]

if __name__ == "__main__":
    make_figure("StdPhaseTransitionSS", SUMMARY_DIR, SWEEPS, r"$1/M$")
