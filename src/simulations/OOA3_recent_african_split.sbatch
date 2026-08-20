#!/bin/bash
#SBATCH --job-name=OOA3_split
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --array=0-49
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=logs/OOA3_split_%A_%a.out
#SBATCH --error=logs/OOA3_split_%A_%a.err

# One task per t_split/Fst value (--array must cover 0..num_t_split-1, default 50 -> 0-49).
# Each task rebuilds the same parameter table and runs only its own row -- see run_one() in
# OOA3_recent_african_split.py. Re-running a task whose .npy already exists is a no-op, so
# a failed array index can be resubmitted alone (--array=<index>) without redoing the rest.
#
# Sample sizes: CEU=CHB=5000, YRI=YRI2=2000 haploid (N=14000 simulated), covering the
# planned phi sweep (sisters up to 2000 each, phi=0.2 at N=10000) and N sweep (others up to
# 4500 each at N=10000).
#
# Submit from the repo root: sbatch src/simulations/OOA3_recent_african_split.sbatch

mkdir -p logs
source /software/bin/miniconda3/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/pca_rmt
python src/simulations/OOA3_recent_african_split.py \
    --n-per-pop-by-pop CEU=5000,CHB=5000,YRI=2000,YRI2=2000 \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --task-index "${SLURM_ARRAY_TASK_ID}"
