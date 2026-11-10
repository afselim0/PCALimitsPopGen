#!/bin/bash
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --job-name=rs_particip
#SBATCH --time=00:15:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_runs/recent_spectrum/particip_%j.out
#SBATCH --error=slurm_runs/recent_spectrum/particip_%j.err

source /software/bin/miniconda3/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/pca_rmt
cd /project/jnovembre/afselim/PCA_RMT
python src/analysis/RecentSpatialParticipation.py
