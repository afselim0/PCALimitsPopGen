#!/bin/bash
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --job-name=rs_spectrum
#SBATCH --time=00:10:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_runs/recent_spectrum/spectrum_%j.out
#SBATCH --error=slurm_runs/recent_spectrum/spectrum_%j.err

source /software/bin/miniconda3/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/pca_rmt
cd /project/jnovembre/afselim/PCA_RMT
python src/analysis/RecentSpatialSpectrum.py
