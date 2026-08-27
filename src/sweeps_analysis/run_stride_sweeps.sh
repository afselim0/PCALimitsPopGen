#!/bin/bash
# Submit the two uniform-dilution sweeps (configs 11 and 12) as SLURM array jobs, each
# followed by its dependent aggregate job. The counterpart of run_all_sweeps.sh, limited
# to the strided-sampling pair so they can be (re)run without touching the other eight.
#
#   ./run_stride_sweeps.sh                # submit both
#   ./run_stride_sweeps.sh --no-submit    # render the .sbatch files only
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$REPO_ROOT/src/sweeps_analysis/configs"

source /software/bin/miniconda3/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/pca_rmt
cd "$REPO_ROOT"

for config in 11_ss1d_vary_deme_stride 12_ss2d_vary_deme_stride; do
    echo "=== submitting $config.yaml ==="
    python src/sweeps_analysis/run_sweep_analysis.py "$CONFIG_DIR/$config.yaml" --submit "$@"
done
