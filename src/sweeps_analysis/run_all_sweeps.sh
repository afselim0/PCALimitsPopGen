#!/bin/bash
# Submit every sweep-analysis config as its own SLURM array job, each followed by a
# dependent aggregate job. Mirrors src/simulations/run_all_sims.sh.
#
#   ./run_all_sweeps.sh                # submit all 10
#   ./run_all_sweeps.sh --no-submit    # render the .sbatch files only
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$REPO_ROOT/src/sweeps_analysis/configs"

source /software/bin/miniconda3/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/pca_rmt
cd "$REPO_ROOT"

for config in "$CONFIG_DIR"/*.yaml; do
    echo "=== submitting $(basename "$config") ==="
    python src/sweeps_analysis/run_sweep_analysis.py "$config" --submit "$@"
done
