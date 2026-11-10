#!/bin/bash
# Submit a SLURM array job for every config in src/simulations/configs/ (run inside the
# conda env).
# Each config is rendered and submitted via run_simulation.py --submit (see sbatch.py),
# which auto-sizes --array from the config's own parameter table and probes mem/time/cpus
# unless the config's cluster: block already gives them.
#
# Usage: src/simulations/run_all_sims.sh [--no-submit]
#   --no-submit: render the .sbatch files under logs/ without calling sbatch

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG_DIR="$HERE/configs"

cd "$REPO_ROOT"
for config in "$CONFIG_DIR"/*.yaml; do
    echo "=== submitting $(basename "$config") ==="
    python src/simulations/run_simulation.py "$config" --submit "$@"
done
