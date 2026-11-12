#!/bin/bash
# Submit every sweep-analysis config as its own SLURM array job, each followed by a
# dependent aggregate job. Mirrors src/simulations/run_all_sims.sh.
#
#   ./run_all_sweeps.sh                # submit all 17 (plus the stage-0 coalescent array)
#   ./run_all_sweeps.sh --no-submit    # render the .sbatch files only
#
# Stage 0. The OOA3 configs (03, 04) need results/analysis/coal_times/ populated: a
# ~15-20 min msprime integration per t_split, shared by both configs and by every task
# within them. It is submitted first here, and the OOA3 sweeps are chained after it with
# --dependency=afterok so they never race an incomplete cache. Everything else is
# independent of it and is submitted immediately.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$REPO_ROOT/src/sweeps_analysis/configs"

cd "$REPO_ROOT"

SUBMIT=1
for arg in "$@"; do
    [[ "$arg" == "--no-submit" ]] && SUBMIT=0
done

COAL_DEP=""
echo "=== stage 0: coalescent times (OOA3) ==="
if [[ "$SUBMIT" == "1" ]]; then
    COAL_OUT=$(python src/sweeps_analysis/render_coal_times_sbatch.py --submit)
    echo "$COAL_OUT"
    # The array's job id, from the first "submitted ... as job N" line; the aggregate is
    # already chained after it inside the renderer.
    COAL_JOB=$(echo "$COAL_OUT" | grep -m1 -oP 'as job \K\d+' || true)
    [[ -n "$COAL_JOB" ]] && COAL_DEP="$COAL_JOB"
else
    python src/sweeps_analysis/render_coal_times_sbatch.py
fi

for config in "$CONFIG_DIR"/*.yaml; do
    name=$(basename "$config")
    echo "=== submitting $name ==="
    # Only the OOA3 configs read the coalescent cache.
    if [[ "$name" == *african_split* && -n "$COAL_DEP" ]]; then
        python src/sweeps_analysis/run_sweep_analysis.py "$config" --submit \
            --after "$COAL_DEP" "$@"
    else
        python src/sweeps_analysis/run_sweep_analysis.py "$config" --submit "$@"
    fi
done
