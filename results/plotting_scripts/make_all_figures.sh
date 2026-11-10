#!/bin/bash
# Draw every manuscript figure into results/figures/{main,supplement} from the files in
# results/analysis and results/sweep_summaries. Run inside the conda env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for fig in RMTPhenomena PhaseTransitionSingleSplit PhaseTransitionAfricanSplit DemoSS \
           PhaseTransitionSS PhaseTransitionRecentSS EffectiveL EigenvalueAfricanSplit \
           EigenvalueSS DemoRandomSS SparseSamplingCeiling DemoRecentSS; do
    echo "=== $fig ==="
    python "$HERE/$fig.py"
done
