#!/bin/bash
# Run cache freeze, attribution, and property-append evals across all multi-seed checkpoints.
# Usage: bash eval_multiseed.sh [RUNDIR] [N_SAMPLES]
set -e
RUNDIR=${1:-_agent/cache/runs/multiseed}
N=${2:-300}
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research
python _agent/scripts/eval_multiseed.py --rundir "$RUNDIR" --n "$N" 2>&1 | tee "$RUNDIR/eval_log.txt"
