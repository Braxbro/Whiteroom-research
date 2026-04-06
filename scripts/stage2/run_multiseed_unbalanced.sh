#!/bin/bash
# Run 5 unbalanced stage-2 seeds in pairs (2 concurrent, then 2, then 1).
# Usage: bash run_multiseed_unbalanced.sh
set -e
SCRIPT="$(dirname "$0")/train_unbalanced.sh"
RUNDIR=_agent/cache/runs/multiseed-unbalanced
mkdir -p "$RUNDIR"

run_seed() {
    local seed=$1
    local log="$RUNDIR/stage2-seed${seed}-log.txt"
    echo "[$(date '+%H:%M:%S')] starting seed $seed" | tee -a "$log"
    bash "$SCRIPT" "$seed" 2>&1 | tee -a "$log"
    echo "[$(date '+%H:%M:%S')] seed $seed done" | tee -a "$log"
}

echo "=== pair 1: seeds 1+2 ==="
run_seed 1 &
run_seed 2 &
wait

echo "=== pair 2: seeds 3+4 ==="
run_seed 3 &
run_seed 4 &
wait

echo "=== final: seed 5 ==="
run_seed 5

echo "=== all seeds done ==="
