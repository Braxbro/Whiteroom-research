#!/bin/bash
# Stage 7 50K sawtooth variants coordinator: 7d → 7e
# Tests if full structural isolation (sawtooth) can handle bidirectional BIND
# where 7a failed spectacularly at both 363K and 50K scales.
#
# Usage: nohup bash train_50k_de.sh > _agent/cache/runs/stage7_50k/nohup-de.out 2>&1 &
set -e
RUNROOT=${1:-_agent/cache/runs/stage7_50k}
mkdir -p "$RUNROOT"

MASTER_LOG="$RUNROOT/50k_de.log"
echo "=== Stage 7 50K sawtooth variants ===" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7d (50K): sawtooth encoder only ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_50k_7d.sh" "$RUNROOT/50k-7d-sawtooth" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7e (50K): sawtooth + bidir-bind ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_50k_7e.sh" "$RUNROOT/50k-7e-sawtooth-bidir" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "=== Stage 7 sawtooth variants complete ===" | tee -a "$MASTER_LOG"
echo "Finished: $(date)" | tee -a "$MASTER_LOG"
