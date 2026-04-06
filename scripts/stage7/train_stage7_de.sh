#!/bin/bash
# Stage 7 363K sawtooth variants coordinator: 7d → 7e
# Tests if sawtooth encoder (7d) improves with more capacity,
# and if bidirectional BIND (7e) recovers from catastrophic failure at 50K.
#
# Hypothesis: at 50K the BIND bottleneck constrains both variants.
# At 363K, more capacity to build richer bridge representations.
#
# Usage: nohup bash train_stage7_de.sh > _agent/cache/runs/stage7/nohup-de-363k.out 2>&1 &
set -e
RUNROOT=${1:-_agent/cache/runs/stage7}
mkdir -p "$RUNROOT"

MASTER_LOG="$RUNROOT/stage7_de.log"
echo "=== Stage 7 363K sawtooth variants ===" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7d (363K): sawtooth encoder only ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_7d_sawtooth.sh" "$RUNROOT/7d-sawtooth" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7e (363K): sawtooth + bidir-bind ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_7e_sawtooth_bidir.sh" "$RUNROOT/7e-sawtooth-bidir" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "=== Stage 7 363K sawtooth variants complete ===" | tee -a "$MASTER_LOG"
echo "Finished: $(date)" | tee -a "$MASTER_LOG"
