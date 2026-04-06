#!/bin/bash
# Stage 7 coordinator: run 7a → 7b → 7c sequentially.
# Designed to be launched under nohup:
#   nohup bash _agent/scripts/stage7/train_stage7_all.sh > _agent/cache/runs/stage7/nohup.out 2>&1 &
#
# Each experiment runs ~75 min; total ~3.5-4 hours.
# 7a: bidir-bind only
# 7b: causal-encoder only
# 7c: causal-encoder + bidir-bind
set -e
RUNROOT=${1:-_agent/cache/runs/stage7}
mkdir -p "$RUNROOT"

MASTER_LOG="$RUNROOT/stage7_all.log"
echo "=== Stage 7 all experiments ===" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7a: bidir-bind ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_7a_bidir_bind.sh" "$RUNROOT/7a-bidir-bind" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7b: causal-encoder ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_7b_causal_enc.sh" "$RUNROOT/7b-causal-enc" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7c: causal-encoder + bidir-bind ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_7c_causal_bidir.sh" "$RUNROOT/7c-causal-bidir" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "=== Stage 7 all experiments complete ===" | tee -a "$MASTER_LOG"
echo "Finished: $(date)" | tee -a "$MASTER_LOG"
