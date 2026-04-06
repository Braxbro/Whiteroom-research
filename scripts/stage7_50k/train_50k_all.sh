#!/bin/bash
# Stage 7 50K scaling test coordinator
# Runs: Stage5 baseline → 7a (bidir-bind) → 7b (causal) → 7c (both)
# Each ~20 min per experiment; total ~100 min
# Model: 50K params (d_model=32, nhead=2, enc_layers=2, dec_layers=2, ffn_dim=128)
#
# Hypothesis test: does 50K bypass the "trying to cheat" zone that breaks 363K 7a?
# Expected: 50K 7a should improve over 363K 7a (forced clean symmetric learning)
#
# Usage: nohup bash train_50k_all.sh > _agent/cache/runs/stage7_50k/nohup.out 2>&1 &
set -e
RUNROOT=${1:-_agent/cache/runs/stage7_50k}
mkdir -p "$RUNROOT"

MASTER_LOG="$RUNROOT/50k_all.log"
echo "=== Stage 7 50K scaling test ===" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"
echo "Model: d_model=32, nhead=2, enc_layers=2, dec_layers=2, ffn_dim=128 (~50K params)" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- Stage 5 baseline (50K) ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_50k_stage5.sh" "$RUNROOT/50k-stage5" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7a (50K): bidir-bind only ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_50k_7a.sh" "$RUNROOT/50k-7a-bidir-bind" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7b (50K): causal-encoder only ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_50k_7b.sh" "$RUNROOT/50k-7b-causal-enc" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "--- 7c (50K): causal-encoder + bidir-bind ---" | tee -a "$MASTER_LOG"
bash "$(dirname "$0")/train_50k_7c.sh" "$RUNROOT/50k-7c-causal-bidir" \
    2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "=== Stage 7 50K scaling test complete ===" | tee -a "$MASTER_LOG"
echo "Finished: $(date)" | tee -a "$MASTER_LOG"
