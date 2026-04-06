#!/bin/bash
# Stage 7c: causal (unidirectional) encoder + bidirectional BIND
# Both interventions combined: A is structurally isolated AND BIND is symmetric
# 5 seeds in parallel via shared data server
set -e
OUTDIR=${1:-_agent/cache/runs/stage7/7c-causal-bidir}
SEEDS=${2:-1,2,3,4,5}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"
echo "=== Stage 7c: causal-encoder + bidir-bind ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python _agent/scripts/stage5/train_stage5_parallel.py \
    --outdir "$OUTDIR" \
    --seeds "$SEEDS" \
    --n-workers 16 \
    --balance-archetypes \
    --cooccurrence-damp 0.7 \
    --curriculum-prob 0.4 \
    --max-steps 200000 \
    --log-every 500 \
    --checkpoint-every 10000 \
    --plateau-window 10 \
    --plateau-threshold 5e-5 \
    --min-phase-steps 10000 \
    --bidir-bind \
    --causal-encoder \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 7c training done ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
python _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage5 \
    --seeds "$SEEDS" \
    2>&1 | tee -a "$LOG"

echo "=== 7c done ===" | tee -a "$LOG"
