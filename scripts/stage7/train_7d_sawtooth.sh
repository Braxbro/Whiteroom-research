#!/bin/bash
# Stage 7d (363K): sawtooth encoder only
# A and B isolated, BIND bridges. Full structural isolation.
# Tests if sawtooth improves with more capacity than 50K version.
set -e
OUTDIR=${1:-_agent/cache/runs/stage7/7d-sawtooth}
SEEDS=${2:-1,2,3,4,5}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"
echo "=== Stage 7d (363K): sawtooth encoder ===" | tee -a "$LOG"
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
    --sawtooth-encoder \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 7d (363K) training done ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
python _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage5 \
    --seeds "$SEEDS" \
    2>&1 | tee -a "$LOG"

echo "=== 7d (363K) done ===" | tee -a "$LOG"
