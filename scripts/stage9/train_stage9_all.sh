#!/bin/bash
# Stage 9: 3-stage model from scratch
# Encoder: block-diagonal bidirectional attention (no causality)
# Middle: adaptation layer
# Decoder: free adaptation

set -e
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

OUTDIR=${1:-_agent/cache/runs/stage9}
SEEDS=${2:-1,2,3,4,5}

mkdir -p "$OUTDIR"

LOG="$OUTDIR/run_log.txt"
echo "=== Stage 9: 3-Stage Model from Scratch ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python _agent/scripts/stage9/train_stage9_parallel.py \
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
    --block-diagonal-encoder \
    --bidirectional-blocks \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Stage 9 training done ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
