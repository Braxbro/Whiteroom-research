#!/bin/bash
# Stage 5 baseline — 50K model (1/7 scale of 363K for rapid iteration)
# Model dims: d_model=32, nhead=2, enc_layers=2, dec_layers=2, ffn_dim=128
# Hypothesis test: does smaller model learn symmetric BIND better?
set -e
OUTDIR=${1:-_agent/cache/runs/stage7_50k/50k-stage5}
SEEDS=${2:-1,2,3,4,5}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"
echo "=== Stage 5 (50K model) ===" | tee -a "$LOG"
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
    --d-model 32 \
    --nhead 2 \
    --enc-layers 2 \
    --dec-layers 2 \
    --ffn-dim 128 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Stage 5 (50K) training done ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
python _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage5 \
    --seeds "$SEEDS" \
    2>&1 | tee -a "$LOG"

echo "=== Stage 5 (50K) done ===" | tee -a "$LOG"
