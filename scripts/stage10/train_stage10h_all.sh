#!/bin/bash
# Stage 10h: d_model=64, enc=3, dec=9, valid_weight=0.25, warmup=2000
# Stage 5-style symmetric-ish architecture but decoder-biased (3+9 vs 3+3)
set -e
OUTDIR=${1:-_agent/cache/runs/stage10/10h-d64-3enc-9dec}
SEEDS=${2:-1,2,3,4,5}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"
echo "=== Stage 10h: d=64, enc=3, dec=9, valid_weight=0.25, warmup=2000 ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python -m _agent.scripts.stage10.train_stage10_parallel \
    --outdir "$OUTDIR" \
    --seeds "$SEEDS" \
    --n-workers 16 \
    --balance-archetypes \
    --cooccurrence-damp 0.7 \
    --curriculum-prob 0.4 \
    --valid-weight 0.25 \
    --warmup-steps 2000 \
    --d-model 64 \
    --nhead 4 \
    --ffn-dim 256 \
    --max-steps 200000 \
    --enc-layers 3 \
    --dec-layers 9 \
    --log-every 500 \
    --checkpoint-every 10000 \
    --plateau-window 10 \
    --plateau-threshold 5e-5 \
    --min-phase-steps 10000 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== training done ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
python _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage10 \
    --seeds "$SEEDS" \
    2>&1 | tee -a "$LOG"

echo "=== Stage 10h done ===" | tee -a "$LOG"
