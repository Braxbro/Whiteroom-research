#!/bin/bash
# Sequentially train full + compact siblings for each seed, then benchmark.
# Each sibling is trained on oracle data from its own seed's checkpoint.
# Usage: bash train_siblings_multiseed.sh [PRIMARY_DIR] [OUTDIR]
set -e
PRIMARY_DIR=${1:-_agent/cache/runs/multiseed}
OUTDIR=${2:-_agent/cache/runs/siblings-multiseed}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/train_log.txt"

for seed in 1 2 3 4 5; do
    PRIMARY="$PRIMARY_DIR/stage2-seed${seed}/checkpoint_final.pt"
    if [ ! -f "$PRIMARY" ]; then
        echo "[seed $seed] missing checkpoint, skipping" | tee -a "$LOG"
        continue
    fi

    echo "" | tee -a "$LOG"
    echo "=== seed $seed — full format ===" | tee -a "$LOG"
    python -m whiteroom.span_predictor train \
        --primary "$PRIMARY" \
        --output  "$OUTDIR/sibling_full_seed${seed}.pt" \
        --n-samples 2000 \
        --steps 2000 \
        --seed "$seed" \
        2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "=== seed $seed — compact format ===" | tee -a "$LOG"
    python -m whiteroom.span_predictor train \
        --primary "$PRIMARY" \
        --output  "$OUTDIR/sibling_compact_seed${seed}.pt" \
        --n-samples 2000 \
        --steps 2000 \
        --seed "$seed" \
        --compact \
        2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== all siblings trained — running per-seed benchmark ===" | tee -a "$LOG"
python _agent/scripts/benchmark_siblings_multiseed.py \
    --primary-dir "$PRIMARY_DIR" \
    --sibling-dir "$OUTDIR" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"
