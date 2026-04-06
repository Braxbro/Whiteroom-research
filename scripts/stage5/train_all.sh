#!/bin/bash
# Stage 5: Train from scratch with adaptive freeze curriculum for all seeds.
# Usage: bash train_stage5_all.sh [OUTDIR] [SEEDS]
set -e
OUTDIR=${1:-_agent/cache/runs/stage5}
SEEDS=${2:-1,2,3,4,5}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do
    SEEDDIR="$OUTDIR/stage5-seed${seed}"

    if [ -f "$SEEDDIR/checkpoint_final.pt" ]; then
        echo "[seed $seed] already done, skipping" | tee -a "$LOG"
        continue
    fi

    echo "" | tee -a "$LOG"
    echo "=== seed $seed ===" | tee -a "$LOG"
    python _agent/scripts/stage5/train_stage5.py \
        --checkpoint-dir "$SEEDDIR" \
        --seed "$seed" \
        --balance-archetypes \
        --cooccurrence-damp 0.7 \
        --curriculum-prob 0.4 \
        --max-steps 200000 \
        --log-every 500 \
        --checkpoint-every 10000 \
        --plateau-window 10 \
        --plateau-threshold 5e-5 \
        --min-phase-steps 10000 \
        --n-workers 16 \
        2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== all seeds done ===" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
python _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage5 \
    --seeds "$SEEDS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"
