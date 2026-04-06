#!/bin/bash
# Stage 4c: Extended partial freeze curriculum for stubborn seeds.
# Phase 1 (partial freeze, 30k steps): long disentanglement pass.
# Phase 2 (full freeze, 10k steps):   refinement toward eval condition.
#
# Usage: bash train_stage4c_stubbornseeds.sh [PRIMARY_DIR] [OUTDIR] [SEEDS]
#   SEEDS: comma-separated, default "2,3"
set -e
PRIMARY_DIR=${1:-_agent/cache/runs/multiseed}
OUTDIR=${2:-_agent/cache/runs/stage4c}
SEEDS=${3:-2,3}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do
    PRIMARY="$PRIMARY_DIR/stage2-seed${seed}/checkpoint_final.pt"
    PHASE1DIR="$OUTDIR/stage4c-phase1-seed${seed}"
    PHASE2DIR="$OUTDIR/stage4c-seed${seed}"

    if [ ! -f "$PRIMARY" ]; then
        echo "[seed $seed] missing checkpoint, skipping" | tee -a "$LOG"
        continue
    fi

    if [ -f "$PHASE2DIR/checkpoint_final.pt" ]; then
        echo "[seed $seed] already done, skipping" | tee -a "$LOG"
        continue
    fi

    echo "" | tee -a "$LOG"
    echo "=== seed $seed — phase 1 (partial freeze, 30k steps) ===" | tee -a "$LOG"
    python -m whiteroom.finetune_curriculum \
        --finetune-from "$PRIMARY" \
        --steps 30000 \
        --lr 1e-4 \
        --curriculum-prob 0.5 \
        --partial-freeze \
        --balance-archetypes \
        --cooccurrence-damp 0.7 \
        --checkpoint-dir "$PHASE1DIR" \
        --log-every 500 \
        --checkpoint-every 10000 \
        --seed "$seed" \
        2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "=== seed $seed — phase 2 (full freeze, 10k steps) ===" | tee -a "$LOG"
    python -m whiteroom.finetune_curriculum \
        --finetune-from "$PHASE1DIR/checkpoint_final.pt" \
        --steps 10000 \
        --lr 5e-5 \
        --curriculum-prob 0.5 \
        --balance-archetypes \
        --cooccurrence-damp 0.7 \
        --checkpoint-dir "$PHASE2DIR" \
        --log-every 500 \
        --checkpoint-every 5000 \
        --seed "$seed" \
        2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== all seeds done ===" | tee -a "$LOG"

# Eval only the seeds we trained
echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
python _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage4c \
    --seeds "$SEEDS" \
    2>&1 | tee -a "$LOG"

# Summary
REPORT="$OUTDIR/summary.txt"
echo "Stage 4c Summary — $(date)" > "$REPORT"
echo "Seeds: $SEEDS  (30k partial freeze + 10k full freeze)" >> "$REPORT"
echo "" >> "$REPORT"
python3 -c "
import json, sys
with open('$OUTDIR/eval_results.json') as f:
    d = json.load(f)
seeds = sorted(d.keys())
print(f'{\"Seed\":<6} {\"freeze_a\":>9} {\"freeze_b\":>9} {\"pickup\":>9} {\"attr\":>9}')
print('-' * 46)
for s in seeds:
    r = d[s]
    fa = r['freeze']['a_frozen']['frozen_seq_acc']
    fb = r['freeze']['b_frozen']['frozen_seq_acc']
    pu = r['property_append']['hybrid_pickup_pct']
    at = r['attribution']['seq_exact_match']
    print(f'{s:<6} {fa:>9.3f} {fb:>9.3f} {pu:>9.3f} {at:>9.3f}')
" >> "$REPORT" 2>&1
echo "" >> "$REPORT"
echo "Full log: $LOG" >> "$REPORT"
echo "Report written to $REPORT"

echo "" | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"
