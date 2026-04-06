#!/bin/bash
# Train Stage 4 curriculum fine-tuning for all 5 balanced seeds sequentially.
# Usage: bash train_stage4_all.sh [PRIMARY_DIR] [OUTDIR]
set -e
PRIMARY_DIR=${1:-_agent/cache/runs/multiseed}
OUTDIR=${2:-_agent/cache/runs/stage4}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"

for seed in 1 2 3 4 5; do
    PRIMARY="$PRIMARY_DIR/stage2-seed${seed}/checkpoint_final.pt"
    SEEDDIR="$OUTDIR/stage4-seed${seed}"

    if [ ! -f "$PRIMARY" ]; then
        echo "[seed $seed] missing checkpoint, skipping" | tee -a "$LOG"
        continue
    fi

    if [ -f "$SEEDDIR/checkpoint_final.pt" ]; then
        echo "[seed $seed] already done, skipping" | tee -a "$LOG"
        continue
    fi

    echo "" | tee -a "$LOG"
    echo "=== seed $seed ===" | tee -a "$LOG"
    python -m whiteroom.finetune_curriculum \
        --finetune-from "$PRIMARY" \
        --steps 20000 \
        --lr 1e-4 \
        --curriculum-prob 0.3 \
        --balance-archetypes \
        --cooccurrence-damp 0.7 \
        --checkpoint-dir "$SEEDDIR" \
        --log-every 500 \
        --checkpoint-every 5000 \
        --seed "$seed" \
        2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== all seeds done ===" | tee -a "$LOG"

# Run eval across all Stage 4 checkpoints
echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
python _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage4 \
    --seeds 1,2,3,4,5 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"

# Dump summary report
REPORT="$OUTDIR/summary.txt"
echo "Stage 4 Summary — $(date)" > "$REPORT"
echo "" >> "$REPORT"
bash "$(dirname "$0")/check_stage4.sh" "$OUTDIR" >> "$REPORT"
echo "" >> "$REPORT"
echo "--- Eval results ---" >> "$REPORT"
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
vals = lambda k: [d[s][k] for s in seeds]
fa_vals = [d[s]['freeze']['a_frozen']['frozen_seq_acc'] for s in seeds]
fb_vals = [d[s]['freeze']['b_frozen']['frozen_seq_acc'] for s in seeds]
pu_vals = [d[s]['property_append']['hybrid_pickup_pct'] for s in seeds]
at_vals = [d[s]['attribution']['seq_exact_match'] for s in seeds]
import statistics
print()
print(f'{\"mean\":<6} {statistics.mean(fa_vals):>9.3f} {statistics.mean(fb_vals):>9.3f} {statistics.mean(pu_vals):>9.3f} {statistics.mean(at_vals):>9.3f}')
print(f'{\"std\":<6} {statistics.stdev(fa_vals):>9.3f} {statistics.stdev(fb_vals):>9.3f} {statistics.stdev(pu_vals):>9.3f} {statistics.stdev(at_vals):>9.3f}')
" >> "$REPORT" 2>&1
echo "" >> "$REPORT"
echo "Full eval: $OUTDIR/eval_results.json" >> "$REPORT"
echo "Full log:  $LOG" >> "$REPORT"
echo "Report written to $REPORT"
