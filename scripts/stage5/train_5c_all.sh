#!/bin/bash
# Stage 5c: Frozen-encoder curriculum fine-tuning on top of Stage 5 checkpoints.
# Encoder weights fully frozen; only decoder + heads are trained.
# Goal: improve pickup without any risk of encoder geometry degradation.
#
# Usage: bash train_stage5c_all.sh [STAGE5_DIR] [OUTDIR]
set -e
STAGE5_DIR=${1:-_agent/cache/runs/stage5}
OUTDIR=${2:-_agent/cache/runs/stage5c}
N_WORKERS=3
PYTHON=/home/babrook/pytorch_env/bin/python
mkdir -p "$OUTDIR"
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"
echo "Stage 5c parallel launch — $(date)" | tee "$LOG"

for seed in 1 2 3 4 5; do
    PRIMARY="$STAGE5_DIR/stage5-seed${seed}/checkpoint_final.pt"
    SEEDDIR="$OUTDIR/stage5c-seed${seed}"

    if [ ! -f "$PRIMARY" ]; then
        echo "[seed $seed] missing stage5 checkpoint, skipping" | tee -a "$LOG"
        continue
    fi

    if [ -f "$SEEDDIR/checkpoint_final.pt" ]; then
        echo "[seed $seed] already done, skipping" | tee -a "$LOG"
        continue
    fi

    echo "[seed $seed] starting..." | tee -a "$LOG"
    $PYTHON -m whiteroom.finetune_curriculum \
        --finetune-from "$PRIMARY" \
        --steps 20000 \
        --lr 1e-4 \
        --curriculum-prob 0.5 \
        --freeze-encoder \
        --balance-archetypes \
        --cooccurrence-damp 0.7 \
        --checkpoint-dir "$SEEDDIR" \
        --log-every 500 \
        --checkpoint-every 5000 \
        --n-workers $N_WORKERS \
        --seed "$seed" \
        >> "$SEEDDIR.log" 2>&1 &
    echo "[seed $seed] PID $!" | tee -a "$LOG"
done

echo "Waiting for all seeds..." | tee -a "$LOG"
wait
echo "All seed processes finished." | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
$PYTHON _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage5c \
    --seeds 1,2,3,4,5 \
    2>&1 | tee -a "$LOG"

REPORT="$OUTDIR/summary.txt"
echo "Stage 5c Summary — $(date)" > "$REPORT"
echo "" >> "$REPORT"
$PYTHON -c "
import json, statistics
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
fa_vals = [d[s]['freeze']['a_frozen']['frozen_seq_acc'] for s in seeds]
fb_vals = [d[s]['freeze']['b_frozen']['frozen_seq_acc'] for s in seeds]
pu_vals = [d[s]['property_append']['hybrid_pickup_pct'] for s in seeds]
at_vals = [d[s]['attribution']['seq_exact_match'] for s in seeds]
print()
print(f'{\"mean\":<6} {statistics.mean(fa_vals):>9.3f} {statistics.mean(fb_vals):>9.3f} {statistics.mean(pu_vals):>9.3f} {statistics.mean(at_vals):>9.3f}')
print(f'{\"std\":<6} {statistics.stdev(fa_vals):>9.3f} {statistics.stdev(fb_vals):>9.3f} {statistics.stdev(pu_vals):>9.3f} {statistics.stdev(at_vals):>9.3f}')
" >> "$REPORT" 2>&1

echo "" >> "$REPORT"
echo "Full log: $LOG" >> "$REPORT"
echo "Report written to $REPORT"

echo "" | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"
