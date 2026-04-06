#!/bin/bash
# Stage 4b: Two-phase curriculum fine-tuning.
#   Phase 1 (partial freeze): each span frozen independently at 50% probability.
#              Teaches general frozen-context integration, loosens entangled representations.
#   Phase 2 (full freeze):    all of A+BIND+B frozen, extra_flag live.
#              Refines toward the target eval condition on a better-prepared decoder.
#
# Usage: bash train_stage4b_all.sh [PRIMARY_DIR] [OUTDIR]
#   PRIMARY_DIR: directory containing stage2-seed{N}/checkpoint_final.pt
#                (can also point to stage4 dir for chaining from Stage 4)
#   OUTDIR:      output directory for stage4b checkpoints
set -e
PRIMARY_DIR=${1:-_agent/cache/runs/multiseed}
OUTDIR=${2:-_agent/cache/runs/stage4b}
mkdir -p "$OUTDIR"
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

LOG="$OUTDIR/run_log.txt"

for seed in 1 2 3 4 5; do
    PRIMARY="$PRIMARY_DIR/stage2-seed${seed}/checkpoint_final.pt"
    PHASE1DIR="$OUTDIR/stage4b-phase1-seed${seed}"
    PHASE2DIR="$OUTDIR/stage4b-seed${seed}"

    if [ ! -f "$PRIMARY" ]; then
        echo "[seed $seed] missing checkpoint, skipping" | tee -a "$LOG"
        continue
    fi

    if [ -f "$PHASE2DIR/checkpoint_final.pt" ]; then
        echo "[seed $seed] already done, skipping" | tee -a "$LOG"
        continue
    fi

    echo "" | tee -a "$LOG"
    echo "=== seed $seed — phase 1 (partial freeze, 10k steps) ===" | tee -a "$LOG"
    python -m whiteroom.finetune_curriculum \
        --finetune-from "$PRIMARY" \
        --steps 10000 \
        --lr 1e-4 \
        --curriculum-prob 0.5 \
        --partial-freeze \
        --balance-archetypes \
        --cooccurrence-damp 0.7 \
        --checkpoint-dir "$PHASE1DIR" \
        --log-every 500 \
        --checkpoint-every 5000 \
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

echo "" | tee -a "$LOG"
echo "=== running eval ===" | tee -a "$LOG"
python _agent/scripts/eval/eval_multiseed.py \
    --rundir "$OUTDIR" \
    --subdir-prefix stage4b \
    --seeds 1,2,3,4,5 \
    2>&1 | tee -a "$LOG"

# Summary report
REPORT="$OUTDIR/summary.txt"
echo "Stage 4b Summary — $(date)" > "$REPORT"
echo "" >> "$REPORT"
python3 -c "
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
