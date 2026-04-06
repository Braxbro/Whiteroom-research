#!/bin/bash
# Stage 4: Curriculum fine-tuning for frozen-context decoder tolerance.
# Fine-tunes a Stage 2 checkpoint with mixed frozen-span + normal batches.
# Usage: bash train.sh [SEED] [FINETUNE_FROM] [OUTDIR]
set -e
SEED=${1:-1}
FINETUNE_FROM=${2:-_agent/cache/runs/multiseed/stage2-seed${SEED}/checkpoint_final.pt}
OUTDIR=${3:-_agent/cache/runs/stage4/stage4-seed${SEED}}
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research
python -m whiteroom.finetune_curriculum \
    --finetune-from "$FINETUNE_FROM" \
    --steps 20000 \
    --lr 1e-4 \
    --curriculum-prob 0.3 \
    --balance-archetypes \
    --cooccurrence-damp 0.7 \
    --checkpoint-dir "$OUTDIR" \
    --log-every 500 \
    --checkpoint-every 5000 \
    --seed "$SEED"
