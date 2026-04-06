#!/bin/bash
# Stage 2: unbalanced (no archetype balancing, no co-occurrence dampening).
# Usage: bash train_unbalanced.sh [SEED] [OUTDIR]
set -e
SEED=${1:-42}
OUTDIR=${2:-_agent/cache/runs/multiseed-unbalanced/stage2-seed${SEED}}
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research
python -m whiteroom.train \
    --steps 50000 \
    --batch-size 64 \
    --checkpoint-dir "$OUTDIR" \
    --log-every 500 \
    --checkpoint-every 10000 \
    --seed "$SEED"
