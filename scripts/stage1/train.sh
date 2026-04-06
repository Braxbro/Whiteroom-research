#!/bin/bash
# Stage 1: composition + cache freezing. No attribution.
# Usage: bash train.sh [SEED] [OUTDIR]
set -e
SEED=${1:-42}
OUTDIR=${2:-_agent/cache/runs/stage1-seed${SEED}}
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research
python -m whiteroom.train \
    --steps 50000 \
    --batch-size 64 \
    --no-attribution \
    --balance-archetypes \
    --cooccurrence-damp 0.7 \
    --checkpoint-dir "$OUTDIR" \
    --log-every 500 \
    --checkpoint-every 10000 \
    --seed "$SEED"
