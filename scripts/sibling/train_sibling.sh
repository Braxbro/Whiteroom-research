#!/bin/bash
set -e
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research
python -m whiteroom.span_predictor train \
    --primary _agent/cache/runs/stage2-holdout-tokens-50k/checkpoint_final.pt \
    --output  _agent/cache/runs/sibling-span-predictor/sibling_final.pt \
    --n-samples 2000 \
    --steps 2000 \
    --batch-size 64 \
    --lr 3e-4 \
    --seed 42
