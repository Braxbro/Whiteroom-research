#!/bin/bash
set -e
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research
python -m whiteroom.train \
    --steps 50000 \
    --batch-size 64 \
    --checkpoint-dir _agent/cache/runs/stage3-combination-holdout-50k \
    --log-every 500 \
    --checkpoint-every 10000 \
    --seed 42
