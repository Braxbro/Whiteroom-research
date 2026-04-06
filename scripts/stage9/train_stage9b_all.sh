#!/bin/bash
# Launch Stage 9b training for all 5 seeds in parallel
# 4-phase curriculum with selective freezing

set -e

# Activate PyTorch environment
source ~/pytorch_env/bin/activate

echo "=========================================="
echo "Stage 9b: 4-Phase Curriculum"
echo "Bridge frozen (1A/2A) → Encoder frozen (1B/2B)"
echo "=========================================="

cd /home/babrook/Documents/research

python3 _agent/scripts/stage9/train_stage9b_parallel.py \
    --outdir _agent/cache/runs/stage9b \
    --seeds 1,2,3,4,5 \
    --max-steps 100000 \
    --batch-size 64 \
    --curriculum-prob 0.4 \
    --balance-archetypes \
    --n-workers 16

echo ""
echo "Training complete. Checkpoints in: _agent/cache/runs/stage9b"
