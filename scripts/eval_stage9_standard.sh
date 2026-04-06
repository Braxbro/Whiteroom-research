#!/bin/bash
# Evaluate Stage 9 using standard eval_multiseed.py with patched freeze_probe.py
# This ensures Stage 9 results use the exact same evaluation pipeline as other models

set -e

# Activate PyTorch environment
source ~/pytorch_env/bin/activate

echo "=========================================="
echo "Stage 9 Evaluation (Standard Test Suite)"
echo "=========================================="

python3 _agent/scripts/eval/eval_multiseed.py \
    --rundir _agent/cache/runs/stage9 \
    --subdir-prefix stage9 \
    --seeds 1,2,3,4,5 \
    --n 300 \
    --seed-eval 1234

echo ""
echo "Results saved to: _agent/cache/runs/stage9/eval_results.json"
echo "=========================================="
