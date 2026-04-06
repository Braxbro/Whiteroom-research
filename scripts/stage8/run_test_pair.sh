#!/bin/bash

# Test run: 7d-seed1 + stage5-seed1

set -e

ENCODER_CKPT="_agent/cache/runs/stage7/7d-sawtooth/stage5-seed1/checkpoint_final.pt"
DECODER_CKPT="_agent/cache/runs/stage5/stage5-seed1/checkpoint_final.pt"
RUN_DIR="_agent/cache/runs/stage8/7d-seed1_stage5-seed1"

echo "Testing Stage 8 translation layer: 7d-seed1 + stage5-seed1"
echo "Encoder: $ENCODER_CKPT"
echo "Decoder: $DECODER_CKPT"
echo "Output: $RUN_DIR"
echo ""

mkdir -p "$RUN_DIR"

python3 -m _agent.scripts.stage8.train_stage8_translation \
    --encoder-ckpt "$ENCODER_CKPT" \
    --decoder-ckpt "$DECODER_CKPT" \
    --checkpoint-dir "$RUN_DIR" \
    --seed 1 \
    --steps 50000 \
    --batch-size 64 \
    --lr 3e-4 \
    --curriculum-prob 0.4 \
    --n-workers 3 \
    --balance-archetypes \
    --cooccurrence-damp 0.7 \
    --log-every 500 \
    --checkpoint-every 10000

echo ""
echo "Training complete! Results in: $RUN_DIR"
