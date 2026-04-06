#!/bin/bash

# Run all 5 matched pairs: 7d-seedN + stage5-seedN

set -e

BASE_DIR="_agent/cache/runs"
STAGE7_DIR="$BASE_DIR/stage7/7d-sawtooth"
STAGE5_DIR="$BASE_DIR/stage5"
RUN_DIR="$BASE_DIR/stage8"

echo "Stage 8: Translation Layer Experiment"
echo "Running all 5 matched pairs (7d-seedN + stage5-seedN)"
echo ""

for seed in 1 2 3 4 5; do
    ENCODER_CKPT="$STAGE7_DIR/stage5-seed$seed/checkpoint_final.pt"
    DECODER_CKPT="$STAGE5_DIR/stage5-seed$seed/checkpoint_final.pt"
    PAIR_DIR="$RUN_DIR/7d-seed${seed}_stage5-seed${seed}"

    if [ ! -f "$ENCODER_CKPT" ]; then
        echo "ERROR: Encoder checkpoint not found: $ENCODER_CKPT"
        exit 1
    fi

    if [ ! -f "$DECODER_CKPT" ]; then
        echo "ERROR: Decoder checkpoint not found: $DECODER_CKPT"
        exit 1
    fi

    mkdir -p "$PAIR_DIR"
    echo "Launching seed $seed in background..."
    python3 -m _agent.scripts.stage8.train_stage8_translation \
        --encoder-ckpt "$ENCODER_CKPT" \
        --decoder-ckpt "$DECODER_CKPT" \
        --checkpoint-dir "$PAIR_DIR" \
        --seed $seed \
        --steps 50000 \
        --batch-size 64 \
        --lr 3e-4 \
        --curriculum-prob 0.4 \
        --n-workers 3 \
        --balance-archetypes \
        --cooccurrence-damp 0.7 > "$PAIR_DIR/train.log" 2>&1 &
done

echo "================================"
echo "All 5 seeds launched in parallel!"
echo "Results in: $RUN_DIR"
echo "================================"
echo "Waiting for all training to complete..."
wait
echo "All training complete!"
echo "================================"
