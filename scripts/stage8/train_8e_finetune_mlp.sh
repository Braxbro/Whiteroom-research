#!/bin/bash
# Stage 8e: Fine-tune MLP projection + decoder (unfrozen)

set -e
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

BASE_DIR="_agent/cache/runs"
STAGE8_DIR="$BASE_DIR/stage8"
RUN_DIR="$BASE_DIR/stage8"

echo "=== Stage 8e: MLP Projection Fine-Tuning (Unfrozen Decoder) ==="
echo "Starting: $(date)"

for seed in 1 2 3 4 5; do
  ENCODER_ORIG="$BASE_DIR/stage7/7d-sawtooth/stage5-seed${seed}/checkpoint_final.pt"
  DECODER_ORIG="$BASE_DIR/stage5/stage5-seed${seed}/checkpoint_final.pt"
  STAGE8_CKPT="$STAGE8_DIR/mlp-7d-seed${seed}_stage5-seed${seed}/checkpoint_translation.pt"
  PAIR_DIR="$RUN_DIR/8e-mlp-unfreeze-seed${seed}"

  if [ ! -f "$ENCODER_ORIG" ] || [ ! -f "$DECODER_ORIG" ] || [ ! -f "$STAGE8_CKPT" ]; then
    echo "ERROR: Missing checkpoint(s) for seed $seed"
    exit 1
  fi

  mkdir -p "$PAIR_DIR"
  echo "Launching 8e seed $seed (MLP projection, unfrozen decoder) in background..."
  python3 -m _agent.scripts.stage8.train_stage8_translation \
    --encoder-ckpt "$ENCODER_ORIG" \
    --decoder-ckpt "$DECODER_ORIG" \
    --resume-ckpt "$STAGE8_CKPT" \
    --checkpoint-dir "$PAIR_DIR" \
    --seed $seed \
    --steps 50000 \
    --batch-size 64 \
    --lr 3e-4 \
    --curriculum-prob 0.4 \
    --n-workers 3 \
    --balance-archetypes \
    --cooccurrence-damp 0.7 \
    --projection-type mlp \
    --unfreeze-decoder > "$PAIR_DIR/train.log" 2>&1 &
done

echo "================================"
echo "All 8e seeds launched in parallel!"
echo "Results in: $RUN_DIR/8e-mlp-unfreeze-seed*"
echo "================================"
echo "Waiting for all training to complete..."
wait
echo "All 8e training complete!"
echo "Finished: $(date)"
