#!/bin/bash
# Sequential orchestration: 8d → 8e → 9
# Waits for each stage to complete before starting the next

set -e
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  STAGE 8d → 8e → 9 SEQUENTIAL PIPELINE                            ║"
echo "║  Stages run one after another to prevent GPU memory collision     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

LOG_ROOT="_agent/cache/runs"
mkdir -p "$LOG_ROOT"

# STAGE 8d: Linear Projection Fine-Tuning (Unfrozen Decoder)
echo "════════════════════════════════════════════════════════════════════"
echo "STAGE 8d: Linear Projection Fine-Tuning"
echo "════════════════════════════════════════════════════════════════════"
echo "Starting: $(date)"

bash _agent/scripts/stage8/train_8d_finetune_linear.sh
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: Stage 8d failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo "✓ Stage 8d complete"
echo ""

# STAGE 8e: MLP Projection Fine-Tuning (Unfrozen Decoder)
echo "════════════════════════════════════════════════════════════════════"
echo "STAGE 8e: MLP Projection Fine-Tuning"
echo "════════════════════════════════════════════════════════════════════"
echo "Starting: $(date)"

bash _agent/scripts/stage8/train_8e_finetune_mlp.sh
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: Stage 8e failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo "✓ Stage 8e complete"
echo ""

# STAGE 9: 3-Stage Model from Scratch
echo "════════════════════════════════════════════════════════════════════"
echo "STAGE 9: 3-Stage Model Training"
echo "════════════════════════════════════════════════════════════════════"
echo "Starting: $(date)"

bash _agent/scripts/stage9/train_stage9_all.sh
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: Stage 9 failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo "✓ Stage 9 complete"
echo ""

echo "════════════════════════════════════════════════════════════════════"
echo "ALL STAGES COMPLETE"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
