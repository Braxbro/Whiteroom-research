#!/bin/bash
# Train compact sibling: input is [old_tokens | SEP | extra_flag_tok] (~14 tokens).
# Usage: bash train_sibling_compact.sh [PRIMARY_CKPT] [OUTPUT]
set -e
PRIMARY=${1:-_agent/cache/runs/multiseed/stage2-seed1/checkpoint_final.pt}
OUTPUT=${2:-_agent/cache/runs/sibling-compact/sibling_compact.pt}
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research
mkdir -p "$(dirname "$OUTPUT")"
python -m whiteroom.span_predictor train \
    --primary "$PRIMARY" \
    --output  "$OUTPUT" \
    --n-samples 2000 \
    --steps 2000 \
    --compact \
    --seed 42
