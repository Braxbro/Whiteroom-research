#!/bin/bash
# Benchmark sibling + model vs static freeze patterns vs full recompute.
# Usage: bash benchmark_sibling.sh [PRIMARY_DIR] [N_PAIRS]
set -e
PRIMARY_DIR=${1:-_agent/cache/runs/multiseed}
N=${2:-300}
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research
python _agent/scripts/benchmark_sibling.py \
    --primary-dir "$PRIMARY_DIR" \
    --n "$N" \
    2>&1 | tee "$PRIMARY_DIR/sibling_benchmark_log.txt"
