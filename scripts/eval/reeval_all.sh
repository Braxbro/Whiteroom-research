#!/bin/bash
# Re-evaluate all final checkpoints with current eval suite.
# Stage 8 uses its own wrapper; all others use eval_multiseed.py.
set -e
source /home/babrook/Documents/research/_agent/pytorch_env/bin/activate
cd /home/babrook/Documents/research

N=${1:-300}  # samples per eval, default 300

run_std() {
    local rundir=$1
    local prefix=$2
    local seeds=${3:-1,2,3,4,5}
    echo ""
    echo ">>> eval_multiseed: $rundir (prefix=$prefix seeds=$seeds)"
    python _agent/scripts/eval/eval_multiseed.py \
        --rundir "$rundir" \
        --subdir-prefix "$prefix" \
        --seeds "$seeds" \
        --n "$N" 2>&1
}

# ---------------------------------------------------------------------------
# Stage 2 / multiseed
# ---------------------------------------------------------------------------
run_std _agent/cache/runs/multiseed stage2
run_std _agent/cache/runs/multiseed-unbalanced stage2

# ---------------------------------------------------------------------------
# Stage 4 / 4b / 4c
# ---------------------------------------------------------------------------
run_std _agent/cache/runs/stage4 stage4
run_std _agent/cache/runs/stage4b stage4b
run_std _agent/cache/runs/stage4c stage4c

# ---------------------------------------------------------------------------
# Stage 5 / 5b / 5c
# ---------------------------------------------------------------------------
run_std _agent/cache/runs/stage5 stage5
run_std _agent/cache/runs/stage5b stage5b
run_std _agent/cache/runs/stage5c stage5c

# ---------------------------------------------------------------------------
# Stage 7 (363K)
# ---------------------------------------------------------------------------
run_std _agent/cache/runs/stage7/7a-bidir-bind stage5
run_std _agent/cache/runs/stage7/7b-causal-enc stage5
run_std _agent/cache/runs/stage7/7c-causal-bidir stage5
run_std _agent/cache/runs/stage7/7d-sawtooth stage5
run_std _agent/cache/runs/stage7/7e-sawtooth-bidir stage5

# ---------------------------------------------------------------------------
# Stage 7 (50K)
# ---------------------------------------------------------------------------
run_std _agent/cache/runs/stage7_50k/50k-stage5 stage5
run_std _agent/cache/runs/stage7_50k/50k-7a-bidir-bind stage5
run_std _agent/cache/runs/stage7_50k/50k-7b-causal-enc stage5
run_std _agent/cache/runs/stage7_50k/50k-7c-causal-bidir stage5
run_std _agent/cache/runs/stage7_50k/50k-7d-sawtooth stage5
run_std _agent/cache/runs/stage7_50k/50k-7e-sawtooth-bidir stage5

echo ""
echo "=== All re-evals complete ==="
echo "NOTE: Stage 8+ already on current eval suite — not re-evaluated."
