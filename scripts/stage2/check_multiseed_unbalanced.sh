#!/bin/bash
# Check unbalanced multi-seed training progress.
# Usage: bash check_multiseed_unbalanced.sh [RUNDIR]
RUNDIR=${1:-_agent/cache/runs/multiseed-unbalanced}

for seed in 1 2 3 4 5; do
    dir="$RUNDIR/stage2-seed${seed}"
    log="$dir/train_log.jsonl"
    final="$dir/checkpoint_final.pt"

    if [ -f "$final" ]; then
        tag="DONE"
    elif ps aux | grep -q "[w]hiteroom.train.*seed $seed\b"; then
        tag="running"
    else
        tag="stopped?"
    fi

    printf "seed %d [%-8s] " "$seed" "$tag"

    if [ -f "$log" ]; then
        tail -1 "$log" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print(f'step {d[\"step\"]:6d} | loss {d[\"total\"]:.4f} | seq {d[\"seq\"]:.4f} | attr {d[\"attr\"]:.4f}')
" 2>/dev/null || echo "(parse error)"
    else
        echo "(no log yet)"
    fi
done
