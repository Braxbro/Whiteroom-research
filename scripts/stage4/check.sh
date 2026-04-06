#!/bin/bash
# Check Stage 4 curriculum fine-tuning progress across all seeds.
# Usage: bash check_stage4.sh [RUNDIR]
RUNDIR=${1:-_agent/cache/runs/stage4}

for seed in 1 2 3 4 5; do
    dir="$RUNDIR/stage4-seed${seed}"
    final="$dir/checkpoint_final.pt"

    if [ -f "$final" ]; then
        tag="DONE"
    elif ps aux | grep -q "[f]inetune_curriculum.*seed $seed\b"; then
        tag="running"
    else
        tag="stopped?"
    fi

    printf "seed %d [%-8s] " "$seed" "$tag"

    log="$dir/train_log.jsonl"
    if [ -f "$log" ]; then
        tail -1 "$log" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print(f'step {d[\"step\"]:6d} | total {d[\"total\"]:.4f} | seq {d[\"seq\"]:.4f} | curr {d[\"curr\"]:.4f}')
" 2>/dev/null || echo "(parse error)"
    else
        echo "(no log yet)"
    fi
done
