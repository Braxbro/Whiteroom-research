"""
Swap-order frozen cache probe (Stage 6 analysis) — memory splice variant.

Tests whether the model can correctly decode compound(A,B) from a fully frozen
cache when A and B's encoder output positions are PHYSICALLY SWAPPED in the
memory tensor.

Method:
  1. Encode [A|BIND|B] normally → mem (shape: 1, seq_len, d_model)
  2. Create mem_spliced: swap A's encoder outputs with B's encoder outputs
     in-place (BIND region kept intact). Only valid when len(A_tokens)==len(B_tokens).
  3. Decode from mem and mem_spliced; compare.

Since compound(A,B) == compound(B,A) by spec, both should produce the same
target. A degradation in spliced accuracy means the decoder relies on positional
cues in the frozen cache rather than semantic content.

Metrics:
  normal_frozen_acc:  decode from normal frozen mem (baseline)
  spliced_frozen_acc: decode from position-swapped frozen mem
  agreement_rate:     how often normal and spliced produce identical output
  splice_cost:        normal_frozen_acc - spliced_frozen_acc

Only pairs where len(A_tokens) == len(B_tokens) are included.

Usage:
    python probe_swap_order.py --checkpoints <path1> [<path2> ...]
    python probe_swap_order.py --rundir _agent/cache/runs/stage5 --subdir-prefix stage5 --seeds 1,2,3,4,5
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whiteroom.model import WhiteroomTransformer
from whiteroom.freeze_probe import (
    make_example_for_ab, make_example_for_ac,
    sample_triplet, _greedy_from_memory,
)
from whiteroom.generator import serialize_entity, serialize_compound_output, sample_primitive
from whiteroom.composition import compose, find_valid_bindings
from whiteroom.vocab import Token, port_idx_token
from whiteroom.entity import ARCHETYPES


# ---------------------------------------------------------------------------
# Pair sampling (equal-length only)
# ---------------------------------------------------------------------------

def sample_equal_length_pair(rng):
    """
    Sample (A, B, port_a_idx, port_b_idx) where len(A_tokens) == len(B_tokens).
    Required for memory splice (positions must be swappable 1-for-1).
    """
    for _ in range(500):
        a = sample_primitive(rng)
        b = sample_primitive(rng)
        if a is b:
            continue
        bindings = find_valid_bindings(a, b)
        if not bindings:
            continue
        a_tokens, _ = serialize_entity(a)
        b_tokens, _ = serialize_entity(b)
        if len(a_tokens) != len(b_tokens):
            continue
        port_a_idx, port_b_idx = rng.choice(bindings)
        return a, b, port_a_idx, port_b_idx, len(a_tokens)
    return None


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_swap(checkpoint_path: str, n: int = 300, seed: int = 42) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WhiteroomTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = random.Random(seed)

    normal_correct  = 0   # normal frozen mem → correct compound
    spliced_correct = 0   # spliced frozen mem → correct compound
    agreement       = 0   # normal and spliced produce identical output

    sampled  = 0
    attempts = 0
    skipped  = 0

    while sampled < n and attempts < n * 20:
        attempts += 1
        pair = sample_equal_length_pair(rng)
        if pair is None:
            skipped += 1
            continue
        a, b, port_a_idx, port_b_idx, entity_len = pair

        from whiteroom.freeze_probe import make_example_for_ab
        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        target = ex.target_tokens
        a_start, a_end = ex.a_token_span      # A positions in src
        b_start, b_end = ex.b_token_span      # B positions in src

        src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            mem = model.encode(src)  # (1, seq_len, d_model)

        # Memory splice: swap A and B encoder outputs (BIND region untouched)
        mem_spliced = mem.clone()
        mem_spliced[0, a_start:a_end, :] = mem[0, b_start:b_end, :]
        mem_spliced[0, b_start:b_end, :] = mem[0, a_start:a_end, :]

        pred_normal  = _greedy_from_memory(model, mem,         device, max_len=32)
        pred_spliced = _greedy_from_memory(model, mem_spliced, device, max_len=32)

        def trim(pred):
            try: return pred[:pred.index(Token.END) + 1]
            except ValueError: return pred

        p_n = trim(pred_normal)
        p_s = trim(pred_spliced)

        normal_correct  += int(p_n == target)
        spliced_correct += int(p_s == target)
        agreement       += int(p_n == p_s)

        sampled += 1

    return {
        "n": sampled,
        "equal_length_fraction": sampled / max(attempts, 1),
        "normal_frozen_acc":  normal_correct  / sampled,
        "spliced_frozen_acc": spliced_correct / sampled,
        "agreement_rate":     agreement       / sampled,
        "splice_cost":        (normal_correct - spliced_correct) / sampled,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Explicit checkpoint paths")
    parser.add_argument("--rundir", type=str, default=None,
                        help="Run directory to scan for checkpoints")
    parser.add_argument("--subdir-prefix", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed-eval", type=int, default=42)
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    checkpoints = []
    if args.checkpoints:
        checkpoints = [(Path(p).stem, p) for p in args.checkpoints]
    elif args.rundir and args.subdir_prefix:
        for s in args.seeds.split(","):
            p = Path(args.rundir) / f"{args.subdir_prefix}-seed{s}" / "checkpoint_final.pt"
            if p.exists():
                checkpoints.append((f"seed{s}", str(p)))
            else:
                print(f"[seed {s}] checkpoint not found: {p}")

    if not checkpoints:
        print("No checkpoints found.")
        sys.exit(1)

    results = {}
    print(f"\n{'Label':<15} {'norm_frz':>9} {'spliced':>9} {'agree':>7} {'cost':>7} {'eq_frac':>8}")
    print("-" * 60)

    for label, ckpt_path in checkpoints:
        print(f"{label:<15} evaluating...", flush=True)
        r = evaluate_swap(ckpt_path, n=args.n, seed=args.seed_eval)
        results[label] = r
        print(f"\r{label:<15} {r['normal_frozen_acc']:>9.3f} {r['spliced_frozen_acc']:>9.3f} "
              f"{r['agreement_rate']:>7.3f} {r['splice_cost']:>+7.3f} "
              f"{r['equal_length_fraction']:>8.3f}")

    print()

    if len(results) > 1:
        import statistics
        keys = ["normal_frozen_acc", "spliced_frozen_acc", "agreement_rate", "splice_cost"]
        print("AGGREGATE (mean ± std)")
        print("-" * 50)
        for k in keys:
            vals = [r[k] for r in results.values()]
            print(f"  {k:<25} {statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}")

    if args.outfile:
        with open(args.outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.outfile}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
