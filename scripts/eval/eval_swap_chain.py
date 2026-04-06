"""
Daisy-chain memory swap evaluation: A+B → compound_AB, compound_AB + C.

Tests whether the cross-pair swap generalises to hierarchical (depth-2)
compositions. The "A BIND B BIND C" chain:

    Stage 1: compound_AB = compose(A, B)
    Stage 2: final = compose(compound_AB, C)

Two swap conditions:

  Condition 1 — swap the LEFT component (compound_AB):
    Normal:  encode([compound_AB | BIND | C])  → decode → final
    Swap:    compound_AB segment from encode([compound_AB | BIND | D])
             C segment from encode([compound_AB | BIND | C])
             → splice → decode
    (Tests: is compound_AB's representation portable across compliant C partners?)

  Condition 2 — swap the RIGHT component (C):
    Same structure, symmetric.

  Condition 3 — cross-pair swap (both sides from independent encodings):
    compound_AB from encode([compound_AB | BIND | D])
    E from encode([compound_EF | BIND | E])   ← E is a compliant swap for C
    → splice → decode compound(compound_AB, E)
    (Tests: full portability — both sides from different encoding contexts.)

The chain-specific question is Condition 3: if you pre-computed compound_AB in
one call and C (or its equivalent E) in another, can you splice and decode?

Usage:
    python -m _agent.scripts.eval.eval_swap_chain \\
        --rundir _agent/cache/runs/stage10/10e-d32-2enc-21dec \\
        --subdir-prefix stage10 \\
        --seeds 1,2,3,4,5 \\
        --n 300
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn

from whiteroom.entity import Entity, ARCHETYPES
from whiteroom.composition import compose, find_valid_bindings
from whiteroom.generator import (
    serialize_entity, serialize_compound_output,
    sample_primitive, VOCAB_SIZE,
)
from whiteroom.model import WhiteroomTransformer
from whiteroom.vocab import Token, port_idx_token
from whiteroom.freeze_probe import make_example_for_ab, _greedy_from_memory, FLAG_TOKENS


# ---------------------------------------------------------------------------
# Chain quadruple sampling
# ---------------------------------------------------------------------------

def sample_chain_quad(
    rng: random.Random,
    max_outer: int = 500,
    max_inner: int = 100,
) -> Optional[Tuple]:
    """
    Sample (A, B, C, D, pa, pb, pc_ab, pc, pd_ab, pd) where:
        compound_AB = compose(A, B, pa, pb)
        compound_AB + C valid at (pc_ab, pc)
        compound_AB + D valid at (pd_ab, pd)   ← D is a compliant swap for C
        D differs from C in flags or op_type

    Returns None if no valid chain quad found.
    """
    for _ in range(max_outer):
        a = sample_primitive(rng)
        b = sample_primitive(rng)
        ab_bindings = find_valid_bindings(a, b)
        if not ab_bindings:
            continue
        pa, pb = rng.choice(ab_bindings)
        compound_ab = compose(a, b, pa, pb)

        for _ in range(max_inner):
            c = sample_primitive(rng)
            abc_bindings = find_valid_bindings(compound_ab, c)
            if not abc_bindings:
                continue
            pc_ab, pc = rng.choice(abc_bindings)

            # Find D — compliant swap for C (compatible with compound_AB, different from C)
            d = sample_primitive(rng)
            if d is c:
                continue
            abd_bindings = find_valid_bindings(compound_ab, d)
            if not abd_bindings:
                continue
            # D must bind at same port on compound_AB side as C
            abd_at_pc_ab = [(p_ab, p_d) for p_ab, p_d in abd_bindings if p_ab == pc_ab]
            if not abd_at_pc_ab:
                continue
            _, pd = rng.choice(abd_at_pc_ab)
            pd_ab = pc_ab

            # D must differ from C
            if d.flags == c.flags and d.op_types == c.op_types:
                continue

            return a, b, c, d, pa, pb, pc_ab, pc, pd_ab, pd

    return None


# ---------------------------------------------------------------------------
# Example builders for chain compositions
# ---------------------------------------------------------------------------

def make_chain_example(entity_left, entity_right, port_left, port_right):
    """Build an Example for compose(entity_left, entity_right)."""
    compound = compose(entity_left, entity_right, port_left, port_right)
    left_tokens, left_map = serialize_entity(entity_left)
    right_tokens, right_map = serialize_entity(entity_right)
    rel_left = left_map[port_left]
    rel_right = right_map[port_right]
    a_end = len(left_tokens)
    b_start = a_end + 3
    b_end = b_start + len(right_tokens)
    input_tokens = (left_tokens
                    + [Token.BIND, port_idx_token(rel_left), port_idx_token(rel_right)]
                    + right_tokens)
    from whiteroom.generator import Example
    return Example(
        input_tokens=input_tokens,
        target_tokens=serialize_compound_output(compound),
        is_valid=True,
        entity_a=entity_left,
        entity_b=entity_right,
        compound=compound,
        a_token_span=(0, a_end),
        b_token_span=(b_start, b_end),
    )


def encode_chain_pair(model, entity_left, entity_right, port_left, port_right, device):
    """Encode entity_left + entity_right and return (memory, a_end, b_start, b_end)."""
    ex = make_chain_example(entity_left, entity_right, port_left, port_right)
    src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        mem = model.encode(src)
    return mem, ex.a_token_span[1], ex.b_token_span[0], ex.b_token_span[1]


# ---------------------------------------------------------------------------
# Chain swap test
# ---------------------------------------------------------------------------

def run_chain_swap_test(model, a, b, c, d, pa, pb, pc_ab, pc, pd_ab, pd, device):
    """
    Run three conditions for the chain (compound_AB) + C/D:

    Condition 1: swap right component only (C → D in a different encoding context)
    Condition 2: swap left component (compound_AB from different C context) — standard
    Condition 3: full cross-pair swap (compound_AB from ABC context, C/D from separate)

    Returns dict with metrics for all three conditions.
    """
    model.eval()
    compound_ab = compose(a, b, pa, pb)

    # Encode all needed pairs
    mem_abc, ab_end_abc, c_start, c_end = encode_chain_pair(
        model, compound_ab, c, pc_ab, pc, device)
    mem_abd, ab_end_abd, d_start, d_end = encode_chain_pair(
        model, compound_ab, d, pd_ab, pd, device)

    # Fresh reference encodings
    mem_abc_fresh, _, _, _ = encode_chain_pair(
        model, compound_ab, c, pc_ab, pc, device)
    mem_abd_fresh, _, _, _ = encode_chain_pair(
        model, compound_ab, d, pd_ab, pd, device)

    target_abc = make_chain_example(compound_ab, c, pc_ab, pc).target_tokens
    target_abd = make_chain_example(compound_ab, d, pd_ab, pd).target_tokens

    def decode_check(mem, target):
        pred = _greedy_from_memory(model, mem, device, max_len=32)
        try:
            pred = pred[:pred.index(Token.END) + 1]
        except ValueError:
            pass
        seq_ok = (pred == target)
        pred_flags = {t for t in pred if t in FLAG_TOKENS}
        tgt_flags  = {t for t in target if t in FLAG_TOKENS}
        return seq_ok, pred_flags == tgt_flags

    def cos(m, s, e, m2, s2, e2):
        v1 = m[0, s:e, :].mean(0)
        v2 = m2[0, s2:e2, :].mean(0)
        return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

    # --- Condition 1: fresh ABC vs swap right (C from ABD encoding, splice into ABC) ---
    # Use compound_AB from mem_abc, C positions from mem_abc (fresh) = just fresh
    # Actually swap right: take compound_AB from mem_abd, C from mem_abc
    hybrid_swap_right = mem_abc.clone()
    hybrid_swap_right[0, :ab_end_abc, :] = mem_abd[0, :ab_end_abd, :]
    # Note: ab_end should be same length since compound_ab tokens are same
    swap_right_seq, swap_right_flag = decode_check(hybrid_swap_right, target_abc)
    fresh_abc_seq,  fresh_abc_flag  = decode_check(mem_abc, target_abc)

    ab_cos_right = cos(mem_abc, 0, ab_end_abc, mem_abd, 0, ab_end_abd)

    # --- Condition 2: full cross-swap ---
    # compound_AB from ABC encoding context, D side from ABD encoding context
    # Use mem_abd as base (has correct BIND and D tokens), replace compound_AB segment
    # with compound_AB from mem_abc. Both ab_end values are same entity so same length.
    hybrid_full_cross = mem_abd.clone()
    hybrid_full_cross[0, :ab_end_abd, :] = mem_abc[0, :ab_end_abc, :]
    swap_full_seq, swap_full_flag = decode_check(hybrid_full_cross, target_abd)
    fresh_abd_seq, fresh_abd_flag = decode_check(mem_abd, target_abd)

    c_cos = cos(mem_abc, c_start, c_end, mem_abd, d_start, d_end)
    ab_cos_full = cos(mem_abc, 0, ab_end_abc, mem_abd, 0, ab_end_abd)

    return {
        # compound_AB representation consistency across C vs D partners
        "ab_cos_across_partners": ab_cos_right,
        # Swap right: use compound_AB from ABD context into ABC decoding
        "fresh_abc_seq":     fresh_abc_seq,
        "swap_right_seq":    swap_right_seq,
        "swap_right_flag":   swap_right_flag,
        # Full cross swap: compound_AB from ABC, D-side from ABD
        "fresh_abd_seq":     fresh_abd_seq,
        "swap_full_seq":     swap_full_seq,
        "swap_full_flag":    swap_full_flag,
        "ab_cos_full":       ab_cos_full,
        "cd_cos":            c_cos,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_chain_experiment(checkpoint_path: str, n: int = 300, seed: int = 42) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"].copy()
    model_type = config.pop("model_type", "2stage")
    for key in ("sawtooth_encoder",):
        config.pop(key, None)
    if model_type == "3stage":
        from whiteroom.model import WhiteroomTransformer3Stage
        model = WhiteroomTransformer3Stage(**config).to(device)
    else:
        model = WhiteroomTransformer(**config).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = random.Random(seed)
    results = []
    n_failed = 0

    for _ in range(n):
        quad = sample_chain_quad(rng)
        if quad is None:
            n_failed += 1
            continue
        a, b, c, d, pa, pb, pc_ab, pc, pd_ab, pd = quad
        r = run_chain_swap_test(model, a, b, c, d, pa, pb, pc_ab, pc, pd_ab, pd, device)
        results.append(r)

    n_res = len(results)
    if n_res == 0:
        return {"n_failed": n_failed}

    def mean(key):
        return sum(r[key] for r in results) / n_res

    fresh_abc = mean("fresh_abc_seq")
    swap_right = mean("swap_right_seq")
    fresh_abd = mean("fresh_abd_seq")
    swap_full = mean("swap_full_seq")

    return {
        "n": n_res,
        "n_failed": n_failed,
        # compound_AB representation consistency (cos sim across different C/D partners)
        "ab_cos_across_partners": mean("ab_cos_across_partners"),
        # Swap right: compound_AB from different context → same target
        "fresh_abc_seq_acc":   fresh_abc,
        "swap_right_seq_acc":  swap_right,
        "swap_right_vs_fresh": swap_right / fresh_abc if fresh_abc > 0 else 0.0,
        # Full cross swap: both sides from different contexts
        "fresh_abd_seq_acc":   fresh_abd,
        "swap_full_seq_acc":   swap_full,
        "swap_full_vs_fresh":  swap_full / fresh_abd if fresh_abd > 0 else 0.0,
        "ab_cos_full":         mean("ab_cos_full"),
        "cd_cos":              mean("cd_cos"),
    }


def run_multiseed(rundir: str, subdir_prefix: str, seeds: str,
                  n: int = 300, seed_eval: int = 42):
    import statistics
    rundir = Path(rundir)
    seed_list = [int(s) for s in seeds.split(",")]
    all_results = {}

    for seed in seed_list:
        ckpt = rundir / f"{subdir_prefix}-seed{seed}" / "checkpoint_final.pt"
        if not ckpt.exists():
            print(f"[seed {seed}] not found, skipping")
            continue
        print(f"\n[seed {seed}] chain swap eval ...")
        r = run_chain_experiment(str(ckpt), n=n, seed=seed_eval)
        all_results[str(seed)] = r
        print(f"  ab_cos_across_partners: {r['ab_cos_across_partners']:.4f}")
        print(f"  swap_right: fresh={r['fresh_abc_seq_acc']:.4f} swap={r['swap_right_seq_acc']:.4f} ratio={r['swap_right_vs_fresh']:.4f}")
        print(f"  swap_full:  fresh={r['fresh_abd_seq_acc']:.4f} swap={r['swap_full_seq_acc']:.4f} ratio={r['swap_full_vs_fresh']:.4f}")
        print(f"  ab_cos_full={r['ab_cos_full']:.4f}  cd_cos={r['cd_cos']:.4f}  failed={r['n_failed']}")

    if all_results:
        print("\n" + "=" * 70)
        print("AGGREGATE")
        print("=" * 70)
        keys = ["swap_right_vs_fresh", "swap_full_vs_fresh",
                "ab_cos_across_partners", "ab_cos_full", "cd_cos",
                "fresh_abc_seq_acc", "swap_right_seq_acc",
                "fresh_abd_seq_acc", "swap_full_seq_acc"]
        for k in keys:
            vals = [v[k] for v in all_results.values() if k in v]
            if vals:
                mu = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"  {k:35s}: {mu:.4f} ± {sd:.4f}")

    out_path = rundir / "eval_swap_chain_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daisy-chain memory swap evaluation")
    parser.add_argument("--rundir",        type=str, default=None)
    parser.add_argument("--subdir-prefix", type=str, default=None)
    parser.add_argument("--seeds",         type=str, default="1,2,3,4,5")
    parser.add_argument("--n",             type=int, default=300)
    parser.add_argument("--seed-eval",     type=int, default=42)
    parser.add_argument("--checkpoint",    type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint:
        r = run_chain_experiment(args.checkpoint, n=args.n, seed=args.seed_eval)
        print(json.dumps(r, indent=2))
    else:
        if not args.rundir or not args.subdir_prefix:
            parser.error("--rundir and --subdir-prefix required unless --checkpoint is used")
        run_multiseed(args.rundir, args.subdir_prefix, args.seeds,
                      n=args.n, seed_eval=args.seed_eval)
