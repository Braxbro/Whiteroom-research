"""
Zero-shot two-BIND evaluation: A BIND B BIND C → compound(compound(A,B), C)

The model was trained exclusively on single-BIND sequences:
    [A_tokens | BIND(pa,pb) | B_tokens] → compound(A,B)

This test feeds it a two-BIND sequence with NO masking:
    [A_tokens | BIND(pa,pb) | B_tokens | BIND(pab,pc) | C_tokens]

and asks whether the decoder produces the correct output for compound(compound(A,B), C).

Pure zero-shot: no fine-tuning, no masking changes, just raw inference.

Three conditions:
    fresh_chain    — encode the full two-BIND sequence, decode
    fresh_flat     — encode the flattened compound_AB + C (single BIND, trained format)
                     as a sanity check baseline
    swap_chain     — encode A+B in one call, C separately, splice memories + two BINDs
                     (tests whether the swap works even with the novel format)

Metrics:
    seq_acc        — exact sequence match against compound(compound(A,B), C)
    flag_acc       — flag set match
    partial_acc    — at least all flags correct (looser measure)

Usage:
    python -m _agent.scripts.eval.eval_twobin_zeroshot \\
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

from whiteroom.entity import Entity
from whiteroom.composition import compose, find_valid_bindings
from whiteroom.generator import (
    serialize_entity, serialize_compound_output,
    sample_primitive, VOCAB_SIZE,
)
from whiteroom.model import WhiteroomTransformer
from whiteroom.vocab import Token, port_idx_token
from whiteroom.freeze_probe import _greedy_from_memory, FLAG_TOKENS
from whiteroom.generator import Example


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_abc_triple(
    rng: random.Random,
    max_attempts: int = 500,
) -> Optional[Tuple]:
    """
    Sample (A, B, C, pa, pb, pab, pc) where:
        compound_AB = compose(A, B, pa, pb)
        final = compose(compound_AB, C, pab, pc)
    """
    for _ in range(max_attempts):
        a = sample_primitive(rng)
        b = sample_primitive(rng)
        ab_bindings = find_valid_bindings(a, b)
        if not ab_bindings:
            continue
        pa, pb = rng.choice(ab_bindings)
        compound_ab = compose(a, b, pa, pb)

        c = sample_primitive(rng)
        abc_bindings = find_valid_bindings(compound_ab, c)
        if not abc_bindings:
            continue
        pab, pc = rng.choice(abc_bindings)
        final = compose(compound_ab, c, pab, pc)

        return a, b, c, pa, pb, pab, pc, compound_ab, final

    return None


# ---------------------------------------------------------------------------
# Sequence builders
# ---------------------------------------------------------------------------

def build_twobin_tokens(a, b, c, pa, pb, pab, pc):
    """
    Build the two-BIND input sequence:
        [A_tokens | BIND rel_pa rel_pb | B_tokens | BIND rel_pab rel_pc | C_tokens]
    """
    a_tokens, a_map = serialize_entity(a)
    b_tokens, b_map = serialize_entity(b)
    c_tokens, c_map = serialize_entity(c)

    rel_pa  = a_map[pa]
    rel_pb  = b_map[pb]

    # pab is a port index on compound_ab — need compound_ab's port map
    compound_ab = compose(a, b, pa, pb)
    _, ab_map = serialize_entity(compound_ab)
    rel_pab = ab_map[pab]
    rel_pc  = c_map[pc]

    tokens = (
        a_tokens
        + [Token.BIND, port_idx_token(rel_pa), port_idx_token(rel_pb)]
        + b_tokens
        + [Token.BIND, port_idx_token(rel_pab), port_idx_token(rel_pc)]
        + c_tokens
    )

    # Compute spans
    a_end    = len(a_tokens)
    b_start  = a_end + 3
    b_end    = b_start + len(b_tokens)
    c_start  = b_end + 3
    c_end    = c_start + len(c_tokens)

    return tokens, a_end, b_start, b_end, c_start, c_end


def build_flat_tokens(compound_ab, c, pab, pc):
    """
    Build the single-BIND sequence (trained format):
        [compound_AB_tokens | BIND rel_pab rel_pc | C_tokens]
    """
    ab_tokens, ab_map = serialize_entity(compound_ab)
    c_tokens, c_map = serialize_entity(c)
    rel_pab = ab_map[pab]
    rel_pc  = c_map[pc]

    tokens = (
        ab_tokens
        + [Token.BIND, port_idx_token(rel_pab), port_idx_token(rel_pc)]
        + c_tokens
    )
    return tokens


# ---------------------------------------------------------------------------
# Single triple evaluation
# ---------------------------------------------------------------------------

def run_twobin_test(model, a, b, c, pa, pb, pab, pc, compound_ab, final, device):
    model.eval()

    target = serialize_compound_output(final)

    def decode_check(mem):
        pred = _greedy_from_memory(model, mem, device, max_len=32)
        try:
            pred_trimmed = pred[:pred.index(Token.END) + 1]
        except ValueError:
            pred_trimmed = pred
        seq_ok   = (pred_trimmed == target)
        pred_flags = {t for t in pred if t in FLAG_TOKENS}
        tgt_flags  = {t for t in target if t in FLAG_TOKENS}
        flag_ok  = (pred_flags == tgt_flags)
        # partial: got all expected flags (may have extras)
        partial  = tgt_flags.issubset(pred_flags)
        return seq_ok, flag_ok, partial

    # --- Condition 1: fresh two-BIND (zero-shot, no masking) ---
    twobin_tokens, a_end, b_start, b_end, c_start, c_end = build_twobin_tokens(
        a, b, c, pa, pb, pab, pc)
    src_twobin = torch.tensor(twobin_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        mem_twobin = model.encode(src_twobin)
    zeroshot_seq, zeroshot_flag, zeroshot_partial = decode_check(mem_twobin)

    # --- Condition 2: flat single-BIND (trained format baseline) ---
    flat_tokens = build_flat_tokens(compound_ab, c, pab, pc)
    src_flat = torch.tensor(flat_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        mem_flat = model.encode(src_flat)
    flat_seq, flat_flag, flat_partial = decode_check(mem_flat)

    # --- Condition 3: swap — encode A+B separately, C separately, splice into two-BIND ---
    # Encode A+B in single-BIND format
    ab_tokens_only, ab_map = serialize_entity(compound_ab)
    c_tokens_only, c_map   = serialize_entity(c)
    rel_pab = ab_map[pab]
    rel_pc  = c_map[pc]

    # encode compound_AB alone (as a primitive-style entity with another dummy C)
    # actually just use the flat format for compound_AB encoding
    src_flat2 = torch.tensor(flat_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        mem_flat2 = model.encode(src_flat2)
    ab_end_flat = len(ab_tokens_only)
    c_start_flat = ab_end_flat + 3
    c_end_flat   = c_start_flat + len(c_tokens_only)

    # For swap_chain: take compound_AB segment from mem_twobin (positions 0:b_end which
    # includes A+BIND1+B), and C segment from mem_flat (c_start_flat:c_end_flat)
    # Reconstruct into two-BIND memory shape using mem_twobin as base, swap C from flat
    swap_chain_mem = mem_twobin.clone()
    # Replace C segment in two-BIND memory with C's representation from flat encoding
    c_len = c_end - c_start
    c_len_flat = c_end_flat - c_start_flat
    if c_len == c_len_flat:  # same entity, same token length — should always be true
        swap_chain_mem[0, c_start:c_end, :] = mem_flat2[0, c_start_flat:c_end_flat, :]
    swap_chain_seq, swap_chain_flag, swap_chain_partial = decode_check(swap_chain_mem)

    # Cosine sim: is compound_AB's two-BIND representation similar to flat representation?
    ab_vec_twobin = mem_twobin[0, :b_end, :].mean(0)  # A+BIND1+B segment
    ab_vec_flat   = mem_flat[0, :ab_end_flat, :].mean(0)  # compound_AB flat
    ab_cos = torch.nn.functional.cosine_similarity(
        ab_vec_twobin.unsqueeze(0), ab_vec_flat.unsqueeze(0)).item()

    return {
        "zeroshot_seq":     zeroshot_seq,
        "zeroshot_flag":    zeroshot_flag,
        "zeroshot_partial": zeroshot_partial,
        "flat_seq":         flat_seq,
        "flat_flag":        flat_flag,
        "flat_partial":     flat_partial,
        "swap_chain_seq":   swap_chain_seq,
        "swap_chain_flag":  swap_chain_flag,
        "swap_chain_partial": swap_chain_partial,
        "ab_cos_twobin_vs_flat": ab_cos,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_twobin_experiment(checkpoint_path: str, n: int = 300, seed: int = 42) -> dict:
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
        triple = sample_abc_triple(rng)
        if triple is None:
            n_failed += 1
            continue
        a, b, c, pa, pb, pab, pc, compound_ab, final = triple
        r = run_twobin_test(model, a, b, c, pa, pb, pab, pc, compound_ab, final, device)
        results.append(r)

    n_res = len(results)
    if n_res == 0:
        return {"n_failed": n_failed}

    def mean(key):
        return sum(r[key] for r in results) / n_res

    flat_seq = mean("flat_seq")
    zs_seq   = mean("zeroshot_seq")

    return {
        "n": n_res,
        "n_failed": n_failed,
        # Flat (trained format) baseline
        "flat_seq_acc":          flat_seq,
        "flat_flag_acc":         mean("flat_flag"),
        "flat_partial_acc":      mean("flat_partial"),
        # Zero-shot two-BIND
        "zeroshot_seq_acc":      zs_seq,
        "zeroshot_flag_acc":     mean("zeroshot_flag"),
        "zeroshot_partial_acc":  mean("zeroshot_partial"),
        "zeroshot_vs_flat":      zs_seq / flat_seq if flat_seq > 0 else 0.0,
        # Swap chain
        "swap_chain_seq_acc":    mean("swap_chain_seq"),
        "swap_chain_flag_acc":   mean("swap_chain_flag"),
        "swap_chain_partial_acc": mean("swap_chain_partial"),
        # Representation similarity
        "ab_cos_twobin_vs_flat": mean("ab_cos_twobin_vs_flat"),
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
        print(f"\n[seed {seed}] two-BIND zero-shot eval ...")
        r = run_twobin_experiment(str(ckpt), n=n, seed=seed_eval)
        all_results[str(seed)] = r
        print(f"  flat (trained):   seq={r['flat_seq_acc']:.4f}  flag={r['flat_flag_acc']:.4f}  partial={r['flat_partial_acc']:.4f}")
        print(f"  zeroshot 2-BIND:  seq={r['zeroshot_seq_acc']:.4f}  flag={r['zeroshot_flag_acc']:.4f}  partial={r['zeroshot_partial_acc']:.4f}  vs_flat={r['zeroshot_vs_flat']:.4f}")
        print(f"  swap_chain:       seq={r['swap_chain_seq_acc']:.4f}  flag={r['swap_chain_flag_acc']:.4f}")
        print(f"  ab_cos(2bind vs flat): {r['ab_cos_twobin_vs_flat']:.4f}  failed={r['n_failed']}")

    if all_results:
        print("\n" + "=" * 70)
        print("AGGREGATE")
        print("=" * 70)
        keys = ["flat_seq_acc", "flat_flag_acc",
                "zeroshot_seq_acc", "zeroshot_flag_acc", "zeroshot_partial_acc",
                "zeroshot_vs_flat", "swap_chain_seq_acc", "swap_chain_flag_acc",
                "ab_cos_twobin_vs_flat"]
        for k in keys:
            vals = [v[k] for v in all_results.values() if k in v]
            if vals:
                mu = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"  {k:35s}: {mu:.4f} ± {sd:.4f}")

    out_path = rundir / "eval_twobin_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot two-BIND evaluation")
    parser.add_argument("--rundir",        type=str, default=None)
    parser.add_argument("--subdir-prefix", type=str, default=None)
    parser.add_argument("--seeds",         type=str, default="1,2,3,4,5")
    parser.add_argument("--n",             type=int, default=300)
    parser.add_argument("--seed-eval",     type=int, default=42)
    parser.add_argument("--checkpoint",    type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint:
        r = run_twobin_experiment(args.checkpoint, n=args.n, seed=args.seed_eval)
        print(json.dumps(r, indent=2))
    else:
        if not args.rundir or not args.subdir_prefix:
            parser.error("--rundir and --subdir-prefix required unless --checkpoint is used")
        run_multiseed(args.rundir, args.subdir_prefix, args.seeds,
                      n=args.n, seed_eval=args.seed_eval)
