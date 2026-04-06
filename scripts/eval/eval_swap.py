"""
Cross-pair memory swap evaluation.

Tests whether encoder representations are portable across compliant partners —
i.e., whether pre-computed encoder memory segments can be spliced at inference
time without re-encoding the full compound.

Setup:
    Sample a compliant quadruple (A, B, C, D) where all four cross-combinations
    are valid:
        A+B  (pa, pb)   — pair 1
        C+D  (pc, pd)   — pair 2
        A+D  (pa, pd)   — cross swap 1
        C+B  (pc, pb)   — cross swap 2

    Encodings:
        memory_AB = encode([A | BIND(pa,pb) | B])
        memory_CD = encode([C | BIND(pc,pd) | D])

    For cross swap 1 (A+D):
        memory_AD_fresh = encode([A | BIND(pa,pd) | D])   ← ground truth reference
        hybrid_AD = splice(
            A positions from memory_AB,
            BIND positions from memory_AD_fresh,   ← fresh BIND with correct ports
            D positions from memory_CD,
        )
        decode(hybrid_AD) → compare to compound(A, D)

    For cross swap 2 (C+B):
        memory_CB_fresh = encode([C | BIND(pc,pb) | B])   ← ground truth reference
        hybrid_CB = splice(
            C positions from memory_CD,
            BIND positions from memory_CB_fresh,
            B positions from memory_AB,
        )
        decode(hybrid_CB) → compare to compound(C, B)

Metrics:
    fresh_seq_acc      — accuracy of fully fresh encode (sanity check)
    swap_seq_acc       — accuracy of cross-swap hybrid
    swap_vs_fresh      — swap_seq_acc / fresh_seq_acc (relative performance)
    a_cos_sim          — cosine sim of A's representation in memory_AB vs memory_AD_fresh
    d_cos_sim          — cosine sim of D's representation in memory_CD vs memory_AD_fresh
    (same for C and B in the CB swap)

The key result: if swap_seq_acc is high, encoder representations are genuinely
portable — pre-computed segments can be spliced at O(1) without re-encoding.

Usage:
    python -m _agent.scripts.eval.eval_swap \\
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
from whiteroom.generator import serialize_entity, serialize_compound_output, sample_primitive, VOCAB_SIZE
from whiteroom.model import WhiteroomTransformer
from whiteroom.vocab import Token, port_idx_token
from whiteroom.freeze_probe import make_example_for_ab, _greedy_from_memory, FLAG_TOKENS


# ---------------------------------------------------------------------------
# Quadruple sampling
# ---------------------------------------------------------------------------

def sample_quad(
    rng: random.Random,
    max_outer: int = 300,
    max_inner: int = 100,
) -> Optional[Tuple]:
    """
    Sample (A, B, C, D, pa, pb, pc, pd) where:
        A+B valid at (pa, pb)
        C+D valid at (pc, pd)
        A+D valid at (pa, pd)   ← same pa, same pd
        C+B valid at (pc, pb)   ← same pc, same pb

    Returns None if no valid quadruple found within budget.
    """
    for _ in range(max_outer):
        a = sample_primitive(rng)
        b = sample_primitive(rng)
        ab_bindings = find_valid_bindings(a, b)
        if not ab_bindings:
            continue
        pa, pb = rng.choice(ab_bindings)

        for _ in range(max_inner):
            c = sample_primitive(rng)
            d = sample_primitive(rng)
            if c is a or d is b or c is b or d is a:
                continue

            # A+D valid at (pa, pd) — same port on A's side
            ad_bindings = find_valid_bindings(a, d)
            ad_at_pa = [(pa2, pd2) for pa2, pd2 in ad_bindings if pa2 == pa]
            if not ad_at_pa:
                continue
            _, pd = rng.choice(ad_at_pa)

            # C+B valid at (pc, pb) — same port on B's side
            cb_bindings = find_valid_bindings(c, b)
            cb_at_pb = [(pc2, pb2) for pc2, pb2 in cb_bindings if pb2 == pb]
            if not cb_at_pb:
                continue
            pc, _ = rng.choice(cb_at_pb)

            # C+D valid at (pc, pd)
            cd_bindings = find_valid_bindings(c, d)
            cd_at_pc_pd = [(pc2, pd2) for pc2, pd2 in cd_bindings
                           if pc2 == pc and pd2 == pd]
            if not cd_at_pc_pd:
                continue

            # All four combinations valid — return
            return a, b, c, d, pa, pb, pc, pd

    return None


# ---------------------------------------------------------------------------
# Memory construction helpers
# ---------------------------------------------------------------------------

def encode_pair(model, a, b, pa, pb, device):
    """Encode A+B and return (memory, a_end, b_start, b_end)."""
    ex = make_example_for_ab(a, b, pa, pb)
    src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        mem = model.encode(src)
    a_end = ex.a_token_span[1]
    b_start = ex.b_token_span[0]
    b_end = ex.b_token_span[1]
    return mem, a_end, b_start, b_end


def splice_memory(mem_left, a_end_left, mem_right, b_start_right, b_end_right,
                  mem_bind, a_end_bind, device):
    """
    Construct hybrid memory:
        [A positions from mem_left] [BIND positions from mem_bind] [D positions from mem_right]

    mem_left:       memory of some A+X encoding
    a_end_left:     end of A segment in mem_left
    mem_right:      memory of some Y+D encoding
    b_start_right:  start of D segment in mem_right
    b_end_right:    end of D segment in mem_right
    mem_bind:       memory of A+D fresh encoding (for BIND positions)
    a_end_bind:     end of A segment in mem_bind (= start of BIND)
    """
    a_seg   = mem_left[0, :a_end_left, :]           # A from left
    bind_seg = mem_bind[0, a_end_bind:a_end_bind+3, :]  # BIND from fresh
    d_seg   = mem_right[0, b_start_right:b_end_right, :]  # D from right

    hybrid = torch.cat([a_seg, bind_seg, d_seg], dim=0).unsqueeze(0)
    return hybrid


# ---------------------------------------------------------------------------
# Single swap test
# ---------------------------------------------------------------------------

def run_swap_test(model, a, b, c, d, pa, pb, pc, pd, device):
    """
    Run both cross-swap directions for one quadruple:
        Swap 1: A (from AB) + D (from CD)  →  compound(A, D)
        Swap 2: C (from CD) + B (from AB)  →  compound(C, B)

    Returns dict with metrics for both swaps.
    """
    model.eval()

    # Encode all four combinations
    mem_ab, a_end_ab, b_start_ab, b_end_ab = encode_pair(model, a, b, pa, pb, device)
    mem_cd, a_end_cd, b_start_cd, b_end_cd = encode_pair(model, c, d, pc, pd, device)
    mem_ad, a_end_ad, b_start_ad, b_end_ad = encode_pair(model, a, d, pa, pd, device)
    mem_cb, a_end_cb, b_start_cb, b_end_cb = encode_pair(model, c, b, pc, pb, device)

    def decode_and_check(mem, target_tokens):
        pred = _greedy_from_memory(model, mem, device, max_len=32)
        try:
            pred = pred[:pred.index(Token.END) + 1]
        except ValueError:
            pass
        seq_correct = (pred == target_tokens)
        pred_flags = {t for t in pred if t in FLAG_TOKENS}
        tgt_flags  = {t for t in target_tokens if t in FLAG_TOKENS}
        flag_correct = (pred_flags == tgt_flags)
        return seq_correct, flag_correct

    def cosine(v1, v2):
        return torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)).item()

    target_ad = make_example_for_ab(a, d, pa, pd).target_tokens
    target_cb = make_example_for_ab(c, b, pc, pb).target_tokens

    # --- Swap 1: A from AB, D from CD ---
    hybrid_ad = splice_memory(
        mem_ab, a_end_ab,          # A from AB
        mem_cd, b_start_cd, b_end_cd,  # D from CD
        mem_ad, a_end_ad,          # BIND from fresh AD
        device,
    )

    fresh_ad_seq, fresh_ad_flag = decode_and_check(mem_ad, target_ad)
    swap_ad_seq,  swap_ad_flag  = decode_and_check(hybrid_ad, target_ad)

    # Cosine sim: how similar are A and D representations across partner contexts?
    a_cos = cosine(
        mem_ab[0, :a_end_ab, :].mean(0),
        mem_ad[0, :a_end_ad, :].mean(0),
    )
    d_cos = cosine(
        mem_cd[0, b_start_cd:b_end_cd, :].mean(0),
        mem_ad[0, b_start_ad:b_end_ad, :].mean(0),
    )

    # --- Swap 2: C from CD, B from AB ---
    hybrid_cb = splice_memory(
        mem_cd, a_end_cd,          # C from CD
        mem_ab, b_start_ab, b_end_ab,  # B from AB
        mem_cb, a_end_cb,          # BIND from fresh CB
        device,
    )

    fresh_cb_seq, fresh_cb_flag = decode_and_check(mem_cb, target_cb)
    swap_cb_seq,  swap_cb_flag  = decode_and_check(hybrid_cb, target_cb)

    c_cos = cosine(
        mem_cd[0, :a_end_cd, :].mean(0),
        mem_cb[0, :a_end_cb, :].mean(0),
    )
    b_cos = cosine(
        mem_ab[0, b_start_ab:b_end_ab, :].mean(0),
        mem_cb[0, b_start_cb:b_end_cb, :].mean(0),
    )

    return {
        "ad": {
            "fresh_seq":  fresh_ad_seq,
            "fresh_flag": fresh_ad_flag,
            "swap_seq":   swap_ad_seq,
            "swap_flag":  swap_ad_flag,
            "a_cos_sim":  a_cos,
            "d_cos_sim":  d_cos,
        },
        "cb": {
            "fresh_seq":  fresh_cb_seq,
            "fresh_flag": fresh_cb_flag,
            "swap_seq":   swap_cb_seq,
            "swap_flag":  swap_cb_flag,
            "c_cos_sim":  c_cos,
            "b_cos_sim":  b_cos,
        },
    }


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_swap_experiment(checkpoint_path: str, n: int = 300, seed: int = 42) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"].copy()
    model_type = config.pop("model_type", "2stage")
    # Strip keys not in WhiteroomTransformer.__init__ signature
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
    ad_results, cb_results = [], []
    n_failed = 0

    for i in range(n):
        quad = sample_quad(rng)
        if quad is None:
            n_failed += 1
            continue
        a, b, c, d, pa, pb, pc, pd = quad
        r = run_swap_test(model, a, b, c, d, pa, pb, pc, pd, device)
        ad_results.append(r["ad"])
        cb_results.append(r["cb"])

    def metrics(results, left_cos_key, right_cos_key):
        n = len(results)
        if n == 0:
            return {}
        fresh_seq  = sum(r["fresh_seq"]  for r in results) / n
        swap_seq   = sum(r["swap_seq"]   for r in results) / n
        fresh_flag = sum(r["fresh_flag"] for r in results) / n
        swap_flag  = sum(r["swap_flag"]  for r in results) / n
        left_cos   = sum(r[left_cos_key]  for r in results) / n
        right_cos  = sum(r[right_cos_key] for r in results) / n
        return {
            "n": n,
            "fresh_seq_acc":   fresh_seq,
            "swap_seq_acc":    swap_seq,
            "swap_vs_fresh":   swap_seq / fresh_seq if fresh_seq > 0 else 0.0,
            "fresh_flag_acc":  fresh_flag,
            "swap_flag_acc":   swap_flag,
            "left_cos_sim":    left_cos,
            "right_cos_sim":   right_cos,
        }

    return {
        "ad_swap": metrics(ad_results, "a_cos_sim", "d_cos_sim"),
        "cb_swap": metrics(cb_results, "c_cos_sim", "b_cos_sim"),
        "n_failed_quads": n_failed,
    }


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_multiseed(rundir: str, subdir_prefix: str, seeds: str,
                  n: int = 300, seed_eval: int = 42):
    rundir = Path(rundir)
    seed_list = [int(s) for s in seeds.split(",")]
    all_results = {}

    for seed in seed_list:
        subdir = rundir / f"{subdir_prefix}-seed{seed}"
        ckpt = subdir / "checkpoint_final.pt"
        if not ckpt.exists():
            print(f"[seed {seed}] checkpoint not found, skipping")
            continue

        print(f"\n[seed {seed}] running swap eval on {ckpt.name} ...")
        result = run_swap_experiment(str(ckpt), n=n, seed=seed_eval)
        all_results[str(seed)] = result

        ad = result["ad_swap"]
        cb = result["cb_swap"]
        print(f"  [A+D swap]  fresh={ad['fresh_seq_acc']:.4f}  "
              f"swap={ad['swap_seq_acc']:.4f}  "
              f"swap/fresh={ad['swap_vs_fresh']:.4f}  "
              f"A_cos={ad['left_cos_sim']:.4f}  D_cos={ad['right_cos_sim']:.4f}")
        print(f"  [C+B swap]  fresh={cb['fresh_seq_acc']:.4f}  "
              f"swap={cb['swap_seq_acc']:.4f}  "
              f"swap/fresh={cb['swap_vs_fresh']:.4f}  "
              f"C_cos={cb['left_cos_sim']:.4f}  B_cos={cb['right_cos_sim']:.4f}")
        print(f"  failed quads: {result['n_failed_quads']}")

    # Aggregate
    import statistics
    if all_results:
        print("\n" + "=" * 70)
        print("AGGREGATE")
        print("=" * 70)
        for swap_key, label in [("ad_swap", "A+D swap"), ("cb_swap", "C+B swap")]:
            vals = [v[swap_key] for v in all_results.values() if swap_key in v]
            if not vals:
                continue
            fresh = [v["fresh_seq_acc"] for v in vals]
            swap  = [v["swap_seq_acc"]  for v in vals]
            ratio = [v["swap_vs_fresh"] for v in vals]
            lcos  = [v["left_cos_sim"]  for v in vals]
            rcos  = [v["right_cos_sim"] for v in vals]
            print(f"\n{label}:")
            print(f"  fresh seq acc:   {statistics.mean(fresh):.4f} ± {statistics.stdev(fresh) if len(fresh)>1 else 0:.4f}")
            print(f"  swap seq acc:    {statistics.mean(swap):.4f} ± {statistics.stdev(swap) if len(swap)>1 else 0:.4f}")
            print(f"  swap/fresh:      {statistics.mean(ratio):.4f} ± {statistics.stdev(ratio) if len(ratio)>1 else 0:.4f}")
            print(f"  left  cos sim:   {statistics.mean(lcos):.4f} ± {statistics.stdev(lcos) if len(lcos)>1 else 0:.4f}")
            print(f"  right cos sim:   {statistics.mean(rcos):.4f} ± {statistics.stdev(rcos) if len(rcos)>1 else 0:.4f}")

    out_path = rundir / "eval_swap_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-pair memory swap evaluation")
    parser.add_argument("--rundir",         type=str, default=None)
    parser.add_argument("--subdir-prefix",  type=str, default=None)
    parser.add_argument("--seeds",          type=str, default="1,2,3,4,5")
    parser.add_argument("--n",              type=int, default=300)
    parser.add_argument("--seed-eval",      type=int, default=42)
    # Single checkpoint mode
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Evaluate a single checkpoint directly")
    args = parser.parse_args()

    if args.checkpoint:
        result = run_swap_experiment(args.checkpoint, n=args.n, seed=args.seed_eval)
        print(json.dumps(result, indent=2))
    else:
        if not args.rundir or not args.subdir_prefix:
            parser.error("--rundir and --subdir-prefix are required unless --checkpoint is used")
        run_multiseed(args.rundir, args.subdir_prefix, args.seeds,
                      n=args.n, seed_eval=args.seed_eval)
