"""
Benchmark: sibling-guided freeze vs static freeze patterns vs full recompute.

For each primary checkpoint, evaluates flag accuracy under 7 strategies:
  1. full_fresh       — re-encode everything (upper bound, most compute)
  2. freeze_all       — freeze all positions, use old memory (cheapest, status quo)
  3. freeze_A_only    — freeze A span, re-encode BIND+B+extra fresh
  4. freeze_B_only    — freeze B span, re-encode A+BIND+extra fresh
  5. freeze_A_B       — freeze A and B, re-encode BIND fresh
  6. oracle           — brute-force optimal span mask per example (theoretical best)
  7. sibling          — sibling-predicted span mask (practical learned policy)

Strategies 1-5 are static (no learned policy). The sibling adds ~108K params and
one forward pass. This benchmarks whether learned policy is worth it over static heuristics.

Usage:
    python benchmark_sibling.py [--primary-dir DIR] [--sibling PATH] [--n N]
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whiteroom.model import WhiteroomTransformer
from whiteroom.span_predictor import SpanFreezePredictor
from whiteroom.span_oracle import run_span_oracle, build_hybrid
from whiteroom.freeze_probe import make_example_for_ab, _greedy_from_memory, FLAG_TOKENS, _flag_tok
from whiteroom.generator import sample_primitive, VOCAB_SIZE
from whiteroom.composition import find_valid_bindings
from whiteroom.vocab import Token, TRAINING_FLAGS, flag_token


STRATEGIES = [
    "full_fresh",
    "freeze_all",
    "freeze_A_only",
    "freeze_B_only",
    "freeze_A_B",
    "oracle",
    "sibling",
    "sibling_compact",
]


def evaluate(primary_ckpt: str, sibling_ckpt: str, n_pairs: int, seed: int,
             sibling_compact_ckpt: str = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_p = torch.load(primary_ckpt, map_location=device, weights_only=False)
    primary = WhiteroomTransformer(**ckpt_p["config"]).to(device)
    primary.load_state_dict(ckpt_p["model_state"])
    primary.eval()

    ckpt_s = torch.load(sibling_ckpt, map_location=device, weights_only=False)
    sibling = SpanFreezePredictor(**ckpt_s["config"]).to(device)
    sibling.load_state_dict(ckpt_s["model_state"])
    sibling.eval()
    sibling_compact_model = None
    if sibling_compact_ckpt and Path(sibling_compact_ckpt).exists():
        ckpt_sc = torch.load(sibling_compact_ckpt, map_location=device, weights_only=False)
        sibling_compact_model = SpanFreezePredictor(**ckpt_sc["config"]).to(device)
        sibling_compact_model.load_state_dict(ckpt_sc["model_state"])
        sibling_compact_model.eval()

    rng = random.Random(seed)
    counts = {s: 0 for s in STRATEGIES}
    n_total = 0
    sep = int(Token.SEP)

    for _ in range(n_pairs):
        for _ in range(50):
            a = sample_primitive(rng)
            b = sample_primitive(rng)
            bindings = find_valid_bindings(a, b)
            if bindings:
                break
        else:
            continue

        port_a_idx, port_b_idx = rng.choice(bindings)
        combined = a.flags | b.flags
        available = [f for f in TRAINING_FLAGS if f not in combined]
        if not available:
            continue
        extra_flag = rng.choice(available)
        extra_tok = flag_token(extra_flag)

        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        base_tokens = ex.input_tokens
        ext_tokens  = base_tokens + [extra_tok]
        L   = len(base_tokens)
        a_s, a_e = ex.a_token_span
        b_s, b_e = ex.b_token_span

        target_flags = frozenset(_flag_tok(f) for f in (a.flags | b.flags | {extra_flag}))

        src_old = torch.tensor(base_tokens, dtype=torch.long, device=device).unsqueeze(0)
        src_new = torch.tensor(ext_tokens,  dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            mem_old = primary.encode(src_old)
            mem_new = primary.encode(src_new)

        def flags_correct(mask):
            hybrid = build_hybrid(mem_old, mem_new, mask)
            with torch.no_grad():
                pred = _greedy_from_memory(primary, hybrid, device, 32)
            return frozenset(t for t in pred if t in FLAG_TOKENS) == target_flags

        def span_mask(freeze_a, freeze_bind, freeze_b):
            mask = [0] * L
            if freeze_a:
                for i in range(a_s, a_e): mask[i] = 1
            if freeze_bind:
                for i in range(a_e, b_s): mask[i] = 1
            if freeze_b:
                for i in range(b_s, b_e): mask[i] = 1
            return mask

        # Static strategies
        counts["full_fresh"]    += flags_correct([0] * L)
        counts["freeze_all"]    += flags_correct([1] * L)
        counts["freeze_A_only"] += flags_correct(span_mask(True,  False, False))
        counts["freeze_B_only"] += flags_correct(span_mask(False, False, True))
        counts["freeze_A_B"]    += flags_correct(span_mask(True,  False, True))

        # Oracle
        oracle_result = run_span_oracle(primary, a, b, port_a_idx, port_b_idx, extra_flag, device)
        if oracle_result.optimal_combo is not None:
            fa, fb_ind, fb = oracle_result.optimal_combo
        else:
            fa, fb_ind, fb = 1, 1, 1
        counts["oracle"] += flags_correct(span_mask(fa, fb_ind, fb))

        # Sibling (full format)
        sib_input = torch.tensor(
            base_tokens + [sep] + ext_tokens, dtype=torch.long, device=device
        )
        sfa, sfb_ind, sfb = sibling.predict(sib_input)
        counts["sibling"] += flags_correct(span_mask(sfa, sfb_ind, sfb))

        # Sibling compact format
        if sibling_compact_model is not None:
            from whiteroom.vocab import edit_pos_id
            from whiteroom.generator import VOCAB_SIZE as _VS
            pos_tok = edit_pos_id(len(base_tokens), _VS)
            sib_c_input = torch.tensor(
                base_tokens + [sep] + [pos_tok, extra_tok], dtype=torch.long, device=device
            )
            cfa, cfb_ind, cfb = sibling_compact_model.predict(sib_c_input)
            counts["sibling_compact"] += flags_correct(span_mask(cfa, cfb_ind, cfb))
        else:
            counts["sibling_compact"] += float("nan")

        n_total += 1

    result = {"n": n_total}
    for s in STRATEGIES:
        v = counts[s]
        result[s] = v / n_total if v == v else float("nan")  # nan check
    return result


def mean_std(values):
    n = len(values)
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", default="_agent/cache/runs/multiseed",
                        help="Directory containing stage2-seed{N}/ subdirs")
    parser.add_argument("--seeds",       default="1,2,3,4,5")
    parser.add_argument("--sibling",         default="_agent/cache/runs/sibling-span-predictor/sibling_final.pt")
    parser.add_argument("--sibling-compact", default="_agent/cache/runs/sibling-compact/sibling_compact.pt",
                        help="Optional compact sibling checkpoint for comparison")
    parser.add_argument("--n",           type=int, default=300)
    parser.add_argument("--seed-eval",   type=int, default=99)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    primary_dir = Path(args.primary_dir)
    outfile = primary_dir / "sibling_benchmark.json"

    all_results = {}

    for seed in seeds:
        ckpt = primary_dir / f"stage2-seed{seed}" / "checkpoint_final.pt"
        if not ckpt.exists():
            print(f"[seed {seed}] missing checkpoint, skipping", flush=True)
            continue
        print(f"\n[seed {seed}] ...", flush=True)
        result = evaluate(str(ckpt), args.sibling, args.n, args.seed_eval,
                          sibling_compact_ckpt=args.sibling_compact)
        all_results[seed] = result

        row = "  ".join(f"{s}: {result[s]:.3f}" for s in STRATEGIES)
        print(f"  {row}", flush=True)

    # Aggregate
    print("\n" + "=" * 72)
    print(f"{'Strategy':<16} {'Mean':>8} {'±Std':>8}  {'per-seed'}")
    print("=" * 72)
    for strat in STRATEGIES:
        vals = [all_results[s][strat] for s in sorted(all_results)]
        mu, sd = mean_std(vals)
        per_seed = "  ".join(f"{v:.3f}" for v in vals)
        print(f"{strat:<16} {mu:>8.4f} {sd:>8.4f}  {per_seed}")

    with open(outfile, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
