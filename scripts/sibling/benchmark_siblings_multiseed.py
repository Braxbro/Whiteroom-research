"""
Per-seed sibling benchmark: each sibling evaluated against its own primary checkpoint.

Compares:
  - freeze_all       static baseline
  - full_fresh       upper bound
  - oracle           brute-force optimal
  - sibling_full     full-format sibling trained on this seed
  - sibling_compact  compact-format sibling trained on this seed

Reports per-seed results and aggregate, plus a seed-dependency summary
(does each sibling match oracle on its own seed?).

Usage:
    python benchmark_siblings_multiseed.py [--primary-dir DIR] [--sibling-dir DIR] [--n N]
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whiteroom.model import WhiteroomTransformer
from whiteroom.span_predictor import SpanFreezePredictor
from whiteroom.span_oracle import run_span_oracle, build_hybrid
from whiteroom.freeze_probe import make_example_for_ab, _greedy_from_memory, FLAG_TOKENS, _flag_tok
from whiteroom.generator import sample_primitive
from whiteroom.composition import find_valid_bindings
from whiteroom.vocab import Token, TRAINING_FLAGS, flag_token, edit_pos_id
from whiteroom.generator import VOCAB_SIZE


STRATEGIES = ["freeze_all", "full_fresh", "oracle", "sibling_full", "sibling_compact"]


def load_sibling(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SpanFreezePredictor(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("compact", False)


def evaluate_seed(primary_ckpt, sibling_full_ckpt, sibling_compact_ckpt,
                  n_pairs, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_p = torch.load(primary_ckpt, map_location=device, weights_only=False)
    primary = WhiteroomTransformer(**ckpt_p["config"]).to(device)
    primary.load_state_dict(ckpt_p["model_state"])
    primary.eval()

    sib_full,    _ = load_sibling(sibling_full_ckpt,    device)
    sib_compact, _ = load_sibling(sibling_compact_ckpt, device)

    rng = random.Random(seed)
    counts = {s: 0 for s in STRATEGIES}
    # Latency accumulators: total seconds spent per strategy decision
    times = {s: 0.0 for s in STRATEGIES}
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
        extra_tok  = int(flag_token(extra_flag))

        ex          = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        base_tokens = ex.input_tokens
        ext_tokens  = base_tokens + [extra_tok]
        L           = len(base_tokens)
        a_s, a_e    = ex.a_token_span
        b_s, b_e    = ex.b_token_span

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

        def span_mask(fa, fb_ind, fb):
            mask = [0] * L
            if fa:
                for i in range(a_s, a_e): mask[i] = 1
            if fb_ind:
                for i in range(a_e, b_s): mask[i] = 1
            if fb:
                for i in range(b_s, b_e): mask[i] = 1
            return mask

        # freeze_all: no policy decision needed, just use mem_old
        t0 = time.perf_counter()
        counts["freeze_all"] += flags_correct([1] * L)
        times["freeze_all"] += time.perf_counter() - t0

        # full_fresh: re-encode entire extended sequence
        t0 = time.perf_counter()
        counts["full_fresh"] += flags_correct([0] * L)
        times["full_fresh"] += time.perf_counter() - t0

        # oracle: brute-force 8 span combinations
        t0 = time.perf_counter()
        oracle_result = run_span_oracle(primary, a, b, port_a_idx, port_b_idx, extra_flag, device)
        if oracle_result.optimal_combo is not None:
            fa, fb_ind, fb = oracle_result.optimal_combo
        else:
            fa, fb_ind, fb = 1, 1, 1
        counts["oracle"] += flags_correct(span_mask(fa, fb_ind, fb))
        times["oracle"] += time.perf_counter() - t0

        # Full sibling: [old | SEP | new]
        t0 = time.perf_counter()
        sib_f_input = torch.tensor(
            base_tokens + [sep] + ext_tokens, dtype=torch.long, device=device)
        sfa, sfb_ind, sfb = sib_full.predict(sib_f_input)
        counts["sibling_full"] += flags_correct(span_mask(sfa, sfb_ind, sfb))
        times["sibling_full"] += time.perf_counter() - t0

        # Compact sibling: [old | SEP | pos_tok | new_tok]
        t0 = time.perf_counter()
        pos_tok = edit_pos_id(len(base_tokens), VOCAB_SIZE)
        sib_c_input = torch.tensor(
            base_tokens + [sep] + [pos_tok, extra_tok], dtype=torch.long, device=device)
        cfa, cfb_ind, cfb = sib_compact.predict(sib_c_input)
        counts["sibling_compact"] += flags_correct(span_mask(cfa, cfb_ind, cfb))
        times["sibling_compact"] += time.perf_counter() - t0

        n_total += 1

    return (
        {s: counts[s] / n_total for s in STRATEGIES}
        | {f"ms_{s}": times[s] / n_total * 1000 for s in STRATEGIES}
        | {"n": n_total}
    )


def mean_std(values):
    n = len(values)
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", default="_agent/cache/runs/multiseed")
    parser.add_argument("--sibling-dir", default="_agent/cache/runs/siblings-multiseed")
    parser.add_argument("--seeds",       default="1,2,3,4,5")
    parser.add_argument("--n",           type=int, default=300)
    parser.add_argument("--seed-eval",   type=int, default=99)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    primary_dir = Path(args.primary_dir)
    sibling_dir = Path(args.sibling_dir)
    outfile = sibling_dir / "benchmark_results.json"

    all_results = {}

    for seed in seeds:
        primary_ckpt        = primary_dir / f"stage2-seed{seed}" / "checkpoint_final.pt"
        sibling_full_ckpt   = sibling_dir / f"sibling_full_seed{seed}.pt"
        sibling_compact_ckpt = sibling_dir / f"sibling_compact_seed{seed}.pt"

        for p in [primary_ckpt, sibling_full_ckpt, sibling_compact_ckpt]:
            if not p.exists():
                print(f"[seed {seed}] missing {p.name}, skipping")
                break
        else:
            print(f"\n[seed {seed}] benchmarking...", flush=True)
            result = evaluate_seed(
                str(primary_ckpt), str(sibling_full_ckpt), str(sibling_compact_ckpt),
                args.n, args.seed_eval,
            )
            all_results[seed] = result
            row = "  ".join(f"{s}: {result[s]:.3f}" for s in STRATEGIES)
            print(f"  {row}", flush=True)

    if not all_results:
        print("No results.")
        return

    print("\n" + "=" * 80)
    print(f"{'Strategy':<18} {'Acc mean':>9} {'±Std':>7} {'ms/pair':>9}  per-seed acc")
    print("=" * 80)
    for strat in STRATEGIES:
        vals     = [all_results[s][strat]          for s in sorted(all_results)]
        ms_vals  = [all_results[s][f"ms_{strat}"]  for s in sorted(all_results)]
        mu, sd   = mean_std(vals)
        ms_mean  = sum(ms_vals) / len(ms_vals)
        per_seed = "  ".join(f"{v:.3f}" for v in vals)
        print(f"{strat:<18} {mu:>9.4f} {sd:>7.4f} {ms_mean:>9.2f}ms  {per_seed}")

    print("\n--- Seed-dependency check (sibling_full vs oracle, per seed) ---")
    for seed in sorted(all_results):
        r = all_results[seed]
        gap_full    = r["sibling_full"]    - r["oracle"]
        gap_compact = r["sibling_compact"] - r["oracle"]
        gap_freeze  = r["freeze_all"]      - r["oracle"]
        print(f"  seed {seed}: "
              f"oracle={r['oracle']:.3f}  "
              f"full={r['sibling_full']:.3f} (gap {gap_full:+.3f})  "
              f"compact={r['sibling_compact']:.3f} (gap {gap_compact:+.3f})  "
              f"freeze_all (gap {gap_freeze:+.3f})")

    with open(outfile, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
