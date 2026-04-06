"""
Zero-shot multi-append probe: does the compact sibling generalize to
inputs describing 2 or 3 simultaneous flag appends?

The sibling was trained on [old | SEP | pos | flag] (single edit).
This probe passes [old | SEP | pos1 | flag1 | pos2 | flag2] and
[old | SEP | pos1 | flag1 | pos2 | flag2 | pos3 | flag3] without
any fine-tuning, then checks if the predicted freeze mask matches
the oracle mask for the multi-append compound.

Three conditions per pair:
  - single:  trained distribution (sanity check)
  - double:  2 flags appended, zero-shot
  - triple:  3 flags appended, zero-shot (where available flags permit)

For each condition compares sibling prediction against freeze_all,
full_fresh, and oracle.

Usage:
    python probe_multiappend.py --sibling PATH --primary PATH [--n N]
"""
import argparse
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
from whiteroom.vocab import Token, TRAINING_FLAGS, flag_token, edit_pos_id


def evaluate(primary_ckpt, sibling_ckpt, n_pairs, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_p = torch.load(primary_ckpt, map_location=device, weights_only=False)
    primary = WhiteroomTransformer(**ckpt_p["config"]).to(device)
    primary.load_state_dict(ckpt_p["model_state"])
    primary.eval()

    ckpt_s = torch.load(sibling_ckpt, map_location=device, weights_only=False)
    sibling = SpanFreezePredictor(**ckpt_s["config"]).to(device)
    sibling.load_state_dict(ckpt_s["model_state"])
    sibling.eval()

    rng = random.Random(seed)
    sep = int(Token.SEP)

    STRATEGIES = ("freeze_all", "full_fresh", "oracle", "sibling")
    counts = {label: {s: 0 for s in STRATEGIES}
              for label in ("single", "double", "triple")}
    totals = {"single": 0, "double": 0, "triple": 0}

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

        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        base_tokens = ex.input_tokens
        L = len(base_tokens)
        a_s, a_e = ex.a_token_span
        b_s, b_e = ex.b_token_span

        src_old = torch.tensor(base_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            mem_old = primary.encode(src_old)

        def span_mask(fa, fb_ind, fb):
            mask = [0] * L
            if fa:
                for i in range(a_s, a_e): mask[i] = 1
            if fb_ind:
                for i in range(a_e, b_s): mask[i] = 1
            if fb:
                for i in range(b_s, b_e): mask[i] = 1
            return mask

        def run_condition(chosen_flags, label):
            ext_tokens = base_tokens + [int(flag_token(f)) for f in chosen_flags]
            target_flags = frozenset(
                _flag_tok(f) for f in (a.flags | b.flags | set(chosen_flags))
            )
            src_new = torch.tensor(ext_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                mem_new = primary.encode(src_new)

            def flags_correct(base_mask):
                full_mask = base_mask + [0] * len(chosen_flags)
                hybrid = build_hybrid(mem_old, mem_new, full_mask)
                with torch.no_grad():
                    pred = _greedy_from_memory(primary, hybrid, device, 32)
                return frozenset(t for t in pred if t in FLAG_TOKENS) == target_flags

            counts[label]["freeze_all"] += flags_correct([1] * L)
            counts[label]["full_fresh"]  += flags_correct([0] * L)

            # Oracle uses first extra flag (single-append oracle as reference)
            oracle_result = run_span_oracle(
                primary, a, b, port_a_idx, port_b_idx, chosen_flags[0], device)
            if oracle_result.optimal_combo is not None:
                fa, fb_ind, fb = oracle_result.optimal_combo
            else:
                fa, fb_ind, fb = 1, 1, 1
            counts[label]["oracle"] += flags_correct(span_mask(fa, fb_ind, fb))

            # Compact sibling: [old | SEP | pos1 | tok1 | pos2 | tok2 | ...]
            sib_seq = base_tokens + [sep]
            for i, f in enumerate(chosen_flags):
                sib_seq.append(edit_pos_id(L + i, VOCAB_SIZE))
                sib_seq.append(int(flag_token(f)))
            sib_input = torch.tensor(sib_seq, dtype=torch.long, device=device)
            sfa, sfb_ind, sfb = sibling.predict(sib_input)
            counts[label]["sibling"] += flags_correct(span_mask(sfa, sfb_ind, sfb))

            totals[label] += 1

        rng.shuffle(available)
        run_condition(available[:1], "single")
        if len(available) >= 2:
            run_condition(available[:2], "double")
        if len(available) >= 3:
            run_condition(available[:3], "triple")

    print(f"\n{'Condition':<10} {'n':>5}  " +
          "  ".join(f"{s:>12}" for s in STRATEGIES))
    print("-" * 72)
    for label in ("single", "double", "triple"):
        n = totals[label]
        if n == 0:
            continue
        row = "  ".join(f"{counts[label][s]/n:>12.3f}" for s in STRATEGIES)
        print(f"{label:<10} {n:>5}  {row}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sibling", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--n",    type=int, default=300)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()
    evaluate(args.primary, args.sibling, args.n, args.seed)


if __name__ == "__main__":
    main()
