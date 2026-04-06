"""
Multi-seed evaluation: cache freeze, attribution, and property-append tests.

Usage:
    python eval_multiseed.py [--rundir DIR] [--seeds 1,2,3,4,5] [--n N] [--seed-eval N]

Writes results to <rundir>/eval_results.json and prints a summary table.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Add repo root to path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whiteroom.freeze_probe import run_experiment, run_experiment_property_append
from whiteroom.model import WhiteroomTransformer
from whiteroom.generator import sample_attribution_example, VOCAB_SIZE
from whiteroom.vocab import Token


# ---------------------------------------------------------------------------
# Attribution eval
# ---------------------------------------------------------------------------

def evaluate_attribution(checkpoint_path: str, n_samples: int = 300, seed: int = 77) -> dict:
    """
    Evaluate attribution accuracy: given [A|SEP|B|SEP|compound], predict attribution
    labels (ATTR_A / ATTR_B / ATTR_BOTH) for each compound feature.

    Returns seq_exact_match (all labels correct) and per-token accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect model type and instantiate appropriate class
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
    seq_correct = 0
    total_tokens = 0
    correct_tokens = 0

    for _ in range(n_samples):
        ex = sample_attribution_example(rng)

        src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        tgt = ex.target_tokens  # list of ints, includes END

        bos = torch.tensor([[Token.COMPOUND]], dtype=torch.long, device=device)

        with torch.no_grad():
            mem = model.encode(src)

        # Greedy decode
        ys = bos
        pred = []
        for _ in range(len(tgt) + 5):
            tgt_len = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            with torch.no_grad():
                dec_out = model.decode(ys, mem, tgt_mask=tgt_mask)
                next_tok = model.seq_head(dec_out[:, -1, :]).argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            tok = next_tok.item()
            pred.append(tok)
            if tok == Token.END:
                break

        # Trim pred to same length as target for token-level accuracy
        min_len = min(len(pred), len(tgt))
        correct_tokens += sum(pred[i] == tgt[i] for i in range(min_len))
        total_tokens += len(tgt)

        if pred == tgt:
            seq_correct += 1

    return {
        "n": n_samples,
        "seq_exact_match": seq_correct / n_samples,
        "token_accuracy":  correct_tokens / total_tokens if total_tokens > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def mean_std(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)


def fmt(mu, sd):
    return f"{mu:.4f} ± {sd:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir",    default="_agent/cache/runs/multiseed")
    parser.add_argument("--subdir-prefix", default="stage2",
                        help="Subdir prefix, e.g. 'stage2' -> stage2-seed{N}, 'stage4' -> stage4-seed{N}")
    parser.add_argument("--seeds",     default="1,2,3,4,5")
    parser.add_argument("--n",         type=int, default=300,
                        help="Triplets/pairs per eval task per seed")
    parser.add_argument("--seed-eval", type=int, default=1234,
                        help="RNG seed for eval sampling")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rundir = Path(args.rundir)
    outfile = rundir / "eval_results.json"

    all_results = {}

    for seed in seeds:
        ckpt = rundir / f"{args.subdir_prefix}-seed{seed}" / "checkpoint_final.pt"
        if not ckpt.exists():
            print(f"[seed {seed}] checkpoint not found: {ckpt}", flush=True)
            continue

        print(f"\n[seed {seed}] evaluating {ckpt.name} ...", flush=True)

        # 1. Cache freeze (A-frozen + B-frozen)
        print(f"  cache freeze test ({args.n} triplets) ...", flush=True)
        freeze = run_experiment(str(ckpt), n_triplets=args.n, seed=args.seed_eval)

        # 2. Property-append (both-frozen + fresh flag)
        print(f"  property-append test ({args.n} pairs) ...", flush=True)
        prop = run_experiment_property_append(str(ckpt), n_pairs=args.n, seed=args.seed_eval)

        # 3. Attribution
        print(f"  attribution eval ({args.n} samples) ...", flush=True)
        attr = evaluate_attribution(str(ckpt), n_samples=args.n, seed=args.seed_eval)

        all_results[seed] = {
            "freeze": freeze,
            "property_append": prop,
            "attribution": attr,
        }

        # Per-seed summary
        af = freeze.get("a_frozen", {})
        bf = freeze.get("b_frozen", {})
        print(f"  [cache freeze]  A-frozen seq {af.get('frozen_seq_acc',0):.4f}  "
              f"deg {af.get('seq_deg',0):.4f}  cos {af.get('mean_cos_sim',0):.4f}")
        print(f"                  B-frozen seq {bf.get('frozen_seq_acc',0):.4f}  "
              f"deg {bf.get('seq_deg',0):.4f}  cos {bf.get('mean_cos_sim',0):.4f}")
        print(f"  [prop-append]   hybrid pickup {prop.get('hybrid_pickup_pct',0):.4f}  "
              f"full-fresh {prop.get('full_fresh_pickup_pct',0):.4f}  "
              f"A-pres {prop.get('a_flags_preserved_pct',0):.4f}  "
              f"B-pres {prop.get('b_flags_preserved_pct',0):.4f}")
        print(f"  [attribution]   seq-exact {attr['seq_exact_match']:.4f}  "
              f"tok-acc {attr['token_accuracy']:.4f}")

    # Aggregate across seeds
    if len(all_results) < 2:
        print("\nNot enough seeds to aggregate.")
    else:
        print("\n" + "=" * 70)
        print("AGGREGATE (mean ± std across seeds)")
        print("=" * 70)

        def collect(fn):
            return [fn(all_results[s]) for s in sorted(all_results)]

        # Cache freeze
        af_seq   = collect(lambda r: r["freeze"].get("a_frozen", {}).get("frozen_seq_acc", float("nan")))
        af_deg   = collect(lambda r: r["freeze"].get("a_frozen", {}).get("seq_deg", float("nan")))
        af_cos   = collect(lambda r: r["freeze"].get("a_frozen", {}).get("mean_cos_sim", float("nan")))
        bf_seq   = collect(lambda r: r["freeze"].get("b_frozen", {}).get("frozen_seq_acc", float("nan")))
        bf_deg   = collect(lambda r: r["freeze"].get("b_frozen", {}).get("seq_deg", float("nan")))
        bf_cos   = collect(lambda r: r["freeze"].get("b_frozen", {}).get("mean_cos_sim", float("nan")))

        # Property-append
        pa_hyb   = collect(lambda r: r["property_append"].get("hybrid_pickup_pct", float("nan")))
        pa_ff    = collect(lambda r: r["property_append"].get("full_fresh_pickup_pct", float("nan")))
        pa_apres = collect(lambda r: r["property_append"].get("a_flags_preserved_pct", float("nan")))
        pa_bpres = collect(lambda r: r["property_append"].get("b_flags_preserved_pct", float("nan")))

        # Attribution
        at_seq   = collect(lambda r: r["attribution"]["seq_exact_match"])
        at_tok   = collect(lambda r: r["attribution"]["token_accuracy"])

        print(f"\nCache freeze — A-frozen:")
        print(f"  frozen seq acc  {fmt(*mean_std(af_seq))}")
        print(f"  seq degradation {fmt(*mean_std(af_deg))}")
        print(f"  cosine sim      {fmt(*mean_std(af_cos))}")

        print(f"\nCache freeze — B-frozen:")
        print(f"  frozen seq acc  {fmt(*mean_std(bf_seq))}")
        print(f"  seq degradation {fmt(*mean_std(bf_deg))}")
        print(f"  cosine sim      {fmt(*mean_std(bf_cos))}")

        print(f"\nProperty-append (both frozen):")
        print(f"  hybrid pickup   {fmt(*mean_std(pa_hyb))}")
        print(f"  full-fresh      {fmt(*mean_std(pa_ff))}")
        print(f"  A-flags pres.   {fmt(*mean_std(pa_apres))}")
        print(f"  B-flags pres.   {fmt(*mean_std(pa_bpres))}")

        print(f"\nAttribution:")
        print(f"  seq exact match {fmt(*mean_std(at_seq))}")
        print(f"  token accuracy  {fmt(*mean_std(at_tok))}")

    # Save
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
