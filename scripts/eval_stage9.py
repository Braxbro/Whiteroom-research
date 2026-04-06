#!/usr/bin/env python3
"""
Stage 9 Evaluation: 3-stage model freeze tests.
Handles the WhiteroomTransformer3Stage checkpoint format.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from whiteroom.freeze_probe import (
    sample_triplet, sample_b_frozen_triplet,
    run_freeze_test, run_freeze_test_b_frozen
)
from whiteroom.model import WhiteroomTransformer3Stage
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_stage9_seed(ckpt_path, n_triplets=100, seed_eval=42):
    """Evaluate a single Stage 9 seed."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"].copy()
    config.pop("model_type", None)

    model = WhiteroomTransformer3Stage(**config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = random.Random(seed_eval)
    a_results, b_results = [], []

    for _ in range(n_triplets):
        t = sample_triplet(rng)
        if t:
            a, b, c, pa, pb, pc = t
            a_results.append(run_freeze_test(model, a, b, c, pa, pb, pc, device))

        t = sample_b_frozen_triplet(rng)
        if t:
            a, d, b, pa, pd, pb = t
            b_results.append(run_freeze_test_b_frozen(model, a, d, b, pa, pd, pb, device))

    def calc_metrics(results):
        n = len(results)
        if n == 0:
            return {}
        normal = sum(r.normal_seq_correct for r in results) / n
        frozen = sum(r.frozen_seq_correct for r in results) / n
        return {
            "n": n,
            "normal": normal,
            "frozen": frozen,
            "degradation": frozen - normal,
        }

    a_metrics = calc_metrics(a_results)
    b_metrics = calc_metrics(b_results)

    return {
        "a_frozen_deg": a_metrics.get("degradation", 0),
        "b_frozen_deg": b_metrics.get("degradation", 0),
        "a_metrics": a_metrics,
        "b_metrics": b_metrics,
    }


def main():
    seeds = [1, 2, 3, 4, 5]
    n_triplets = 100

    print("="*70)
    print("STAGE 9: Isolation Evaluation")
    print("="*70)

    results = {}
    for seed in seeds:
        ckpt_path = f"_agent/cache/runs/stage9/stage9-seed{seed}/checkpoint_final.pt"
        if not Path(ckpt_path).exists():
            print(f"Seed {seed}: checkpoint not found")
            continue

        print(f"\nSeed {seed}...")
        try:
            res = eval_stage9_seed(ckpt_path, n_triplets=n_triplets, seed_eval=42)
            results[seed] = res

            a_deg = res["a_frozen_deg"]
            b_deg = res["b_frozen_deg"]
            a_n = res["a_metrics"].get("n", 0)
            b_n = res["b_metrics"].get("n", 0)

            print(f"  A-frozen: {a_deg:+.4f} ({a_n} triplets)")
            print(f"  B-frozen: {b_deg:+.4f} ({b_n} triplets)")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        a_vals = [r["a_frozen_deg"] for r in results.values()]
        b_vals = [r["b_frozen_deg"] for r in results.values()]
        avg_a = sum(a_vals) / len(a_vals)
        avg_b = sum(b_vals) / len(b_vals)
        std_a = (sum((x - avg_a) ** 2 for x in a_vals) / len(a_vals)) ** 0.5
        std_b = (sum((x - avg_b) ** 2 for x in b_vals) / len(b_vals)) ** 0.5

        print(f"\n{'='*70}")
        print(f"STAGE 9 SUMMARY (n={len(results)} seeds)")
        print(f"{'='*70}")
        print(f"A-frozen degradation: {avg_a:+.4f} ± {std_a:.4f}")
        print(f"B-frozen degradation: {avg_b:+.4f} ± {std_b:.4f}")
        print(f"Average degradation:  {(avg_a+avg_b)/2:+.4f}")

        print(f"\nCOMPARISON TO BASELINES:")
        print(f"  Perfect isolation:        ~+0.000")
        print(f"  Stage 5 (2-stage):        ~+0.002")
        print(f"  7d (frozen encoder):      ~+0.002")
        print(f"  8a (linear projection):   ~+0.030-0.050")
        print(f"  8c (MLP projection):      ~+0.020")
        print(f"  Stage 9 (3-stage):        {(avg_a+avg_b)/2:+.4f}")

        # Save results
        output_path = Path("_agent/cache/runs/stage9/eval_summary.json")
        with open(output_path, "w") as f:
            json.dump({
                "a_frozen_deg": avg_a,
                "b_frozen_deg": avg_b,
                "a_std": std_a,
                "b_std": std_b,
                "avg_deg": (avg_a + avg_b) / 2,
                "by_seed": results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
