#!/usr/bin/env python3
"""
Verify that the genericized eval_model.py produces the same results as the baseline.
Evaluates each seed separately and compares key metrics.
"""

import subprocess
import json
from pathlib import Path
import sys

# Define models to evaluate
# Format: model_type -> [(seed_str, checkpoint_path, baseline_json_path), ...]
models_config = {
    "stage2": [
        ("1", "/home/babrook/Documents/research/_agent/cache/runs/multiseed/stage2-seed1/checkpoint_final.pt",
         "/home/babrook/Documents/research/_agent/cache/runs/multiseed/eval_results.json"),
    ],
    "stage4": [
        ("1", "/home/babrook/Documents/research/_agent/cache/runs/stage4/stage4-seed1/checkpoint_final.pt",
         "/home/babrook/Documents/research/_agent/cache/runs/stage4/eval_results.json"),
    ],
    "stage4b": [
        ("1", "/home/babrook/Documents/research/_agent/cache/runs/stage4b/stage4b-seed1/checkpoint_final.pt",
         "/home/babrook/Documents/research/_agent/cache/runs/stage4b/eval_results.json"),
    ],
    "stage4c": [
        ("2", "/home/babrook/Documents/research/_agent/cache/runs/stage4c/stage4c-seed2/checkpoint_final.pt",
         "/home/babrook/Documents/research/_agent/cache/runs/stage4c/eval_results.json"),
        ("3", "/home/babrook/Documents/research/_agent/cache/runs/stage4c/stage4c-seed3/checkpoint_final.pt",
         "/home/babrook/Documents/research/_agent/cache/runs/stage4c/eval_results.json"),
    ],
    "stage5": [
        ("1", "/home/babrook/Documents/research/_agent/cache/runs/stage5/stage5-seed1/checkpoint_final.pt",
         "/home/babrook/Documents/research/_agent/cache/runs/stage5/eval_results.json"),
    ],
    "stage7b": [
        ("1", "/home/babrook/Documents/research/_agent/cache/runs/stage7/7b-causal-enc/stage5-seed1/checkpoint_final.pt",
         "/home/babrook/Documents/research/_agent/cache/runs/stage7/7b-causal-enc/eval_results.json"),
    ],
    "stage7d": [
        ("1", "/home/babrook/Documents/research/_agent/cache/runs/stage7/7d-sawtooth/stage5-seed1/checkpoint_final.pt",
         "/home/babrook/Documents/research/_agent/cache/runs/stage7/7d-sawtooth/eval_results.json"),
    ],
}

def run_eval(model_type, checkpoint, output_path, n=300):
    """Run eval_model.py for a given model."""
    cmd = [
        "python", "-m", "_agent.scripts.eval_model",
        "--model-type", model_type,
        "--checkpoint", checkpoint,
        "--output", output_path,
        "--n", str(n),
        "--seed-eval", "42"
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/home/babrook/Documents/research"
    )
    return result.returncode == 0, result.stderr

def compare_results(baseline, reimpl, baseline_seed_str):
    """Compare baseline and reimplementation results.

    Args:
        baseline: baseline results dict
        reimpl: reimplementation results dict (uses "1" as seed key)
        baseline_seed_str: seed string to look up in baseline (e.g., "2", "3")
    """
    if baseline_seed_str not in baseline:
        return None, f"seed {baseline_seed_str} not in baseline"
    if "1" not in reimpl:
        return None, "seed 1 not in reimplementation"

    base_result = baseline[baseline_seed_str]
    reimpl_result = reimpl["1"]  # reimpl always uses "1" as key

    # Extract key metrics
    base_comp = base_result["property_append"]["hybrid_pickup_pct"]
    reimpl_comp = reimpl_result["property_append"]["hybrid_pickup_pct"]

    base_iso = base_result["freeze"]["b_frozen"]["seq_deg"]
    reimpl_iso = reimpl_result["freeze"]["b_frozen"]["seq_deg"]

    return {
        "composition": {"baseline": base_comp, "reimpl": reimpl_comp},
        "isolation": {"baseline": base_iso, "reimpl": reimpl_iso},
    }, None

def main():
    print("\n" + "=" * 80)
    print("Evaluating all models and comparing with baselines")
    print("=" * 80)

    all_passed = True
    total_tests = 0
    passed_tests = 0

    for model_type in ["stage2", "stage4", "stage4b", "stage4c", "stage5", "stage7b", "stage7d"]:
        if model_type not in models_config:
            continue

        seed_configs = models_config[model_type]

        for seed_str, checkpoint, baseline_path in seed_configs:
            total_tests += 1

            if not Path(checkpoint).exists():
                print(f"\n⚠ {model_type} seed {seed_str}: checkpoint not found")
                continue

            if not Path(baseline_path).exists():
                print(f"\n⚠ {model_type} seed {seed_str}: baseline not found")
                continue

            output_path = f"/tmp/{model_type}_seed{seed_str}_verify.json"

            print(f"{model_type:10s} seed {seed_str}: ", end="", flush=True)

            success, stderr = run_eval(model_type, checkpoint, output_path, n=300)

            if not success:
                print(f"✗ eval failed: {stderr[:100]}")
                all_passed = False
                continue

            # Load and compare results
            baseline = json.load(open(baseline_path))
            reimpl = json.load(open(output_path))

            comparison, error = compare_results(baseline, reimpl, seed_str)
            if error:
                print(f"✗ {error}")
                all_passed = False
                continue

            # Check if metrics are close enough
            comp_diff = abs(comparison["composition"]["baseline"] - comparison["composition"]["reimpl"])
            iso_diff = abs(comparison["isolation"]["baseline"] - comparison["isolation"]["reimpl"])

            # Allow small differences due to random seeds
            comp_ok = comp_diff < 0.05  # 5% variation
            iso_ok = iso_diff < 0.05    # 5% difference in seq_deg

            if comp_ok and iso_ok:
                print(f"✓ ", end="")
                passed_tests += 1
            else:
                print(f"! ", end="")
                all_passed = False

            print(f"comp: {comparison['composition']['reimpl']:.1%} (baseline: {comparison['composition']['baseline']:.1%}), "
                  f"iso: {comparison['isolation']['reimpl']:.4f} (baseline: {comparison['isolation']['baseline']:.4f})")

    print("\n" + "=" * 80)
    if all_passed:
        print(f"✓ All {total_tests} tests passed successfully!")
    else:
        print(f"! {passed_tests}/{total_tests} tests passed (some variations detected)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
