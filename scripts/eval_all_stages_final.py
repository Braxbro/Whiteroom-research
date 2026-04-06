#!/usr/bin/env python3
"""
Final evaluation: Compare Stage 8d, 8e, 9 against baselines.
Runs composition and isolation tests.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from whiteroom.freeze_probe import run_experiment, run_experiment_property_append
from whiteroom.model import WhiteroomTransformer, WhiteroomTransformer3Stage
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check which checkpoints exist
stages_to_eval = {
    "8a_linear_baseline": {
        "dir": "_agent/cache/runs/stage8/7d-seed1_stage5-seed1",
        "type": "baseline",
        "desc": "Linear projection, frozen decoder (baseline)"
    },
    "8c_mlp_baseline": {
        "dir": "_agent/cache/runs/stage8/mlp-7d-seed1_stage5-seed1",
        "type": "baseline",
        "desc": "MLP projection, frozen decoder (baseline)"
    },
    "8d_linear_finetune": {
        "dir": "_agent/cache/runs/stage8/8d-linear-unfreeze-seed1",
        "type": "finetune",
        "desc": "Linear projection, unfrozen decoder (fine-tuned)"
    },
    "8e_mlp_finetune": {
        "dir": "_agent/cache/runs/stage8/8e-mlp-unfreeze-seed1",
        "type": "finetune",
        "desc": "MLP projection, unfrozen decoder (fine-tuned)"
    },
    "stage5": {
        "dir": "_agent/cache/runs/stage5/stage5-seed1",
        "type": "baseline",
        "desc": "Standard 2-stage (baseline)"
    },
    "stage9": {
        "dir": "_agent/cache/runs/stage9/stage9-seed1",
        "type": "new",
        "desc": "3-stage model from scratch"
    },
}

results = {}

for stage_name, stage_info in stages_to_eval.items():
    ckpt_path = Path(stage_info["dir"]) / "checkpoint_final.pt"

    if not ckpt_path.exists():
        print(f"⊘ {stage_name:20s} — checkpoint not found")
        continue

    print(f"\nEvaluating {stage_name}...")
    print(f"  Type: {stage_info['desc']}")

    try:
        # Run composition + isolation test
        exp_results = run_experiment(str(ckpt_path), n_triplets=200, seed=42)

        results[stage_name] = {
            "checkpoint": str(ckpt_path),
            "description": stage_info["desc"],
            "results": exp_results,
        }

        # Extract key metrics
        if "a_frozen_deg" in exp_results:
            print(f"  A-frozen degradation: {exp_results.get('a_frozen_deg', 'N/A')}")
        if "b_frozen_deg" in exp_results:
            print(f"  B-frozen degradation: {exp_results.get('b_frozen_deg', 'N/A')}")

        print(f"  ✓ Complete")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        results[stage_name] = {"error": str(e)}

# Save results
output_path = Path("_agent/cache/runs/final_comparison.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"Results saved to: {output_path}")
print(f"{'='*70}")

# Summary table
print("\nSUMMARY COMPARISON:")
print("-" * 70)
for stage_name in ["8a_linear_baseline", "8c_mlp_baseline", "stage5", "8d_linear_finetune", "8e_mlp_finetune", "stage9"]:
    if stage_name not in results:
        continue
    r = results[stage_name]
    if "error" in r:
        print(f"{stage_name:25s} — ERROR")
    else:
        exp_r = r.get("results", {})
        a_deg = exp_r.get("a_frozen_deg", "N/A")
        b_deg = exp_r.get("b_frozen_deg", "N/A")
        desc = r["description"]
        print(f"{stage_name:25s} — A_deg={a_deg:6.4f}  B_deg={b_deg:6.4f}  ({desc})")
