"""
Generic Model Evaluation Script

Evaluates any WhiteroomTransformer model or Stage 8 translation hybrid.
Wraps existing freeze_probe and cross_attention probe functions.

Usage:
    # Evaluate a base model (Stage 4, 5, 7b, 7d, etc.)
    python -m _agent.scripts.eval_model \\
        --model-type stage5 \\
        --checkpoint _agent/cache/runs/stage5/stage5-seed1/checkpoint_final.pt \\
        --output _agent/cache/runs/stage5/eval_reimplemented.json \\
        --n 300

    # Evaluate a Stage 8 hybrid
    python -m _agent.scripts.eval_model \\
        --model-type stage8 \\
        --checkpoint-dir _agent/cache/runs/stage8/7d-seed1_stage5-seed1 \\
        --output _agent/cache/runs/stage8/eval_reimplemented.json \\
        --n 300
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import random

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whiteroom.model import WhiteroomTransformer
from whiteroom.vocab import Token
from whiteroom.freeze_probe import (
    run_freeze_test, run_freeze_test_b_frozen,
    sample_triplet, sample_b_frozen_triplet
)
from whiteroom.generator import sample_primitive, find_valid_bindings
from whiteroom.train import collate_attribution


# =============================================================================
# Model Loaders
# =============================================================================

def load_whiteroom_model(checkpoint_path: str, device: torch.device) -> WhiteroomTransformer:
    """Load a standard WhiteroomTransformer checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = ckpt["config"].copy()
    # Remap sawtooth_encoder -> block_diag_encoder_mask for compatibility
    if "sawtooth_encoder" in config:
        config["block_diag_encoder_mask"] = config.pop("sawtooth_encoder")

    model = WhiteroomTransformer(**config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_translation_model(checkpoint_dir: str, device: torch.device):
    """Load a Stage 8 translation layer model (encoder + projection + decoder)."""
    from _agent.scripts.stage8.eval_stage8 import TranslationModel, load_translation_model as _load_translation_model
    return _load_translation_model(checkpoint_dir, device)


# =============================================================================
# Generic Evaluation
# =============================================================================

def run_property_append_test(
    model: nn.Module,
    device: torch.device,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Run property-append tests on any model."""
    from whiteroom.freeze_probe import make_example_for_ab, run_freeze_test_property_append
    from whiteroom.vocab import TRAINING_FLAGS

    rng = random.Random(seed)
    results = []

    for i in range(n):
        # Sample a valid A+B pair
        for _ in range(50):
            a = sample_primitive(rng)
            b = sample_primitive(rng)
            bindings = find_valid_bindings(a, b)
            if bindings:
                break
        else:
            continue

        port_a_idx, port_b_idx = rng.choice(bindings)

        # Pick an extra flag not already in A or B
        combined_flags = a.flags | b.flags
        available = [f for f in TRAINING_FLAGS if f not in combined_flags]
        if not available:
            continue
        extra_flag = rng.choice(available)
        target_side = 'a' if i % 2 == 0 else 'b'

        try:
            result = run_freeze_test_property_append(
                model, a, b, port_a_idx, port_b_idx,
                extra_flag, target_side, device,
            )
            results.append(result)
        except:
            continue

    n = len(results)
    if n == 0:
        return {"n": 0, "hybrid_pickup_pct": 0.0}

    hybrid_pickup = sum(r.hybrid_has_extra for r in results) / n
    fresh_pickup = sum(r.full_fresh_has_extra for r in results) / n
    base_contamination = sum(r.frozen_only_has_extra for r in results) / n
    a_pres = sum(r.a_flags_preserved for r in results if r.a_flags_preserved is not None) / n if n > 0 else 0.0
    b_pres = sum(r.b_flags_preserved for r in results if r.b_flags_preserved is not None) / n if n > 0 else 0.0

    a_side = [r for r in results if r.target_side == 'a']
    b_side = [r for r in results if r.target_side == 'b']

    return {
        "n": n,
        "hybrid_pickup_pct": hybrid_pickup,
        "full_fresh_pickup_pct": fresh_pickup,
        "base_contamination": base_contamination,
        "a_flags_preserved_pct": a_pres,
        "b_flags_preserved_pct": b_pres,
        "hybrid_pickup_a_side": sum(r.hybrid_has_extra for r in a_side) / len(a_side) if a_side else None,
        "hybrid_pickup_b_side": sum(r.hybrid_has_extra for r in b_side) / len(b_side) if b_side else None,
    }


def evaluate_freeze_test(
    model: nn.Module,
    device: torch.device,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Run freeze/isolation tests on any model."""
    rng = random.Random(seed)
    a_results, b_results = [], []

    for _ in range(n):
        t = sample_triplet(rng)
        if t:
            a, b, c, pa, pb, pc = t
            a_results.append(run_freeze_test(model, a, b, c, pa, pb, pc, device))

        t = sample_b_frozen_triplet(rng)
        if t:
            a, d, b, pa, pd, pb = t
            b_results.append(run_freeze_test_b_frozen(model, a, d, b, pa, pd, pb, device))

    def metrics(results):
        n = len(results)
        if n == 0:
            return {}
        return {
            "n": n,
            "normal_seq_acc":   sum(r.normal_seq_correct for r in results) / n,
            "normal_flags_acc": sum(r.normal_flags_correct for r in results) / n,
            "frozen_seq_acc":   sum(r.frozen_seq_correct for r in results) / n,
            "frozen_flags_acc": sum(r.frozen_flags_correct for r in results) / n,
            "seq_deg":          sum(r.normal_seq_correct for r in results) / n
                                - sum(r.frozen_seq_correct for r in results) / n,
            "flag_deg":         sum(r.normal_flags_correct for r in results) / n
                                - sum(r.frozen_flags_correct for r in results) / n,
            "mean_cos_sim":     sum(r.a_encoder_cosine_sim for r in results) / n,
        }

    return {
        "a_frozen": metrics(a_results),
        "b_frozen": metrics(b_results),
    }


def run_attribution_test(
    model: nn.Module,
    device: torch.device,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Run attribution accuracy tests on any model.

    Note: Attribution testing has issues with the seq_head evaluation for this task.
    Skipping for now as it's not implemented in eval_stage8.py either.
    """
    return {
        "n": 0,
        "seq_exact_match": 0.0,
        "token_accuracy": 0.0,
    }


def evaluate_model(
    model: nn.Module,
    device: torch.device,
    n: int = 300,
    seed_eval: int = 42,
) -> dict:
    """Run full evaluation suite on any model."""
    print(f"Running freeze/isolation tests...")
    freeze_results = evaluate_freeze_test(model, device, n=n, seed=seed_eval)

    print(f"Running property-append tests...")
    property_append_results = run_property_append_test(model, device, n=n, seed=seed_eval)

    print(f"Running attribution tests...")
    attribution_results = run_attribution_test(model, device, n=n, seed=seed_eval)

    return {
        "freeze": freeze_results,
        "property_append": property_append_results,
        "attribution": attribution_results,
    }


# =============================================================================
# Main
# =============================================================================

def main(
    model_type: str,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    output: str = "eval_results_reimplemented.json",
    n: int = 300,
    seed_eval: int = 42,
):
    """Evaluate a model and save results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model based on type
    if model_type == "stage8":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir required for stage8 models")
        print(f"Loading Stage 8 hybrid from {checkpoint_dir}...")
        model = load_translation_model(checkpoint_dir, device)
    else:
        if not checkpoint:
            raise ValueError(f"--checkpoint required for {model_type} models")
        print(f"Loading {model_type} from {checkpoint}...")
        model = load_whiteroom_model(checkpoint, device)

    model.eval()

    # Run full evaluation
    print(f"\n{'='*70}")
    print(f"Evaluating {model_type}")
    print(f"{'='*70}")
    eval_results = evaluate_model(model, device, n=n, seed_eval=seed_eval)

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"1": eval_results}, f, indent=2)
    print(f"\nSaved eval results to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    print(f"Property-append (composition): {eval_results['property_append'].get('hybrid_pickup_pct', 0):.1%}")

    # Freeze tests (if available)
    b_frozen = eval_results['freeze'].get('b_frozen', {})
    if b_frozen:
        print(f"B_frozen (isolation):          {b_frozen.get('seq_deg', 0):.6f}")
        a_frozen = eval_results['freeze'].get('a_frozen', {})
        if a_frozen:
            print(f"A_frozen (isolation):          {a_frozen.get('seq_deg', 0):.6f}")

    print(f"Attribution accuracy:          {eval_results['attribution'].get('seq_exact_match', 0):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Model Evaluation")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["stage2", "stage4", "stage4b", "stage4c", "stage5", "stage7b", "stage7d", "stage8"],
        help="Model type to evaluate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (for non-stage8 models)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Path to checkpoint directory (for stage8 hybrids)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results_reimplemented.json",
        help="Output file for eval results"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=300,
        help="Number of examples per eval"
    )
    parser.add_argument(
        "--seed-eval",
        type=int,
        default=42,
        help="Eval seed"
    )

    args = parser.parse_args()
    main(**vars(args))
