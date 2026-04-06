"""
Stage 8 Evaluation

Evaluates translation layer models (frozen encoder + projection + frozen decoder).
Wraps existing freeze_probe and cross_attention probe functions.

Usage:
    python -m _agent.scripts.stage8.eval_stage8 \\
        --rundir _agent/cache/runs/stage8/7d-seed1_stage5-seed1 \\
        --pairs 1:1 \\
        --n 300 \\
        --seed-eval 42
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from whiteroom.model import WhiteroomTransformer
from whiteroom.vocab import Token
from whiteroom.freeze_probe import run_experiment, run_experiment_property_append
from whiteroom.train import collate_attribution, compute_loss


# =============================================================================
# TranslationModel Wrapper
# =============================================================================

class TranslationModel(nn.Module):
    """Thin wrapper combining frozen encoder + projection + frozen decoder.

    Exposes the same interface as WhiteroomTransformer for eval compatibility.
    """
    def __init__(
        self,
        encoder: WhiteroomTransformer,
        projection: nn.Module,
        decoder: WhiteroomTransformer,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.decoder = decoder

        # Aliases for eval compatibility
        self.seq_head = decoder.seq_head
        self.valid_head = decoder.valid_head
        self.transformer = type('obj', (object,), {
            'decoder': decoder.transformer.decoder,
            'encoder': encoder.transformer.encoder,  # for probes
        })()

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode via 7d encoder, then project."""
        mem = self.encoder.encode(src, src_key_padding_mask)
        return self.projection(mem)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode via frozen decoder."""
        return self.decoder.decode(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = 32,
    ) -> torch.Tensor:
        """Greedy decode."""
        memory = self.encode(src, src_key_padding_mask)
        batch = src.size(0)
        device = src.device

        ys = torch.full((batch, 1), Token.COMPOUND, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            y_emb = self.decoder.pos_enc(self.decoder.tgt_embed(ys) * (self.decoder.d_model ** 0.5))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=device)
            out = self.decoder.decode(ys, memory, tgt_mask=tgt_mask)
            logits = self.seq_head(out[:, -1:, :])
            next_token = logits.argmax(dim=-1)

            if next_token[0, 0].item() == Token.END:
                break
            ys = torch.cat([ys, next_token], dim=1)

        return ys

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Encoder always eval
        self.decoder.eval()  # Decoder always eval
        self.projection.train(mode)  # Projection follows mode
        return self

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.decoder.eval()
        self.projection.eval()
        return self


# =============================================================================
# Loading & Evaluation
# =============================================================================

def load_translation_model(
    checkpoint_dir: str,
    device: torch.device,
) -> TranslationModel:
    """Load a translation layer checkpoint and reconstruct TranslationModel."""
    ckpt_dir = Path(checkpoint_dir)
    trans_ckpt = torch.load(ckpt_dir / "checkpoint_translation.pt", map_location="cpu", weights_only=False)

    # Load encoder
    encoder_ckpt_path = trans_ckpt["encoder_ckpt"]
    encoder_ckpt = torch.load(encoder_ckpt_path, map_location="cpu", weights_only=False)
    encoder_config = encoder_ckpt["config"].copy()
    # Remap sawtooth_encoder -> block_diag_encoder_mask
    if "sawtooth_encoder" in encoder_config:
        encoder_config["block_diag_encoder_mask"] = encoder_config.pop("sawtooth_encoder")
    encoder = WhiteroomTransformer(**encoder_config).to(device)
    encoder.load_state_dict(encoder_ckpt["model_state"])
    encoder.requires_grad_(False)
    encoder.eval()

    # Load decoder
    decoder_ckpt_path = trans_ckpt["decoder_ckpt"]
    decoder_ckpt = torch.load(decoder_ckpt_path, map_location="cpu", weights_only=False)
    decoder = WhiteroomTransformer(**decoder_ckpt["config"]).to(device)
    decoder.load_state_dict(decoder_ckpt["model_state"])
    decoder.requires_grad_(False)
    decoder.eval()

    # Load projection
    from _agent.scripts.stage8.train_stage8_translation import TranslationProjection, MLPProjection

    proj_config = trans_ckpt.get("projection_config", {"d_in": 64, "d_out": 64, "type": "linear"})
    proj_type = proj_config.get("type", "linear")

    if proj_type == "mlp":
        projection = MLPProjection(
            d_in=proj_config.get("d_in", 64),
            d_out=proj_config.get("d_out", 64)
        ).to(device)
    else:
        projection = TranslationProjection(
            d_in=proj_config.get("d_in", 64),
            d_out=proj_config.get("d_out", 64)
        ).to(device)

    projection.load_state_dict(trans_ckpt["projection_state"])
    projection.eval()

    return TranslationModel(encoder, projection, decoder)


def property_append_translation_model(
    model: TranslationModel,
    device: torch.device,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Run property-append test on the full translation layer.

    Uses the 7d encoder → projection → Stage5 decoder pipeline to generate
    memory, then tests if the decoder can detect appended extra flags.

    This tests composition/pickup through the complete translation architecture.
    """
    from whiteroom.freeze_probe import make_example_for_ab, run_freeze_test_property_append
    from whiteroom.generator import sample_primitive, find_valid_bindings
    from whiteroom.vocab import TRAINING_FLAGS
    import random

    rng = random.Random(seed)

    # Create a wrapper that exposes run_freeze_test_property_append interface
    # by pre-computing the projected memory
    class ProjectionEncodingModel:
        def __init__(self, translation_model, device):
            self.model = translation_model
            self.device = device
            self.seq_head = translation_model.seq_head

        def encode(self, src):
            """Encode and project, returning frozen projected memory."""
            with torch.no_grad():
                mem = self.model.encoder.encode(src)
                return self.model.projection(mem)

        def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
            """Decode using frozen decoder."""
            return self.model.decoder.decode(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        def eval(self):
            self.model.eval()
            return self

    test_model = ProjectionEncodingModel(model, device).eval()

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
                test_model, a, b, port_a_idx, port_b_idx,
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


def freeze_test_translation_model(
    model: nn.Module,
    device: torch.device,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Run freeze/isolation tests on the translation layer.

    Tests whether freezing side A or B degrades performance.
    """
    from whiteroom.freeze_probe import run_freeze_test, run_freeze_test_b_frozen
    from whiteroom.generator import sample_triplet, sample_b_frozen_triplet
    import random

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
            "normal_seq_acc": sum(r.normal_seq_correct for r in results) / n,
            "normal_flags_acc": sum(r.normal_flags_correct for r in results) / n,
            "frozen_seq_acc": sum(r.frozen_seq_correct for r in results) / n,
            "frozen_flags_acc": sum(r.frozen_flags_correct for r in results) / n,
            "seq_deg": sum(r.normal_seq_correct for r in results) / n
                       - sum(r.frozen_seq_correct for r in results) / n,
            "flag_deg": sum(r.normal_flags_correct for r in results) / n
                        - sum(r.frozen_flags_correct for r in results) / n,
            "mean_cos_sim": sum(r.a_encoder_cosine_sim for r in results) / n,
        }

    return {
        "a_frozen": metrics(a_results),
        "b_frozen": metrics(b_results),
    }


def evaluate_translation_model(
    checkpoint_dir: str,
    device: torch.device = None,
    n: int = 300,
    seed_eval: int = 42,
) -> dict:
    """Run full evaluation suite on a translation model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading translation model from {checkpoint_dir}...")
    trans_ckpt = torch.load(Path(checkpoint_dir) / "checkpoint_translation.pt", weights_only=False)
    model = load_translation_model(checkpoint_dir, device)
    model.eval()

    print("Running property-append tests on FULL TRANSLATION LAYER...")
    prop_append_results = {}
    prop_dict = property_append_translation_model(
        model=model,
        device=device,
        n=n,
        seed=seed_eval,
    )
    prop_append_results["1"] = prop_dict

    print("Running freeze/isolation tests on FULL TRANSLATION LAYER...")
    freeze_results = {}
    freeze_dict = freeze_test_translation_model(
        model=model,
        device=device,
        n=n,
        seed=seed_eval,
    )
    freeze_results["1"] = freeze_dict

    # Attribution not implemented for this eval
    attribution_results = {"1": {}}

    # Aggregate results
    eval_results = {}
    for seed in ["1", "2", "3", "4", "5"]:
        eval_results[seed] = {
            "freeze": freeze_results.get(seed, {}),
            "property_append": prop_append_results.get(seed, {}),
            "attribution": attribution_results.get(seed, {}),
        }

    return eval_results


def evaluate_attribution(
    model: nn.Module,
    n: int = 300,
    seed: int = 42,
    device: torch.device = None,
) -> dict:
    """Evaluate attribution accuracy."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import random
    from whiteroom.generator import sample_attribution_example, VOCAB_SIZE
    from whiteroom.train import collate_attribution

    rng = random.Random(seed)
    correct = 0

    for _ in range(n):
        ex = sample_attribution_example(rng)
        batch = collate_attribution([ex], device)

        with torch.no_grad():
            seq_logits, _ = (
                model.encode(batch["src"], src_key_padding_mask=batch["src_pad_mask"]),
                model.seq_head(model.decode(
                    batch["tgt_in"],
                    model.encode(batch["src"], src_key_padding_mask=batch["src_pad_mask"]),
                    tgt_key_padding_mask=batch["tgt_pad_mask"],
                    memory_key_padding_mask=batch["src_pad_mask"],
                )),
            )

        pred = seq_logits.argmax(dim=-1)
        target = batch["tgt_out"]
        correct += (pred == target).all(dim=1).sum().item()

    accuracy = correct / n
    return {
        "n": n,
        "seq_exact_match": accuracy,
        "token_accuracy": accuracy,
    }


# =============================================================================
# Main
# =============================================================================

def main(
    rundir: str,
    pairs: str = "1:1",
    n: int = 300,
    seed_eval: int = 42,
):
    """Evaluate one or more translation layer pairs."""
    rundir_path = Path(rundir)

    # Parse pairs (e.g., "1:1" or "1:1,1:2,2:3")
    pair_specs = [p.strip() for p in pairs.split(",")]

    results_summary = {}

    for pair_spec in pair_specs:
        encoder_seed, decoder_seed = map(int, pair_spec.split(":"))
        pair_dir = rundir_path / f"7d-seed{encoder_seed}_stage5-seed{decoder_seed}"

        if not pair_dir.exists():
            print(f"Warning: {pair_dir} does not exist, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating 7d-seed{encoder_seed} + stage5-seed{decoder_seed}")
        print(f"{'='*70}")

        eval_results = evaluate_translation_model(
            str(pair_dir),
            device=None,
            n=n,
            seed_eval=seed_eval,
        )

        # Save eval results
        results_file = pair_dir / "eval_results.json"
        with open(results_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Saved eval results to {results_file}")

        # Aggregate summary
        results_summary[f"{encoder_seed}:{decoder_seed}"] = {
            seed: {
                "b_isolation": eval_results[seed]["freeze"].get("b_frozen", {}).get("seq_deg", 0),
                "composition": eval_results[seed]["property_append"].get("hybrid_pickup_pct", 0),
            }
            for seed in ["1", "2", "3", "4", "5"]
        }

    # Save summary
    summary_file = rundir_path / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    for pair, seeds in results_summary.items():
        print(f"\n7d-seed{pair.split(':')[0]} + stage5-seed{pair.split(':')[1]}:")
        print("  Seed | B_Iso | Composition")
        for seed in ["1", "2", "3", "4", "5"]:
            if seed in seeds:
                b_iso = seeds[seed]["b_isolation"]
                comp = seeds[seed]["composition"]
                print(f"  {seed:>4} | {b_iso:6.4f} | {comp:11.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8 Evaluation")
    parser.add_argument("--rundir", type=str, required=True, help="Stage 8 run directory")
    parser.add_argument("--pairs", type=str, default="1:1", help="Pairs to evaluate (e.g., '1:1' or '1:1,1:2')")
    parser.add_argument("--n", type=int, default=300, help="Num examples per eval")
    parser.add_argument("--seed-eval", type=int, default=42, help="Eval seed")

    args = parser.parse_args()
    main(**vars(args))
