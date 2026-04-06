"""
Evaluate Stage 8d/8e using standard eval_multiseed.py approach.

Stage 8 checkpoints have a different format (projection_state + encoder/decoder paths).
This script loads them into a working model and evaluates using the standard test suite.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whiteroom.freeze_probe import (
    sample_triplet, sample_b_frozen_triplet, run_freeze_test, run_freeze_test_b_frozen,
    run_freeze_test_property_append
)
from whiteroom.composition import find_valid_bindings
from whiteroom.generator import sample_primitive
from whiteroom.vocab import TRAINING_FLAGS
from whiteroom.model import WhiteroomTransformer
from whiteroom.generator import sample_attribution_example, VOCAB_SIZE
from whiteroom.vocab import Token


class Stage8Model(nn.Module):
    """Wrapper that combines frozen encoder + projection + frozen decoder."""

    def __init__(self, encoder, projection, decoder):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.decoder = decoder
        self.seq_head = encoder.seq_head
        self.valid_head = encoder.valid_head

    def encode(self, src, src_key_padding_mask=None):
        """Encode with frozen encoder."""
        mem = self.encoder.encode(src, src_key_padding_mask)
        return self.projection(mem)

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Decode with frozen decoder."""
        return self.decoder.decode(tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)


def load_stage8_model(ckpt_path, device):
    """Load Stage 8 checkpoint (projection + encoder/decoder paths)."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Valid WhiteroomTransformer parameters
    valid_params = {
        'vocab_size', 'd_model', 'nhead', 'num_encoder_layers', 'num_decoder_layers',
        'dim_feedforward', 'dropout', 'max_seq_len', 'causal_encoder'
    }

    # Load encoder from its checkpoint
    encoder_ckpt_path = ckpt["encoder_ckpt"]
    encoder_ckpt = torch.load(encoder_ckpt_path, map_location=device)
    encoder_config = {k: v for k, v in encoder_ckpt["config"].items() if k in valid_params}
    encoder = WhiteroomTransformer(**encoder_config).to(device)
    encoder.load_state_dict(encoder_ckpt["model_state"])
    encoder.eval()
    encoder.requires_grad_(False)

    # Load decoder from its checkpoint
    decoder_ckpt_path = ckpt["decoder_ckpt"]
    decoder_ckpt = torch.load(decoder_ckpt_path, map_location=device)
    decoder_config = {k: v for k, v in decoder_ckpt["config"].items() if k in valid_params}
    decoder = WhiteroomTransformer(**decoder_config).to(device)
    decoder.load_state_dict(decoder_ckpt["model_state"])
    decoder.eval()
    decoder.requires_grad_(False)

    # Build projection from config
    proj_config = ckpt["projection_config"]
    d_model = proj_config.get("d_model", 64)
    proj_type = proj_config.get("type", "layernorm_linear")

    if proj_type in ("layernorm_linear", "linear"):
        # TranslationProjection: LayerNorm + Linear
        class TranslationProjection(nn.Module):
            def __init__(self, d_in=64, d_out=64):
                super().__init__()
                self.norm = nn.LayerNorm(d_in)
                self.linear = nn.Linear(d_in, d_out, bias=True)
            def forward(self, x):
                return self.linear(self.norm(x))

        projection = TranslationProjection(d_model, d_model)
    elif proj_type == "mlp":
        # MLPProjection: LayerNorm + 2-layer MLP
        class MLPProjection(nn.Module):
            def __init__(self, d_model=64):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.fc1 = nn.Linear(d_model, d_model * 4)
                self.fc2 = nn.Linear(d_model * 4, d_model)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(self.norm(x))))

        projection = MLPProjection(d_model)
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")

    projection.load_state_dict(ckpt["projection_state"])
    projection.to(device)
    projection.eval()
    projection.requires_grad_(False)

    # Combine into Stage8Model
    model = Stage8Model(encoder, projection, decoder)
    model.eval()

    return model


def evaluate_attribution(checkpoint_path: str, n_samples: int = 300, seed: int = 77) -> dict:
    """Evaluate attribution accuracy on Stage 8 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_stage8_model(checkpoint_path, device)

    rng = random.Random(seed)
    seq_correct = 0
    total_tokens = 0
    correct_tokens = 0

    for _ in range(n_samples):
        ex = sample_attribution_example(rng)

        src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        tgt = ex.target_tokens

        bos = torch.tensor([[Token.COMPOUND]], dtype=torch.long, device=device)

        with torch.no_grad():
            mem = model.encode(src)

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

        min_len = min(len(pred), len(tgt))
        correct_tokens += sum(pred[i] == tgt[i] for i in range(min_len))
        total_tokens += len(tgt)

        if pred == tgt:
            seq_correct += 1

    return {
        "n": n_samples,
        "seq_exact_match": seq_correct / n_samples,
        "token_accuracy": correct_tokens / total_tokens if total_tokens > 0 else 0.0,
    }


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
    parser.add_argument("--stage", choices=["8d", "8e"], required=True, help="Stage 8d or 8e")
    parser.add_argument("--rundir", default="_agent/cache/runs/stage8")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed-eval", type=int, default=1234)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rundir = Path(args.rundir)
    outfile = rundir / f"eval_results_{args.stage}_standard.json"

    all_results = {}

    for seed in seeds:
        ckpt = rundir / f"{args.stage}-linear-unfreeze-seed{seed}" / "checkpoint_translation.pt"
        if args.stage == "8e":
            ckpt = rundir / f"{args.stage}-mlp-unfreeze-seed{seed}" / "checkpoint_translation.pt"

        if not ckpt.exists():
            print(f"[seed {seed}] checkpoint not found: {ckpt}", flush=True)
            continue

        print(f"\n[seed {seed}] evaluating {ckpt.name} ...", flush=True)

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_stage8_model(str(ckpt), device)

        # 1. Cache freeze
        print(f"  cache freeze test ({args.n} triplets) ...", flush=True)
        try:
            rng = random.Random(args.seed_eval)
            a_results, b_results = [], []

            for _ in range(args.n):
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
            freeze = {
                "a_frozen": {
                    "n": a_metrics.get("n", 0),
                    "normal_seq_acc": a_metrics.get("normal", 0),
                    "frozen_seq_acc": a_metrics.get("frozen", 0),
                    "seq_deg": a_metrics.get("degradation", 0),
                },
                "b_frozen": {
                    "n": b_metrics.get("n", 0),
                    "normal_seq_acc": b_metrics.get("normal", 0),
                    "frozen_seq_acc": b_metrics.get("frozen", 0),
                    "seq_deg": b_metrics.get("degradation", 0),
                }
            }
        except Exception as e:
            print(f"  ERROR in freeze test: {e}")
            freeze = {}

        # 2. Property-append
        print(f"  property-append test ({args.n} pairs) ...", flush=True)
        try:
            rng = random.Random(args.seed_eval)
            results = []

            for i in range(args.n):
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

                # Pick a flag not already in A or B
                combined_flags = a.flags | b.flags
                available = [f for f in TRAINING_FLAGS if f not in combined_flags]
                if not available:
                    continue
                extra_flag = rng.choice(available)
                target_side = 'a' if i % 2 == 0 else 'b'

                results.append(run_freeze_test_property_append(
                    model, a, b, port_a_idx, port_b_idx,
                    extra_flag, target_side, device,
                ))

            n = len(results)
            if n > 0:
                hybrid_pickup = sum(r.hybrid_has_extra for r in results) / n
                fresh_pickup = sum(r.full_fresh_has_extra for r in results) / n
                base_contamination = sum(r.frozen_only_has_extra for r in results) / n
                a_pres = sum(r.a_flags_preserved for r in results if r.a_flags_preserved is not None) / n
                b_pres = sum(r.b_flags_preserved for r in results if r.b_flags_preserved is not None) / n

                a_side = [r for r in results if r.target_side == 'a']
                b_side = [r for r in results if r.target_side == 'b']

                prop = {
                    "n": n,
                    "hybrid_pickup_pct": hybrid_pickup,
                    "full_fresh_pickup_pct": fresh_pickup,
                    "base_contamination": base_contamination,
                    "a_flags_preserved_pct": a_pres,
                    "b_flags_preserved_pct": b_pres,
                    "hybrid_pickup_a_side": sum(r.hybrid_has_extra for r in a_side) / len(a_side) if a_side else None,
                    "hybrid_pickup_b_side": sum(r.hybrid_has_extra for r in b_side) / len(b_side) if b_side else None,
                }
            else:
                prop = {}
        except Exception as e:
            print(f"  ERROR in property-append: {e}")
            import traceback
            traceback.print_exc()
            prop = {}

        # 3. Attribution
        print(f"  attribution eval ({args.n} samples) ...", flush=True)
        try:
            attr = evaluate_attribution(str(ckpt), n_samples=args.n, seed=args.seed_eval)
        except Exception as e:
            print(f"  ERROR in attribution: {e}")
            attr = {}

        all_results[seed] = {
            "freeze": freeze,
            "property_append": prop,
            "attribution": attr,
        }

        # Per-seed summary
        af = freeze.get("a_frozen", {})
        bf = freeze.get("b_frozen", {})
        print(f"  [cache freeze]  A-frozen deg {af.get('seq_deg',0):.4f}  "
              f"B-frozen deg {bf.get('seq_deg',0):.4f}")
        print(f"  [prop-append]   hybrid pickup {prop.get('hybrid_pickup_pct',0):.4f}")
        print(f"  [attribution]   seq-exact {attr.get('seq_exact_match',0):.4f}  "
              f"tok-acc {attr.get('token_accuracy',0):.4f}")

    # Aggregate
    if len(all_results) < 2:
        print("\nNot enough seeds to aggregate.")
    else:
        print("\n" + "=" * 70)
        print(f"AGGREGATE (mean ± std across seeds) — {args.stage}")
        print("=" * 70)

        def collect(fn):
            return [fn(all_results[s]) for s in sorted(all_results)]

        af_deg = collect(lambda r: r["freeze"].get("a_frozen", {}).get("seq_deg", float("nan")))
        bf_deg = collect(lambda r: r["freeze"].get("b_frozen", {}).get("seq_deg", float("nan")))
        pa_hyb = collect(lambda r: r["property_append"].get("hybrid_pickup_pct", float("nan")))
        at_seq = collect(lambda r: r["attribution"].get("seq_exact_match", float("nan")))
        at_tok = collect(lambda r: r["attribution"].get("token_accuracy", float("nan")))

        print(f"\nIsolation:")
        print(f"  A-frozen degradation {fmt(*mean_std(af_deg))}")
        print(f"  B-frozen degradation {fmt(*mean_std(bf_deg))}")

        print(f"\nComposition:")
        print(f"  hybrid pickup {fmt(*mean_std(pa_hyb))}")

        print(f"\nAttribution:")
        print(f"  seq exact match {fmt(*mean_std(at_seq))}")
        print(f"  token accuracy {fmt(*mean_std(at_tok))}")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
