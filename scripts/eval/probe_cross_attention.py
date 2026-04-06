"""
Stage 6 cross-attention pattern probe.

Decodes from normal and position-spliced encoder memory while capturing
decoder cross-attention weights at every layer and decoding step.

Core question: when A and B's encoder outputs are physically swapped in memory,
do the decoder's cross-attention patterns:
  (a) stay fixed at original positions  → positional routing
  (b) shift to follow the content       → semantic routing

We already know output is identical (agreement_rate=1.000). This tells us HOW.

Method:
  For each equal-length (A, B) pair:
    1. Encode [A|BIND|B] → mem_normal
    2. Create mem_spliced: swap A and B encoder output positions
    3. Decode greedily from both, capturing cross-attention weights per layer per step
    4. For each decoder step, compute attention mass on A-region vs B-region
    5. Compare: in spliced, does mass shift to follow content (semantic) or stay (positional)?

Metrics per layer, aggregated over steps and samples:
  normal_a_mass    : attention mass on A positions in normal decode
  normal_b_mass    : attention mass on B positions in normal decode
  spliced_a_mass   : attention mass on positions [a_start:a_end] in spliced decode
                     (these positions now contain B's representations)
  spliced_b_mass   : attention mass on positions [b_start:b_end] in spliced decode
                     (these positions now contain A's representations)

  content_follow_score : how much mass followed A's content to its new location
    = spliced_b_mass / (spliced_b_mass + spliced_a_mass)
    1.0 = purely semantic (attention follows A's content to B's positions)
    0.0 = purely positional (attention stays at A's original positions)

Also saves a few raw attention map examples to disk for visualization.

Usage:
    python probe_cross_attention.py --rundir _agent/cache/runs/stage5 \\
        --subdir-prefix stage5 --seeds 1,2,3,4,5
    python probe_cross_attention.py --checkpoints path/to/checkpoint.pt
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whiteroom.model import WhiteroomTransformer
from whiteroom.freeze_probe import make_example_for_ab, _greedy_from_memory
from whiteroom.generator import serialize_entity, sample_primitive
from whiteroom.composition import find_valid_bindings
from whiteroom.vocab import Token


# ---------------------------------------------------------------------------
# Cross-attention capture via monkey-patching
# ---------------------------------------------------------------------------

class AttentionCapture:
    """
    Context manager that patches each decoder layer's cross-attention to
    capture weights with need_weights=True, average_attn_weights=False
    (per-head, shape: batch × n_heads × tgt_len × src_len).

    Usage:
        capture = AttentionCapture(model)
        with capture:
            pred = _greedy_from_memory(model, mem, device, max_len)
        # capture.weights[step][layer] = (1, n_heads, 1, src_len)
    """

    def __init__(self, model: WhiteroomTransformer):
        self.model = model
        self.n_layers = len(model.transformer.decoder.layers)
        # weights[step_idx][layer_idx] = tensor(1, n_heads, 1, src_len)
        self.weights: List[Dict[int, torch.Tensor]] = []
        self._step_buf: Dict[int, torch.Tensor] = {}
        self._patches: List = []

    def __enter__(self):
        self.weights = []
        self._step_buf = {}
        self._patches = []

        for i, layer in enumerate(self.model.transformer.decoder.layers):
            orig_forward = layer.multihead_attn.forward

            def patched(q, k, v, *args, _i=i, _orig=orig_forward, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out, w = _orig(q, k, v, *args, **kwargs)
                # w: (batch, n_heads, tgt_len, src_len) — tgt_len=1 during greedy
                self._step_buf[_i] = w.detach().cpu()
                return out, w

            layer.multihead_attn.forward = patched
            self._patches.append((layer, orig_forward))

        # Hook greedy loop: flush _step_buf after each token step
        # We do this by wrapping _greedy_from_memory externally; instead,
        # we expose a flush() method the caller calls per step.
        return self

    def flush_step(self):
        """Call after each decoder step to snapshot the current layer weights."""
        if self._step_buf:
            self.weights.append(dict(self._step_buf))
            self._step_buf = {}

    def __exit__(self, *args):
        for layer, orig in self._patches:
            layer.multihead_attn.forward = orig


# ---------------------------------------------------------------------------
# Greedy decode with per-step attention capture
# ---------------------------------------------------------------------------

def greedy_with_attention(
    model: WhiteroomTransformer,
    memory: torch.Tensor,
    device: torch.device,
    capture: AttentionCapture,
    max_len: int = 32,
) -> List[int]:
    """
    Greedy decode from memory, calling capture.flush_step() after each token.
    Returns predicted token ids (without BOS).
    """
    batch = memory.size(0)
    ys = torch.full((batch, 1), Token.COMPOUND, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_len = ys.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
        with torch.no_grad():
            dec_out = model.decode(ys, memory, tgt_mask=tgt_mask)
            next_tok = model.seq_head(dec_out[:, -1, :]).argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        capture.flush_step()
        if next_tok.item() == Token.END:
            break

    return ys[0, 1:].cpu().tolist()


# ---------------------------------------------------------------------------
# Attention mass computation
# ---------------------------------------------------------------------------

def region_mass(weights_per_step: List[Dict[int, torch.Tensor]],
                region_start: int, region_end: int,
                n_layers: int) -> Dict[int, float]:
    """
    For each layer, compute mean attention mass on [region_start:region_end]
    averaged over all decoding steps and all heads.

    weights_per_step[step][layer] shape: (1, n_heads, 1, src_len)
    Returns {layer_idx: mean_mass}
    """
    totals = {l: 0.0 for l in range(n_layers)}
    counts = {l: 0   for l in range(n_layers)}

    for step_weights in weights_per_step:
        for layer_idx, w in step_weights.items():
            # w: (1, n_heads, 1, src_len)
            w_np = w[0, :, 0, :]  # (n_heads, src_len)
            mass = w_np[:, region_start:region_end].sum(dim=-1).mean().item()
            totals[layer_idx] += mass
            counts[layer_idx] += 1

    return {l: totals[l] / counts[l] if counts[l] > 0 else 0.0
            for l in range(n_layers)}


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def sample_equal_length_pair(rng):
    for _ in range(500):
        a = sample_primitive(rng)
        b = sample_primitive(rng)
        if a is b:
            continue
        bindings = find_valid_bindings(a, b)
        if not bindings:
            continue
        a_tokens, _ = serialize_entity(a)
        b_tokens, _ = serialize_entity(b)
        if len(a_tokens) != len(b_tokens):
            continue
        port_a_idx, port_b_idx = rng.choice(bindings)
        return a, b, port_a_idx, port_b_idx, len(a_tokens)
    return None


def evaluate_attention(checkpoint_path: str, n: int = 100, seed: int = 42,
                       save_examples: int = 3) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WhiteroomTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_layers = len(model.transformer.decoder.layers)
    rng = random.Random(seed)

    # Accumulators: per layer, normal and spliced region masses
    # normal_a_mass[l]: attention mass on A positions, normal decode
    # spliced_a_mass[l]: attention mass on A positions, spliced decode (A's content gone)
    # spliced_b_mass[l]: attention mass on B positions, spliced decode (A's content here now)
    acc_normal_a  = {l: 0.0 for l in range(n_layers)}
    acc_normal_b  = {l: 0.0 for l in range(n_layers)}
    acc_spliced_a = {l: 0.0 for l in range(n_layers)}  # old A positions, now B content
    acc_spliced_b = {l: 0.0 for l in range(n_layers)}  # old B positions, now A content
    sampled = 0
    agreement = 0

    raw_examples = []

    while sampled < n:
        pair = sample_equal_length_pair(rng)
        if pair is None:
            break
        a, b, port_a_idx, port_b_idx, entity_len = pair

        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        target = ex.target_tokens
        a_start, a_end = ex.a_token_span
        b_start, b_end = ex.b_token_span

        src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            mem = model.encode(src)

        mem_spliced = mem.clone()
        mem_spliced[0, a_start:a_end, :] = mem[0, b_start:b_end, :]
        mem_spliced[0, b_start:b_end, :] = mem[0, a_start:a_end, :]

        # Decode normal with attention capture
        cap_normal = AttentionCapture(model)
        with cap_normal:
            pred_normal = greedy_with_attention(model, mem, device, cap_normal)

        # Decode spliced with attention capture
        cap_spliced = AttentionCapture(model)
        with cap_spliced:
            pred_spliced = greedy_with_attention(model, mem_spliced, device, cap_spliced)

        def trim(pred):
            try: return pred[:pred.index(Token.END) + 1]
            except ValueError: return pred

        p_n = trim(pred_normal)
        p_s = trim(pred_spliced)
        agreement += int(p_n == p_s)

        # Compute attention mass per region per layer
        nm_a = region_mass(cap_normal.weights,  a_start, a_end, n_layers)
        nm_b = region_mass(cap_normal.weights,  b_start, b_end, n_layers)
        sm_a = region_mass(cap_spliced.weights, a_start, a_end, n_layers)  # B content now here
        sm_b = region_mass(cap_spliced.weights, b_start, b_end, n_layers)  # A content now here

        for l in range(n_layers):
            acc_normal_a[l]  += nm_a[l]
            acc_normal_b[l]  += nm_b[l]
            acc_spliced_a[l] += sm_a[l]
            acc_spliced_b[l] += sm_b[l]

        # Save raw attention maps for first few examples
        if sampled < save_examples:
            raw_examples.append({
                "sample_idx": sampled,
                "a_span": [a_start, a_end],
                "b_span": [b_start, b_end],
                "seq_len": src.size(1),
                "n_decode_steps_normal":  len(cap_normal.weights),
                "n_decode_steps_spliced": len(cap_spliced.weights),
                # Store mean attention map over steps per layer (n_heads x src_len)
                "normal_attn_by_layer": {
                    str(l): _mean_attn(cap_normal.weights, l).tolist()
                    for l in range(n_layers)
                },
                "spliced_attn_by_layer": {
                    str(l): _mean_attn(cap_spliced.weights, l).tolist()
                    for l in range(n_layers)
                },
                "pred_normal":  p_n,
                "pred_spliced": p_s,
                "target":       target,
            })

        sampled += 1

    # Aggregate
    layers = {}
    for l in range(n_layers):
        na  = acc_normal_a[l]  / sampled
        nb  = acc_normal_b[l]  / sampled
        sa  = acc_spliced_a[l] / sampled  # old A pos, now B content
        sb  = acc_spliced_b[l] / sampled  # old B pos, now A content

        # content_follow: in spliced, fraction of (A-region + B-region) mass
        # that went to B positions (where A's content now lives)
        # 1.0 = fully semantic, 0.0 = fully positional
        ab_total = sa + sb
        content_follow = sb / ab_total if ab_total > 1e-9 else float("nan")

        layers[l] = {
            "normal_a_mass":     na,
            "normal_b_mass":     nb,
            "spliced_old_a_mass": sa,   # attention to old A positions (now B content)
            "spliced_old_b_mass": sb,   # attention to old B positions (now A content)
            "content_follow_score": content_follow,
        }

    return {
        "n": sampled,
        "agreement_rate": agreement / sampled,
        "n_layers": n_layers,
        "layers": layers,
        "raw_examples": raw_examples,
    }


def _mean_attn(weights_per_step: List[Dict[int, torch.Tensor]], layer_idx: int) -> torch.Tensor:
    """Mean attention map over all steps for one layer. Shape: (n_heads, src_len)."""
    maps = [w[layer_idx][0, :, 0, :] for w in weights_per_step if layer_idx in w]
    if not maps:
        return torch.zeros(1, 1)
    return torch.stack(maps).mean(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--rundir", type=str, default=None)
    parser.add_argument("--subdir-prefix", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--n", type=int, default=100,
                        help="Samples per checkpoint (fewer needed; maps are rich)")
    parser.add_argument("--seed-eval", type=int, default=42)
    parser.add_argument("--save-examples", type=int, default=3,
                        help="Number of raw attention map examples to save per checkpoint")
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    checkpoints = []
    if args.checkpoints:
        checkpoints = [(Path(p).stem, p) for p in args.checkpoints]
    elif args.rundir and args.subdir_prefix:
        for s in args.seeds.split(","):
            p = Path(args.rundir) / f"{args.subdir_prefix}-seed{s}" / "checkpoint_final.pt"
            if p.exists():
                checkpoints.append((f"seed{s}", str(p)))
            else:
                print(f"[warn] not found: {p}")

    if not checkpoints:
        print("No checkpoints found.")
        sys.exit(1)

    all_results = {}
    for label, ckpt_path in checkpoints:
        print(f"\n{'='*50}\n{label}: evaluating {args.n} samples...", flush=True)
        r = evaluate_attention(ckpt_path, n=args.n, seed=args.seed_eval,
                               save_examples=args.save_examples)

        print(f"  agreement_rate: {r['agreement_rate']:.3f}  (n={r['n']})")
        print(f"  {'Layer':<6} {'norm_A':>8} {'norm_B':>8} "
              f"{'spl_oldA':>10} {'spl_oldB':>10} {'content_follow':>15}")
        print(f"  {'-'*60}")
        for l, v in r["layers"].items():
            print(f"  {l:<6} {v['normal_a_mass']:>8.3f} {v['normal_b_mass']:>8.3f} "
                  f"{v['spliced_old_a_mass']:>10.3f} {v['spliced_old_b_mass']:>10.3f} "
                  f"{v['content_follow_score']:>15.3f}")

        all_results[label] = r

    if args.outfile:
        with open(args.outfile, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults (including raw attention maps) saved to {args.outfile}")
    else:
        # Print summary without raw maps
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "raw_examples"}
                   for k, v in all_results.items()}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
