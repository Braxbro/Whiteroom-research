"""
Stage 6 memory manipulation probes — four zero-cost eval tests.

All probes operate on pre-computed encoder memory tensors; no new training.
Each probe loads a checkpoint once and runs all four tests in sequence.

Probes:
  1. duplicate_a  — replace B's positions with A's representations (A in both slots)
  2. shuffle_a    — randomly permute token order within A's segment
  3. corruption   — add Gaussian noise scaled to repr norm; sweep sigma levels
  4. bind_dir     — split pairs by binding direction (A.out→B.in vs A.in←B.out);
                    compare freeze accuracy across directions

Usage:
    python probe_memory_ops.py --checkpoints <path1> [<path2> ...]
    python probe_memory_ops.py --rundir _agent/cache/runs/stage5 --subdir-prefix stage5 --seeds 1,2,3,4,5
    python probe_memory_ops.py --rundir _agent/cache/runs/stage5 --subdir-prefix stage5 --seeds 1,2,3,4,5 \\
        --rundir2 _agent/cache/runs/stage2 --subdir-prefix2 stage2
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whiteroom.model import WhiteroomTransformer
from whiteroom.freeze_probe import make_example_for_ab, _greedy_from_memory
from whiteroom.generator import serialize_entity, sample_primitive
from whiteroom.composition import find_valid_bindings
from whiteroom.vocab import Token, port_idx_token


# ---------------------------------------------------------------------------
# Shared sampling
# ---------------------------------------------------------------------------

def sample_pair(rng, equal_length=False):
    """Sample (A, B, port_a_idx, port_b_idx, entity_len).
    If equal_length=True, only returns pairs where len(A_tokens)==len(B_tokens).
    """
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
        if equal_length and len(a_tokens) != len(b_tokens):
            continue
        port_a_idx, port_b_idx = rng.choice(bindings)
        return a, b, port_a_idx, port_b_idx, len(a_tokens), len(b_tokens)
    return None


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WhiteroomTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def trim(pred):
    try:
        return pred[:pred.index(Token.END) + 1]
    except ValueError:
        return pred


# ---------------------------------------------------------------------------
# Probe 1: Duplicate A in both slots
# ---------------------------------------------------------------------------

def probe_duplicate_a(model, device, n=300, seed=42):
    """
    Replace B's encoder output positions with A's representations.
    Memory layout: [A_reps | BIND | A_reps_again] instead of [A_reps | BIND | B_reps].

    Since compound(A,B)==compound(B,A) by spec, and A≠B, this is feeding the decoder
    two copies of A. Question: does it produce anything coherent? Does it hallucinate
    B's flags from A's content?

    Requires equal-length A and B so positions swap 1-for-1.
    """
    rng = random.Random(seed)
    normal_correct = 0
    duped_correct = 0
    agreement = 0
    sampled = 0

    while sampled < n:
        pair = sample_pair(rng, equal_length=True)
        if pair is None:
            break
        a, b, port_a_idx, port_b_idx, a_len, b_len = pair

        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        target = ex.target_tokens
        a_start, a_end = ex.a_token_span
        b_start, b_end = ex.b_token_span

        src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            mem = model.encode(src)

        mem_duped = mem.clone()
        # Replace B's positions with A's representations
        mem_duped[0, b_start:b_end, :] = mem[0, a_start:a_end, :]

        p_n = trim(_greedy_from_memory(model, mem, device, 32))
        p_d = trim(_greedy_from_memory(model, mem_duped, device, 32))

        normal_correct += int(p_n == target)
        duped_correct  += int(p_d == target)
        agreement      += int(p_n == p_d)
        sampled += 1

    return {
        "n": sampled,
        "normal_acc":  normal_correct / sampled,
        "duped_acc":   duped_correct  / sampled,
        "agreement":   agreement      / sampled,
        "cost":        (normal_correct - duped_correct) / sampled,
    }


# ---------------------------------------------------------------------------
# Probe 2: Shuffle tokens within A's segment
# ---------------------------------------------------------------------------

def probe_shuffle_a(model, device, n=300, seed=42):
    """
    Randomly permute the token order within A's encoder output span.
    BIND region and B are untouched. Tests whether the decoder relies on
    intra-segment token ordering within A.
    """
    rng = random.Random(seed)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(seed)

    normal_correct  = 0
    shuffle_correct = 0
    agreement       = 0
    sampled = 0

    while sampled < n:
        pair = sample_pair(rng, equal_length=False)
        if pair is None:
            break
        a, b, port_a_idx, port_b_idx, a_len, b_len = pair

        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        target = ex.target_tokens
        a_start, a_end = ex.a_token_span

        src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            mem = model.encode(src)

        mem_shuffled = mem.clone()
        seg_len = a_end - a_start
        perm = torch.randperm(seg_len, generator=torch_rng, device=device)
        mem_shuffled[0, a_start:a_end, :] = mem[0, a_start:a_end, :][perm]

        p_n = trim(_greedy_from_memory(model, mem, device, 32))
        p_s = trim(_greedy_from_memory(model, mem_shuffled, device, 32))

        normal_correct  += int(p_n == target)
        shuffle_correct += int(p_s == target)
        agreement       += int(p_n == p_s)
        sampled += 1

    return {
        "n": sampled,
        "normal_acc":   normal_correct  / sampled,
        "shuffle_acc":  shuffle_correct / sampled,
        "agreement":    agreement       / sampled,
        "cost":         (normal_correct - shuffle_correct) / sampled,
    }


# ---------------------------------------------------------------------------
# Probe 3: Content corruption (noise sweep)
# ---------------------------------------------------------------------------

CORRUPTION_SIGMAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def probe_corruption(model, device, n=300, seed=42, sigmas=None):
    """
    Add Gaussian noise to A's encoder output scaled to each token's L2 norm.
    noise = N(0, sigma^2 * ||repr||^2) per token.

    Sweep sigma levels; measure accuracy degradation curve.
    Corruption target: A's segment only (B and BIND untouched).
    """
    if sigmas is None:
        sigmas = CORRUPTION_SIGMAS

    rng = random.Random(seed)

    # Pre-sample pairs and encode once; apply different sigmas
    pairs_data = []
    temp_count = 0
    while temp_count < n:
        pair = sample_pair(rng, equal_length=False)
        if pair is None:
            break
        a, b, port_a_idx, port_b_idx, a_len, b_len = pair
        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        src = torch.tensor(ex.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            mem = model.encode(src)
        pairs_data.append((mem, ex.target_tokens, ex.a_token_span))
        temp_count += 1

    results_by_sigma = {}
    torch_rng = torch.Generator(device=device)

    for sigma in sigmas:
        torch_rng.manual_seed(seed)  # same noise seed for each sigma level
        correct = 0
        for mem, target, (a_start, a_end) in pairs_data:
            mem_c = mem.clone()
            if sigma > 0:
                seg = mem_c[0, a_start:a_end, :]  # (seg_len, d_model)
                norms = seg.norm(dim=-1, keepdim=True)  # (seg_len, 1)
                noise = torch.randn_like(seg, generator=torch_rng) * sigma * norms
                mem_c[0, a_start:a_end, :] = seg + noise
            with torch.no_grad():
                pred = trim(_greedy_from_memory(model, mem_c, device, 32))
            correct += int(pred == target)

        results_by_sigma[sigma] = correct / len(pairs_data)

    return {
        "n": len(pairs_data),
        "sigmas": sigmas,
        "accuracy_by_sigma": results_by_sigma,
    }


# ---------------------------------------------------------------------------
# Probe 4: Binding direction
# ---------------------------------------------------------------------------

def probe_bind_direction(model, device, n=300, seed=42):
    """
    Split pairs by binding direction:
      forward:  A has output port, B has input port  (A.out → B.in)
      reverse:  A has input port, B has output port  (A.in ← B.out)

    For each, run A-frozen test: freeze A's encoding from [A|BIND|B],
    decode against fresh [A|BIND|C] target.

    Tests whether freeze accuracy differs based on which entity is the
    "source" vs "sink" in the binding.
    """
    from whiteroom.freeze_probe import sample_triplet, run_freeze_test

    rng = random.Random(seed)
    forward_results = []  # A is output side
    reverse_results = []  # A is input side

    attempts = 0
    while len(forward_results) + len(reverse_results) < n and attempts < n * 20:
        attempts += 1
        t = sample_triplet(rng)
        if t is None:
            continue
        a, b, c, pa, pb, pc = t

        # Determine binding direction from A's perspective
        port_a = dict(a.ports)[pa]
        is_forward = port_a.is_output  # A provides output to B's input

        result = run_freeze_test(model, a, b, c, pa, pb, pc, device)

        if is_forward:
            forward_results.append(result)
        else:
            reverse_results.append(result)

    def summarize(results):
        if not results:
            return None
        n = len(results)
        return {
            "n": n,
            "normal_acc":  sum(r.normal_seq_correct for r in results) / n,
            "frozen_acc":  sum(r.frozen_seq_correct for r in results) / n,
            "cost":        sum(r.normal_seq_correct - r.frozen_seq_correct for r in results) / n,
            "mean_cos_sim": sum(r.a_encoder_cosine_sim for r in results) / n,
        }

    return {
        "forward": summarize(forward_results),
        "reverse": summarize(reverse_results),
    }


# ---------------------------------------------------------------------------
# Per-checkpoint runner
# ---------------------------------------------------------------------------

def evaluate_all(checkpoint_path: str, n: int = 300, seed: int = 42) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    print(f"  running duplicate_a...", flush=True)
    dup = probe_duplicate_a(model, device, n=n, seed=seed)

    print(f"  running shuffle_a...", flush=True)
    shuf = probe_shuffle_a(model, device, n=n, seed=seed)

    print(f"  running corruption...", flush=True)
    corr = probe_corruption(model, device, n=n, seed=seed)

    print(f"  running bind_direction...", flush=True)
    bdir = probe_bind_direction(model, device, n=n, seed=seed)

    return {
        "duplicate_a":    dup,
        "shuffle_a":      shuf,
        "corruption":     corr,
        "bind_direction": bdir,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_checkpoints(args):
    checkpoints = []
    if args.checkpoints:
        for p in args.checkpoints:
            checkpoints.append((Path(p).stem, p))
    if args.rundir and args.subdir_prefix:
        for s in args.seeds.split(","):
            p = Path(args.rundir) / f"{args.subdir_prefix}-seed{s}" / "checkpoint_final.pt"
            if p.exists():
                checkpoints.append((f"{args.subdir_prefix}-seed{s}", str(p)))
            else:
                print(f"[warn] not found: {p}")
    if hasattr(args, 'rundir2') and args.rundir2 and args.subdir_prefix2:
        for s in args.seeds.split(","):
            p = Path(args.rundir2) / f"{args.subdir_prefix2}-seed{s}" / "checkpoint_final.pt"
            if p.exists():
                checkpoints.append((f"{args.subdir_prefix2}-seed{s}", str(p)))
            else:
                print(f"[warn] not found: {p}")
    return checkpoints


def print_summary(label, r):
    d = r["duplicate_a"]
    s = r["shuffle_a"]
    c = r["corruption"]
    b = r["bind_direction"]

    print(f"\n[{label}]")
    print(f"  duplicate_a : normal={d['normal_acc']:.3f}  duped={d['duped_acc']:.3f}"
          f"  agree={d['agreement']:.3f}  cost={d['cost']:+.3f}")
    print(f"  shuffle_a   : normal={s['normal_acc']:.3f}  shuffled={s['shuffle_acc']:.3f}"
          f"  agree={s['agreement']:.3f}  cost={s['cost']:+.3f}")
    print(f"  corruption  : ", end="")
    for sigma, acc in c["accuracy_by_sigma"].items():
        print(f"σ={sigma}→{acc:.3f}  ", end="")
    print()
    if b["forward"] and b["reverse"]:
        print(f"  bind_dir    : forward n={b['forward']['n']} frozen={b['forward']['frozen_acc']:.3f}"
              f"  reverse n={b['reverse']['n']} frozen={b['reverse']['frozen_acc']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--rundir", type=str, default=None)
    parser.add_argument("--subdir-prefix", type=str, default=None)
    parser.add_argument("--rundir2", type=str, default=None)
    parser.add_argument("--subdir-prefix2", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed-eval", type=int, default=42)
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    checkpoints = collect_checkpoints(args)
    if not checkpoints:
        print("No checkpoints found.")
        sys.exit(1)

    results = {}
    for label, ckpt_path in checkpoints:
        print(f"\n{'='*50}\n{label}: {ckpt_path}")
        r = evaluate_all(ckpt_path, n=args.n, seed=args.seed_eval)
        results[label] = r
        print_summary(label, r)

    if args.outfile:
        # Convert tuple keys to strings for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            return obj
        with open(args.outfile, "w") as f:
            json.dump(make_serializable(results), f, indent=2)
        print(f"\nResults saved to {args.outfile}")


if __name__ == "__main__":
    main()
