"""
Oracle freeze mask analysis and dataset generation for sibling model training.

For each (old, new) pair — using the property-append setup — enumerates freeze
masks over the old positions and finds the most-frozen mask that achieves correct
flag output. The new extra-flag token is always fresh (never frozen).

Two levels:
  span_level   — 2^3 = 8 combinations over (A-span, BIND-span, B-span)
  position_level — 2^L individual positions, GPU-batched

The oracle dataset (old_tokens, new_tokens, optimal_mask) is the supervised
training signal for the sibling model.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch

from .entity import Entity, ARCHETYPES
from .composition import compose, find_valid_bindings
from .generator import serialize_entity, sample_primitive, VOCAB_SIZE
from .model import WhiteroomTransformer
from .vocab import Token, Flag, TRAINING_FLAGS, flag_token
from .freeze_probe import (
    make_example_for_ab, _greedy_from_memory, FLAG_TOKENS, _flag_tok,
)


# ---------------------------------------------------------------------------
# Core: build hybrid memory from a mask
# ---------------------------------------------------------------------------

def build_hybrid(
    mem_old: torch.Tensor,   # (1, L, d_model) — fully frozen base
    mem_new: torch.Tensor,   # (1, L+1, d_model) — fresh extended encoding
    mask: List[int],         # length-L binary vector; 1 = freeze (use old), 0 = fresh
) -> torch.Tensor:
    """
    Build hybrid memory of shape (1, L+1, d_model).
    Positions 0..L-1: use mem_old[i] if mask[i]==1, else mem_new[i].
    Position L (new extra token): always mem_new[L].
    """
    L = len(mask)
    hybrid = mem_new.clone()
    for i, freeze in enumerate(mask):
        if freeze:
            hybrid[0, i, :] = mem_old[0, i, :]
    return hybrid


def check_flags(pred: List[int], target_flags: frozenset) -> bool:
    pred_flags = frozenset(t for t in pred if t in FLAG_TOKENS)
    return pred_flags == target_flags


# ---------------------------------------------------------------------------
# Span-level oracle (8 combinations)
# ---------------------------------------------------------------------------

SPAN_NAMES = ("A", "BIND", "B")


@dataclass
class SpanOracleResult:
    # Span boundaries in old sequence
    a_span: Tuple[int, int]
    bind_span: Tuple[int, int]
    b_span: Tuple[int, int]

    # Per-combination results: key = (freeze_A, freeze_BIND, freeze_B) as tuple
    combo_results: Dict[Tuple[int,int,int], bool] = field(default_factory=dict)

    # Optimal: most-frozen combo achieving correct flags
    optimal_combo: Optional[Tuple[int,int,int]] = None
    optimal_freeze_rate: float = 0.0

    # Baselines
    freeze_all_correct: bool = False   # freeze all old positions (our existing test)
    freeze_none_correct: bool = False  # fully fresh (upper bound)

    extra_flag_tok: int = 0
    target_flags: frozenset = field(default_factory=frozenset)


def run_span_oracle(
    model: WhiteroomTransformer,
    a: Entity,
    b: Entity,
    port_a_idx: int,
    port_b_idx: int,
    extra_flag: Flag,
    device: torch.device,
) -> SpanOracleResult:
    model.eval()

    ex_ab = make_example_for_ab(a, b, port_a_idx, port_b_idx)
    a_start, a_end = ex_ab.a_token_span
    b_start, b_end = ex_ab.b_token_span
    bind_start, bind_end = a_end, b_start  # BIND + idx + idx

    L = len(ex_ab.input_tokens)
    extra_tok = flag_token(extra_flag)
    base_tokens = ex_ab.input_tokens
    extended_tokens = base_tokens + [extra_tok]

    target_flags = frozenset(
        _flag_tok(f) for f in (a.flags | b.flags | {extra_flag})
    )

    src_old = torch.tensor(base_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_new = torch.tensor(extended_tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        mem_old = model.encode(src_old)
        mem_new = model.encode(src_new)

    def make_mask(freeze_a, freeze_bind, freeze_b):
        mask = [0] * L
        if freeze_a:
            for i in range(a_start, a_end):
                mask[i] = 1
        if freeze_bind:
            for i in range(bind_start, bind_end):
                mask[i] = 1
        if freeze_b:
            for i in range(b_start, b_end):
                mask[i] = 1
        return mask

    result = SpanOracleResult(
        a_span=(a_start, a_end),
        bind_span=(bind_start, bind_end),
        b_span=(b_start, b_end),
        extra_flag_tok=extra_tok,
        target_flags=target_flags,
    )

    best_combo = None
    best_rate = -1.0

    for fa in (0, 1):
        for fb_ind in (0, 1):
            for fb in (0, 1):
                combo = (fa, fb_ind, fb)
                mask = make_mask(fa, fb_ind, fb)
                hybrid = build_hybrid(mem_old, mem_new, mask)
                with torch.no_grad():
                    pred = _greedy_from_memory(model, hybrid, device, 32)
                correct = check_flags(pred, target_flags)
                result.combo_results[combo] = correct

                if correct:
                    rate = sum(mask) / L
                    if rate > best_rate:
                        best_rate = rate
                        best_combo = combo

    result.optimal_combo = best_combo
    result.optimal_freeze_rate = best_rate

    # Baselines
    all_frozen_mask = [1] * L
    hybrid_all = build_hybrid(mem_old, mem_new, all_frozen_mask)
    with torch.no_grad():
        pred_all = _greedy_from_memory(model, hybrid_all, device, 32)
    result.freeze_all_correct = check_flags(pred_all, target_flags)

    none_frozen_mask = [0] * L
    hybrid_none = build_hybrid(mem_old, mem_new, none_frozen_mask)
    with torch.no_grad():
        pred_none = _greedy_from_memory(model, hybrid_none, device, 32)
    result.freeze_none_correct = check_flags(pred_none, target_flags)

    return result


# ---------------------------------------------------------------------------
# Position-level oracle (2^L, GPU-batched)
# ---------------------------------------------------------------------------

@dataclass
class PositionOracleResult:
    L: int
    # Binary mask over old positions; optimal = most frozen achieving correct flags
    optimal_mask: Optional[List[int]]
    optimal_freeze_rate: float
    # For each position: fraction of accurate masks that freeze it
    position_freeze_freq: List[float]  # length L
    # How many masks achieved correct flags (out of 2^L)
    n_accurate_masks: int
    total_masks: int
    # Span boundaries
    a_span: Tuple[int, int]
    b_span: Tuple[int, int]
    target_flags: frozenset


def run_position_oracle(
    model: WhiteroomTransformer,
    a: Entity,
    b: Entity,
    port_a_idx: int,
    port_b_idx: int,
    extra_flag: Flag,
    device: torch.device,
    batch_size: int = 512,
) -> Optional[PositionOracleResult]:
    """
    Enumerate all 2^L freeze masks over old positions.
    Batches decoder calls for speed. Skips sequences with L > 20 (>1M masks).
    """
    model.eval()

    ex_ab = make_example_for_ab(a, b, port_a_idx, port_b_idx)
    L = len(ex_ab.input_tokens)

    if L > 20:
        return None  # impractical

    extra_tok = flag_token(extra_flag)
    base_tokens = ex_ab.input_tokens
    extended_tokens = base_tokens + [extra_tok]

    target_flags = frozenset(
        _flag_tok(f) for f in (a.flags | b.flags | {extra_flag})
    )

    src_old = torch.tensor(base_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_new = torch.tensor(extended_tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        mem_old = model.encode(src_old)   # (1, L, d_model)
        mem_new = model.encode(src_new)   # (1, L+1, d_model)

    d_model = mem_old.size(-1)
    n_masks = 2 ** L

    # Build all 2^L hybrid memories: shape (n_masks, L+1, d_model)
    # For each mask_int, bit i of mask_int → freeze position i
    all_hybrids = mem_new.expand(n_masks, -1, -1).clone()  # (n_masks, L+1, d_model)

    # Apply freeze: for each position i, gather which masks freeze it
    for i in range(L):
        # masks that freeze position i: those where bit i is set
        freeze_these = torch.tensor(
            [1 if (m >> i) & 1 else 0 for m in range(n_masks)],
            dtype=torch.bool, device=device
        )  # (n_masks,)
        # Replace those rows' position i with mem_old
        old_vec = mem_old[0, i, :]  # (d_model,)
        all_hybrids[freeze_these, i, :] = old_vec.unsqueeze(0)

    # Decode in batches
    accurate = torch.zeros(n_masks, dtype=torch.bool)
    freeze_counts = torch.zeros(n_masks, dtype=torch.long)

    for m in range(n_masks):
        bits = sum(((m >> i) & 1) for i in range(L))
        freeze_counts[m] = bits

    for start in range(0, n_masks, batch_size):
        end = min(start + batch_size, n_masks)
        batch_mem = all_hybrids[start:end]  # (bs, L+1, d_model)

        # Decode each in the batch sequentially (greedy decode isn't easily batched
        # across variable-length outputs, so we loop)
        for j, mem in enumerate(batch_mem):
            with torch.no_grad():
                pred = _greedy_from_memory(model, mem.unsqueeze(0), device, 32)
            accurate[start + j] = check_flags(pred, target_flags)

    # Find optimal: most frozen among accurate masks
    accurate_indices = accurate.nonzero(as_tuple=True)[0]
    n_accurate = len(accurate_indices)

    optimal_mask = None
    optimal_freeze_rate = 0.0

    if n_accurate > 0:
        best_count = freeze_counts[accurate_indices].max().item()
        # Pick first mask with that count
        for idx in accurate_indices:
            if freeze_counts[idx].item() == best_count:
                m = idx.item()
                optimal_mask = [(m >> i) & 1 for i in range(L)]
                optimal_freeze_rate = best_count / L
                break

    # Per-position freeze frequency among accurate masks
    pos_freq = []
    for i in range(L):
        if n_accurate > 0:
            frozen_and_accurate = sum(
                1 for idx in accurate_indices if (idx.item() >> i) & 1
            )
            pos_freq.append(frozen_and_accurate / n_accurate)
        else:
            pos_freq.append(0.0)

    return PositionOracleResult(
        L=L,
        optimal_mask=optimal_mask,
        optimal_freeze_rate=optimal_freeze_rate,
        position_freeze_freq=pos_freq,
        n_accurate_masks=n_accurate,
        total_masks=n_masks,
        a_span=ex_ab.a_token_span,
        b_span=ex_ab.b_token_span,
        target_flags=target_flags,
    )


# ---------------------------------------------------------------------------
# Dataset generation for sibling model training
# ---------------------------------------------------------------------------

@dataclass
class OracleSample:
    old_tokens: List[int]
    new_tokens: List[int]       # old_tokens + [extra_flag_tok]
    optimal_mask: List[int]     # length L, binary; 1 = freeze
    optimal_freeze_rate: float
    a_span: Tuple[int, int]
    b_span: Tuple[int, int]
    extra_flag_tok: int
    has_accurate_mask: bool     # False if no mask achieves correct flags
    # Span-level label: (freeze_A, freeze_BIND, freeze_B); None if no accurate mask.
    # For unsolvable cases, (1,1,1) is stored as the safe default.
    span_combo: Tuple[int, int, int] = (1, 1, 1)


def generate_oracle_dataset(
    checkpoint_path: str,
    n_samples: int = 1000,
    seed: int = 42,
    use_position_level: bool = False,
    batch_size: int = 256,
) -> List[OracleSample]:
    """
    Generate oracle dataset: (old_tokens, new_tokens) → optimal_mask.

    Uses span-level oracle by default (fast, interpretable).
    Set use_position_level=True for full position-level enumeration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = WhiteroomTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = random.Random(seed)
    samples: List[OracleSample] = []

    while len(samples) < n_samples:
        # Sample valid A+B pair
        for _ in range(50):
            a = sample_primitive(rng)
            b = sample_primitive(rng)
            bindings = find_valid_bindings(a, b)
            if bindings:
                break
        else:
            continue

        port_a_idx, port_b_idx = rng.choice(bindings)
        combined_flags = a.flags | b.flags
        available = [f for f in TRAINING_FLAGS if f not in combined_flags]
        if not available:
            continue
        extra_flag = rng.choice(available)

        if use_position_level:
            result = run_position_oracle(
                model, a, b, port_a_idx, port_b_idx, extra_flag, device, batch_size
            )
            if result is None:
                continue
            optimal_mask = result.optimal_mask or [0] * result.L
            freeze_rate = result.optimal_freeze_rate
            has_accurate = result.n_accurate_masks > 0
            a_span = result.a_span
            b_span = result.b_span
        else:
            result = run_span_oracle(
                model, a, b, port_a_idx, port_b_idx, extra_flag, device
            )
            ex_ab2 = make_example_for_ab(a, b, port_a_idx, port_b_idx)
            L = len(ex_ab2.input_tokens)
            a_s, a_e = ex_ab2.a_token_span
            b_s, b_e = ex_ab2.b_token_span
            if result.optimal_combo is None:
                span_combo = (1, 1, 1)  # safe default for unsolvable
                optimal_mask = [1] * L
                has_accurate = False
                freeze_rate = 0.0
            else:
                span_combo = result.optimal_combo
                fa, fb_ind, fb = span_combo
                mask = [0] * L
                if fa:
                    for i in range(a_s, a_e):
                        mask[i] = 1
                if fb_ind:
                    for i in range(a_e, b_s):
                        mask[i] = 1
                if fb:
                    for i in range(b_s, b_e):
                        mask[i] = 1
                optimal_mask = mask
                freeze_rate = result.optimal_freeze_rate
                has_accurate = True
            a_span = result.a_span
            b_span = result.b_span

        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        base_tokens = ex.input_tokens
        extra_tok = flag_token(extra_flag)

        samples.append(OracleSample(
            old_tokens=base_tokens,
            new_tokens=base_tokens + [extra_tok],
            optimal_mask=optimal_mask,
            optimal_freeze_rate=freeze_rate,
            a_span=a_span,
            b_span=b_span,
            extra_flag_tok=extra_tok,
            has_accurate_mask=has_accurate,
            span_combo=span_combo,
        ))

    return samples


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def summarize_span_oracle(results: List[SpanOracleResult]) -> dict:
    """Aggregate statistics over a list of span oracle results."""
    n = len(results)
    if n == 0:
        return {}

    # Fraction of pairs where freeze_all achieves correct flags
    freeze_all_acc = sum(r.freeze_all_correct for r in results) / n
    # Fraction where fully fresh achieves correct flags
    fresh_acc = sum(r.freeze_none_correct for r in results) / n
    # Fraction where any mask achieves correct flags
    any_acc = sum(r.optimal_combo is not None for r in results) / n

    # Optimal combo distribution
    from collections import Counter
    combo_counts = Counter(
        r.optimal_combo for r in results if r.optimal_combo is not None
    )
    combo_labels = {
        (0,0,0): "freeze_none",
        (1,0,0): "freeze_A",
        (0,1,0): "freeze_BIND",
        (0,0,1): "freeze_B",
        (1,1,0): "freeze_A+BIND",
        (1,0,1): "freeze_A+B",
        (0,1,1): "freeze_BIND+B",
        (1,1,1): "freeze_all",
    }

    combo_dist = {
        combo_labels.get(k, str(k)): v / n
        for k, v in combo_counts.most_common()
    }

    mean_optimal_rate = sum(
        r.optimal_freeze_rate for r in results if r.optimal_combo is not None
    ) / max(sum(r.optimal_combo is not None for r in results), 1)

    return {
        "n": n,
        "freeze_all_accuracy": freeze_all_acc,
        "full_fresh_accuracy": fresh_acc,
        "any_mask_achieves_accuracy": any_acc,
        "mean_optimal_freeze_rate": mean_optimal_rate,
        "optimal_combo_distribution": combo_dist,
    }


def run_span_oracle_experiment(
    checkpoint_path: str,
    n_pairs: int = 500,
    seed: int = 42,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = WhiteroomTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = random.Random(seed)
    results: List[SpanOracleResult] = []

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
        combined_flags = a.flags | b.flags
        available = [f for f in TRAINING_FLAGS if f not in combined_flags]
        if not available:
            continue
        extra_flag = rng.choice(available)

        results.append(run_span_oracle(
            model, a, b, port_a_idx, port_b_idx, extra_flag, device
        ))

    return summarize_span_oracle(results)
