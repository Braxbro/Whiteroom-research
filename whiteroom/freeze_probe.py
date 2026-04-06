"""
KV cache freezing experiment (Phase 5).

Two symmetric freeze tests:

  A-frozen: triplet (A, B, C) where C swaps in for B, A's positions are frozen.
    - Normal:  encode([A|BIND|C]) → decode → predict compound(A,C)
    - Frozen:  A positions from encode([A|BIND|B]) + BIND+C from encode([A|BIND|C])
               → decode → predict compound(A,C)

  B-frozen: triplet (D, A, B) where D swaps in for A, B's positions are frozen.
    - Normal:  encode([D|BIND|B]) → decode → predict compound(D,B)
    - Frozen:  D+BIND from encode([D|BIND|B]) + B positions from encode([A|BIND|B])
               → decode → predict compound(D,B)

Ideal result: zero degradation in both directions — the model can't tell which
component is frozen, because it learned full bidirectional semantic independence.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .entity import Entity, ARCHETYPES
from .composition import compose, find_valid_bindings
from .generator import (
    serialize_entity, serialize_compound_output,
    sample_primitive, VOCAB_SIZE,
)
from .model import WhiteroomTransformer
from .vocab import Token, Flag, port_idx_token
from .train import collate
from .generator import Example


# ---------------------------------------------------------------------------
# Compliant swap sampling
# ---------------------------------------------------------------------------

def find_compliant_swaps(
    a: Entity,
    b: Entity,
    port_a_idx: int,
    port_b_idx: int,
    candidates: List[Entity],
) -> List[Tuple[Entity, int]]:
    """
    Find all (C, port_c_idx) from candidates where C can bind to A at port_a_idx
    with a port compatible with port_a_idx on A, AND C differs from B in flags or op_type.

    The binding port on C must accept the same type as port_b_idx on B accepted
    (i.e. same compatibility with A's output port).
    """
    port_a = dict(a.ports)[port_a_idx]
    port_b = dict(b.ports)[port_b_idx]

    result = []
    for c in candidates:
        if c is b:
            continue
        for ic, pc in c.ports:
            # Must be compatible binding with port_a
            if port_a.is_output and pc.is_input and port_a.compatible_with(pc):
                compatible = True
            elif port_a.is_input and pc.is_output and pc.compatible_with(port_a):
                compatible = True
            else:
                continue

            # Must differ from B in flags or op_type (compliant but distinct)
            if c.flags != b.flags or c.op_types != b.op_types:
                result.append((c, ic))

    return result


def sample_triplet(
    rng: random.Random,
) -> Optional[Tuple[Entity, Entity, Entity, int, int, int]]:
    """
    Sample (A, B, C, port_a_idx, port_b_idx, port_c_idx) where:
      - A+B form a valid compound via (port_a_idx, port_b_idx)
      - C is a compliant swap for B (compatible binding with A, different flags/op)

    Used for A-frozen test: A stays, B→C swap.
    Returns None if no valid triplet found.
    """
    candidates = [arch.to_entity() for arch in ARCHETYPES]

    for _ in range(100):
        a = sample_primitive(rng)
        b = sample_primitive(rng)
        bindings = find_valid_bindings(a, b)
        if not bindings:
            continue
        port_a_idx, port_b_idx = rng.choice(bindings)
        swaps = find_compliant_swaps(a, b, port_a_idx, port_b_idx, candidates)
        if not swaps:
            continue
        c, port_c_idx = rng.choice(swaps)
        return a, b, c, port_a_idx, port_b_idx, port_c_idx

    return None


def sample_b_frozen_triplet(
    rng: random.Random,
) -> Optional[Tuple[Entity, Entity, Entity, int, int, int]]:
    """
    Sample (A, D, B, port_a_idx, port_d_idx, port_b_idx) where:
      - A+B form a valid compound via (port_a_idx, port_b_idx)
      - D is a compliant swap for A (compatible binding with B, different flags/op)

    Used for B-frozen test: B stays, A→D swap.
    Returns None if no valid triplet found.
    """
    candidates = [arch.to_entity() for arch in ARCHETYPES]

    for _ in range(100):
        a = sample_primitive(rng)
        b = sample_primitive(rng)
        bindings = find_valid_bindings(a, b)
        if not bindings:
            continue
        port_a_idx, port_b_idx = rng.choice(bindings)
        # Swap roles: find D compliant with B at port_b_idx (D replaces A)
        swaps = find_compliant_swaps(b, a, port_b_idx, port_a_idx, candidates)
        if not swaps:
            continue
        d, port_d_idx = rng.choice(swaps)
        return a, d, b, port_a_idx, port_d_idx, port_b_idx

    return None


# ---------------------------------------------------------------------------
# Frozen forward pass
# ---------------------------------------------------------------------------

def make_example_for_ac(
    a: Entity, c: Entity, port_a_idx: int, port_c_idx: int
) -> Example:
    """Build an Example for compound(A, C)."""
    from .composition import compose
    compound = compose(a, c, port_a_idx, port_c_idx)
    a_tokens, a_map = serialize_entity(a)
    c_tokens, c_map = serialize_entity(c)
    rel_a = a_map[port_a_idx]
    rel_c = c_map[port_c_idx]
    a_start, a_end = 0, len(a_tokens)
    b_start = a_end + 3
    b_end = b_start + len(c_tokens)
    input_tokens = a_tokens + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_c)] + c_tokens
    return Example(
        input_tokens=input_tokens,
        target_tokens=serialize_compound_output(compound),
        is_valid=True,
        entity_a=a,
        entity_b=c,
        compound=compound,
        a_token_span=(a_start, a_end),
        b_token_span=(b_start, b_end),
    )


def make_example_for_ab(
    a: Entity, b: Entity, port_a_idx: int, port_b_idx: int
) -> Example:
    """Build an Example for compound(A, B) — used only to get A's encoder output."""
    from .composition import compose
    compound = compose(a, b, port_a_idx, port_b_idx)
    a_tokens, a_map = serialize_entity(a)
    b_tokens, b_map = serialize_entity(b)
    rel_a = a_map[port_a_idx]
    rel_b = b_map[port_b_idx]
    a_start, a_end = 0, len(a_tokens)
    b_start = a_end + 3
    b_end = b_start + len(b_tokens)
    input_tokens = a_tokens + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_b)] + b_tokens
    return Example(
        input_tokens=input_tokens,
        target_tokens=serialize_compound_output(compound),
        is_valid=True,
        entity_a=a,
        entity_b=b,
        compound=compound,
        a_token_span=(a_start, a_end),
        b_token_span=(b_start, b_end),
    )


@dataclass
class FreezeResult:
    # Normal forward on [A|BIND|C]
    normal_seq_correct: bool
    normal_flags_correct: bool
    # Frozen forward: A's encoder output from [A|BIND|B], rest from [A|BIND|C]
    frozen_seq_correct: bool
    frozen_flags_correct: bool
    # Delta in A's encoder output between AB and AC runs (cosine similarity)
    a_encoder_cosine_sim: float
    # Ground truth
    target_tokens: List[int]
    a_flags: List[int]  # flag tokens attributable to A


FLAG_TOKENS = {
    Token.FLAG_SHOOTS, Token.FLAG_ILLUMINATES, Token.FLAG_SCANS,
    Token.FLAG_BROADCASTS, Token.FLAG_ATTRACTS, Token.FLAG_SPAWNS,
}


def run_freeze_test(
    model: WhiteroomTransformer,
    a: Entity,
    b: Entity,
    c: Entity,
    port_a_idx: int,
    port_b_idx: int,
    port_c_idx: int,
    device: torch.device,
) -> FreezeResult:
    model.eval()

    ex_ac = make_example_for_ac(a, c, port_a_idx, port_c_idx)
    ex_ab = make_example_for_ab(a, b, port_a_idx, port_b_idx)

    a_len = ex_ac.a_token_span[1]  # same for both (A is unchanged)

    # Encode both
    src_ac = torch.tensor(ex_ac.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_ab = torch.tensor(ex_ab.input_tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        mem_ac = model.encode(src_ac)  # (1, ac_len, d_model)
        mem_ab = model.encode(src_ab)  # (1, ab_len, d_model)

        # Cosine similarity between A's encoder outputs in AB vs AC runs
        a_vec_ab = mem_ab[0, :a_len, :].mean(0)
        a_vec_ac = mem_ac[0, :a_len, :].mean(0)
        cos_sim = torch.nn.functional.cosine_similarity(
            a_vec_ab.unsqueeze(0), a_vec_ac.unsqueeze(0)
        ).item()

        # Normal decode: use mem_ac as-is
        tgt_len = 32
        normal_pred = _greedy_from_memory(model, mem_ac, device, tgt_len)

        # Frozen decode: replace A's positions in mem_ac with mem_ab's A positions
        frozen_mem = mem_ac.clone()
        frozen_mem[0, :a_len, :] = mem_ab[0, :a_len, :]
        frozen_pred = _greedy_from_memory(model, frozen_mem, device, tgt_len)

    target = ex_ac.target_tokens
    a_flags = [t for t in target if t in FLAG_TOKENS and t in
               {_flag_tok(f) for f in a.flags}]

    def seq_correct(pred):
        try:
            pred = pred[:pred.index(Token.END) + 1]
        except ValueError:
            pass
        return pred == target

    def flags_correct(pred):
        pred_flags = {t for t in pred if t in FLAG_TOKENS}
        tgt_flags = {t for t in target if t in FLAG_TOKENS}
        return pred_flags == tgt_flags

    def a_flags_correct(pred):
        pred_flags = {t for t in pred if t in FLAG_TOKENS}
        return all(t in pred_flags for t in a_flags)

    return FreezeResult(
        normal_seq_correct=seq_correct(normal_pred),
        normal_flags_correct=flags_correct(normal_pred),
        frozen_seq_correct=seq_correct(frozen_pred),
        frozen_flags_correct=flags_correct(frozen_pred),
        a_encoder_cosine_sim=cos_sim,
        target_tokens=target,
        a_flags=a_flags,
    )


def _greedy_from_memory(
    model: WhiteroomTransformer,
    memory: torch.Tensor,
    device: torch.device,
    max_len: int,
) -> List[int]:
    """Greedy decode from a pre-computed memory tensor."""
    batch = memory.size(0)
    ys = torch.full((batch, 1), Token.COMPOUND, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_len = ys.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
        with torch.no_grad():
            dec_out = model.decode(ys, memory, tgt_mask=tgt_mask)
            next_tok = model.seq_head(dec_out[:, -1, :]).argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if next_tok.item() == Token.END:
            break
    return ys[0, 1:].cpu().tolist()


def _flag_tok(flag):
    from .vocab import flag_token
    return flag_token(flag)


def run_freeze_test_b_frozen(
    model: WhiteroomTransformer,
    a: Entity,
    d: Entity,
    b: Entity,
    port_a_idx: int,
    port_d_idx: int,
    port_b_idx: int,
    device: torch.device,
) -> FreezeResult:
    """
    B-frozen test: B stays fixed, A is swapped for D.
    Freeze B's encoder output (taken from [A|BIND|B]) when predicting compound(D,B).

    Normal:  encode([D|BIND|B]) → decode → compound(D,B)
    Frozen:  D+BIND positions from encode([D|BIND|B]),
             B positions from encode([A|BIND|B])
             → decode → compound(D,B)
    """
    model.eval()

    ex_db = make_example_for_ac(d, b, port_d_idx, port_b_idx)  # compound(D,B)
    ex_ab = make_example_for_ab(a, b, port_a_idx, port_b_idx)  # compound(A,B) — for B's cached output

    b_start_db = ex_db.b_token_span[0]
    b_end_db   = ex_db.b_token_span[1]
    b_start_ab = ex_ab.b_token_span[0]
    b_end_ab   = ex_ab.b_token_span[1]
    b_len = b_end_db - b_start_db  # B has same token length in both (same entity)

    src_db = torch.tensor(ex_db.input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_ab = torch.tensor(ex_ab.input_tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        mem_db = model.encode(src_db)
        mem_ab = model.encode(src_ab)

        # Cosine sim: B's encoder output in AB vs DB runs
        b_vec_ab = mem_ab[0, b_start_ab:b_end_ab, :].mean(0)
        b_vec_db = mem_db[0, b_start_db:b_end_db, :].mean(0)
        cos_sim = torch.nn.functional.cosine_similarity(
            b_vec_ab.unsqueeze(0), b_vec_db.unsqueeze(0)
        ).item()

        normal_pred = _greedy_from_memory(model, mem_db, device, 32)

        # Frozen: replace B's positions in mem_db with B's positions from mem_ab
        frozen_mem = mem_db.clone()
        frozen_mem[0, b_start_db:b_end_db, :] = mem_ab[0, b_start_ab:b_end_ab, :]
        frozen_pred = _greedy_from_memory(model, frozen_mem, device, 32)

    target = ex_db.target_tokens
    b_flags = [t for t in target if t in FLAG_TOKENS and t in
               {_flag_tok(f) for f in b.flags}]

    def seq_correct(pred):
        try:
            pred = pred[:pred.index(Token.END) + 1]
        except ValueError:
            pass
        return pred == target

    def flags_correct(pred):
        return {t for t in pred if t in FLAG_TOKENS} == {t for t in target if t in FLAG_TOKENS}

    return FreezeResult(
        normal_seq_correct=seq_correct(normal_pred),
        normal_flags_correct=flags_correct(normal_pred),
        frozen_seq_correct=seq_correct(frozen_pred),
        frozen_flags_correct=flags_correct(frozen_pred),
        a_encoder_cosine_sim=cos_sim,
        target_tokens=target,
        a_flags=b_flags,
    )


# ---------------------------------------------------------------------------
# Stupider freeze test: freeze BOTH A and B, append one fresh flag to one side
# ---------------------------------------------------------------------------
#
# Setup:
#   base:     [A_tokens | BIND | rel_a | rel_b | B_tokens]
#   extended: [A_tokens | BIND | rel_a | rel_b | B_tokens | extra_flag_token]
#
# The extra_flag is a training flag NOT already in the target entity's flag set.
# It is appended after B's last token — no positional shift for A or B spans.
# We can test A-side by modifying whose "logical owner" the flag belongs to,
# but we always insert it at the sequence end to avoid positional encoding
# mismatches in the frozen B span. Two sub-conditions track which side the
# flag was "intended for":
#   extend_b: extra flag should propagate if model reads it as B's property
#   extend_a: same mechanics, but conceptually attributed to A (flag appended
#             at A's end, before BIND — B positions shift by 1, that's the
#             "stupid" part)
#
# Three decodes:
#   frozen_only  — decode from base memory (no extra token)
#   hybrid       — frozen base + fresh extra_flag appended
#   full_fresh   — fully fresh encoding of extended sequence
#
# Ground truth: if extra_flag is picked up, it appears in output flag set.
# The compound's expected flags = A.flags | B.flags | {extra_flag}.

@dataclass
class PropertyAppendResult:
    extra_flag_tok: int          # the flag token appended
    target_side: str             # 'a' or 'b' — which entity "owns" the flag logically
    # frozen-only baseline
    frozen_only_flags: frozenset
    frozen_only_has_extra: bool
    # hybrid: frozen base + fresh extra token
    hybrid_flags: frozenset
    hybrid_has_extra: bool
    # full-fresh encoding of extended sequence
    full_fresh_flags: frozenset
    full_fresh_has_extra: bool
    # flag preservation of original A and B flags in hybrid
    a_flags_preserved: bool      # all original A flags still in hybrid output
    b_flags_preserved: bool      # all original B flags still in hybrid output


def _original_compound_flags(a: Entity, b: Entity) -> frozenset:
    """Flag tokens the compound(A,B) should contain (union of A and B flags)."""
    return frozenset(_flag_tok(f) for f in (a.flags | b.flags))


def run_freeze_test_property_append(
    model: WhiteroomTransformer,
    a: Entity,
    b: Entity,
    port_a_idx: int,
    port_b_idx: int,
    extra_flag: "Flag",
    target_side: str,
    device: torch.device,
) -> PropertyAppendResult:
    """
    Stupider freeze test: freeze both A and B, append one fresh flag token.

    The extra_flag token is appended at the END of B's serialization
    (position b_end in the sequence) — no positional shift for A or B spans.
    Logical ownership (target_side) is tracked for analysis but doesn't change
    the mechanics.

    For A-side ownership: same token placement, different semantic claim.
    """
    from .vocab import flag_token as _ft

    model.eval()

    ex_ab = make_example_for_ab(a, b, port_a_idx, port_b_idx)
    b_end = ex_ab.b_token_span[1]
    base_tokens = ex_ab.input_tokens

    extra_tok = _ft(extra_flag)
    extended_tokens = base_tokens + [extra_tok]

    src_base = torch.tensor(base_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_ext  = torch.tensor(extended_tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        mem_base = model.encode(src_base)   # (1, base_len, d_model)
        mem_ext  = model.encode(src_ext)    # (1, base_len+1, d_model)

        frozen_only_pred = _greedy_from_memory(model, mem_base, device, 32)

        # hybrid: full frozen base + fresh extra_flag at position b_end
        hybrid_mem = torch.cat([
            mem_base,                          # frozen A+BIND+B (positions 0..base_len-1)
            mem_ext[:, b_end:b_end+1, :],      # fresh extra_flag (position base_len in ext)
        ], dim=1)
        hybrid_pred = _greedy_from_memory(model, hybrid_mem, device, 32)

        full_fresh_pred = _greedy_from_memory(model, mem_ext, device, 32)

    def has_flag(seq, tok):
        return tok in seq

    orig_a_flag_toks = frozenset(_ft(f) for f in a.flags)
    orig_b_flag_toks = frozenset(_ft(f) for f in b.flags)

    def all_flags_present(seq, flag_toks):
        return all(t in seq for t in flag_toks)

    fo_flags = frozenset(t for t in frozen_only_pred if t in FLAG_TOKENS)
    hy_flags = frozenset(t for t in hybrid_pred if t in FLAG_TOKENS)
    ff_flags = frozenset(t for t in full_fresh_pred if t in FLAG_TOKENS)

    return PropertyAppendResult(
        extra_flag_tok=extra_tok,
        target_side=target_side,
        frozen_only_flags=fo_flags,
        frozen_only_has_extra=has_flag(frozen_only_pred, extra_tok),
        hybrid_flags=hy_flags,
        hybrid_has_extra=has_flag(hybrid_pred, extra_tok),
        full_fresh_flags=ff_flags,
        full_fresh_has_extra=has_flag(full_fresh_pred, extra_tok),
        a_flags_preserved=all_flags_present(hybrid_pred, orig_a_flag_toks),
        b_flags_preserved=all_flags_present(hybrid_pred, orig_b_flag_toks),
    )


def run_experiment_property_append(
    checkpoint_path: str,
    n_pairs: int = 500,
    seed: int = 1234,
) -> dict:
    """
    Run the stupider freeze test: freeze both A and B, append a fresh flag.

    Tests both 'a' and 'b' side ownership (interleaved).
    """
    from .vocab import Flag, TRAINING_FLAGS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Detect model type and instantiate appropriate class
    config = ckpt["config"].copy()
    model_type = config.pop("model_type", "2stage")
    for key in ("sawtooth_encoder",):
        config.pop(key, None)

    if model_type == "3stage":
        from .model import WhiteroomTransformer3Stage
        model = WhiteroomTransformer3Stage(**config).to(device)
    else:
        model = WhiteroomTransformer(**config).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = random.Random(seed)
    results: List[PropertyAppendResult] = []

    for i in range(n_pairs):
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
    if n == 0:
        return {}

    # Main question: does the hybrid output include the extra flag?
    hybrid_pickup   = sum(r.hybrid_has_extra for r in results) / n
    fresh_pickup    = sum(r.full_fresh_has_extra for r in results) / n
    # Sanity: base should never have the extra flag (it wasn't in A or B)
    base_contamination = sum(r.frozen_only_has_extra for r in results) / n

    # Flag preservation (existing A+B flags survive in hybrid)
    a_pres = sum(r.a_flags_preserved for r in results if r.a_flags_preserved is not None) / n
    b_pres = sum(r.b_flags_preserved for r in results if r.b_flags_preserved is not None) / n

    # Break down by side
    a_side = [r for r in results if r.target_side == 'a']
    b_side = [r for r in results if r.target_side == 'b']

    return {
        "n": n,
        "hybrid_pickup_pct":     hybrid_pickup,
        "full_fresh_pickup_pct": fresh_pickup,
        "base_contamination":    base_contamination,
        "a_flags_preserved_pct": a_pres,
        "b_flags_preserved_pct": b_pres,
        "hybrid_pickup_a_side":  sum(r.hybrid_has_extra for r in a_side) / len(a_side) if a_side else None,
        "hybrid_pickup_b_side":  sum(r.hybrid_has_extra for r in b_side) / len(b_side) if b_side else None,
    }


# ---------------------------------------------------------------------------
# Run experiment across a checkpoint
# ---------------------------------------------------------------------------

def run_experiment(
    checkpoint_path: str,
    n_triplets: int = 500,
    seed: int = 1234,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Detect model type and instantiate appropriate class
    config = ckpt["config"].copy()
    model_type = config.pop("model_type", "2stage")
    for key in ("sawtooth_encoder",):
        config.pop(key, None)

    if model_type == "3stage":
        from .model import WhiteroomTransformer3Stage
        model = WhiteroomTransformer3Stage(**config).to(device)
    else:
        model = WhiteroomTransformer(**config).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = random.Random(seed)
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

    return {"a_frozen": metrics(a_results), "b_frozen": metrics(b_results)}
