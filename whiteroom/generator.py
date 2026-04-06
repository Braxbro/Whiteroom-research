"""
Generator: samples entity pairs with valid (or invalid) bindings, serializes
them as token sequences, and computes labels programmatically from the
composition function. No human annotation required.

Data generation is infinite — call sample_example() in a loop during training.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .entity import Entity, Archetype, ARCHETYPES, HOLDOUT_ARCHETYPES, ALL_ARCHETYPES, N_ARCHETYPES, HOLDOUT_ARCHETYPE_IDS
from .composition import compose, find_valid_bindings
from .vocab import (
    Token, PortType, OpType, Flag,
    port_token, op_token, flag_token, port_idx_token,
    VOCAB_SIZE_BASE,
)



# ---------------------------------------------------------------------------
# Archetype token offset
# Each archetype gets its own token ID starting after the base vocab.
# ---------------------------------------------------------------------------

ARCH_TOKEN_OFFSET = VOCAB_SIZE_BASE  # archetype i → token (ARCH_TOKEN_OFFSET + i)

def arch_token(archetype_id: int) -> int:
    return ARCH_TOKEN_OFFSET + archetype_id

N_ALL_ARCHETYPES = len(ALL_ARCHETYPES)
VOCAB_SIZE = ARCH_TOKEN_OFFSET + N_ALL_ARCHETYPES  # includes holdout archetype tokens


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _port_tokens(port) -> List[int]:
    """Emit token(s) for a single port."""
    if port.is_output:
        pt = next(iter(port.type_set))
        return [port_token(pt, "out")]
    else:
        return [port_token(t, "in") for t in sorted(port.type_set, key=lambda x: x.value)]


def serialize_primitive(entity: Entity) -> Tuple[List[int], Dict[int, int]]:
    """
    Returns (tokens, port_index_map) where port_index_map maps the entity's
    internal port index → relative position (0-based) in the serialized port list.
    Format: [arch_id_token] [port_tokens...] [op_token] [flag_tokens...]
    """
    tokens: List[int] = [arch_token(entity.archetype_id)]
    port_index_map: Dict[int, int] = {}
    rel_pos = 0
    for entity_idx, port in sorted(entity.ports, key=lambda x: x[0]):
        port_index_map[entity_idx] = rel_pos
        tokens.extend(_port_tokens(port))
        rel_pos += 1
    tokens.append(op_token(entity.op_types[0]))
    for flag in sorted(entity.flags, key=lambda x: x.value):
        tokens.append(flag_token(flag))
    return tokens, port_index_map


def serialize_entity(entity: Entity) -> Tuple[List[int], Dict[int, int]]:
    """
    Returns (tokens, port_index_map).

    Primitive: [arch_id] [port_tokens...] [op_token] [flag_tokens...]
    Compound:  [COMPOUND] [A tokens] [BIND port_idx_A port_idx_B] [B tokens] [END]

    Binding references use PORT_IDX tokens (relative positions within each
    component's serialized port list), not raw entity port indices.
    """
    if entity.is_primitive:
        return serialize_primitive(entity)

    a_tokens, a_map = serialize_entity(entity.component_a)
    b_tokens, b_map = serialize_entity(entity.component_b)
    port_a_idx, port_b_idx = entity.binding

    rel_a = a_map[port_a_idx]
    rel_b = b_map[port_b_idx]

    tokens = (
        [Token.COMPOUND]
        + a_tokens
        + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_b)]
        + b_tokens
        + [Token.END]
    )

    # Build port_index_map for the compound's surviving external ports
    # A's surviving ports keep their a_map entries; B's are offset
    b_offset = max(idx for idx, _ in entity.component_a.ports) + 1
    compound_map: Dict[int, int] = {}
    rel_pos = 0
    for entity_idx, _ in sorted(entity.ports, key=lambda x: x[0]):
        compound_map[entity_idx] = rel_pos
        rel_pos += 1

    return tokens, compound_map


def serialize_compound_output(entity: Entity) -> List[int]:
    """
    Serialize the compound's external signature as the prediction target.
    Format: [port_tokens...] [op_tokens...] [flag_tokens...] [END]
    Op tokens emitted per component in order (A's first, then B's).
    """
    tokens: List[int] = []
    for _, port in sorted(entity.ports, key=lambda x: x[0]):
        tokens.extend(_port_tokens(port))
    for op in entity.op_types:
        tokens.append(op_token(op))
    for flag in sorted(entity.flags, key=lambda x: x.value):
        tokens.append(flag_token(flag))
    tokens.append(Token.END)
    return tokens


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def balanced_archetype_weights(archetype_pool=None) -> List[float]:
    """
    Compute per-archetype sampling weights that:
      1. Equalise per-flag appearance rate across all training flags.
      2. Strongly downweight zero-flag archetypes (reduce zero-flag compound bias).

    Analytically: every flagged archetype gets weight 1.0; zero-flag archetypes
    get weight 0.25. This keeps all six flag rates equal while cutting zero-flag
    compound frequency from ~25% to ~8%.
    """
    pool = archetype_pool if archetype_pool is not None else ARCHETYPES
    from .vocab import Flag as _Flag
    training_flags = {f for f in _Flag if f not in (_Flag.HOLDOUT1, _Flag.HOLDOUT2)}
    weights = []
    for arch in pool:
        arch_flags = {f for f in arch.flags if f in training_flags}
        weights.append(0.25 if len(arch_flags) == 0 else 1.0)
    return weights


def sample_primitive(
    rng: random.Random = random,
    archetype_pool=None,
    weights=None,
) -> Entity:
    pool = archetype_pool if archetype_pool is not None else ARCHETYPES
    if weights is not None:
        return rng.choices(pool, weights=weights, k=1)[0].to_entity()
    return rng.choice(pool).to_entity()


# Flag pairs that co-occur within the same archetype — these create compound-level
# co-occurrence biases because entities carrying both flags are always sampled together.
# Used by the co-occurrence dampening option in sample_example.
_COOCCURRING_FLAG_PAIRS: frozenset = frozenset([
    (Flag.SCANS, Flag.ILLUMINATES),      # arch 3
    (Flag.ATTRACTS, Flag.SPAWNS),        # arch 10
    (Flag.SHOOTS, Flag.ILLUMINATES),     # arch 11
])


def _flags_cooccur_across(a: Entity, b: Entity) -> bool:
    """True if a and b together contain a co-occurring flag pair split across them."""
    for f1, f2 in _COOCCURRING_FLAG_PAIRS:
        if ((f1 in a.flags and f2 in b.flags) or
                (f2 in a.flags and f1 in b.flags)):
            return True
    return False


def sample_entity(
    rng: random.Random = random,
    depth: int = 0,
    max_depth: int = 2,
    archetype_pool=None,
    weights=None,
) -> Entity:
    """Recursively sample a primitive or compound entity."""
    if depth >= max_depth or rng.random() < 0.6:
        return sample_primitive(rng, archetype_pool, weights)
    for _ in range(20):
        a = sample_entity(rng, depth + 1, max_depth, archetype_pool, weights)
        b = sample_entity(rng, depth + 1, max_depth, archetype_pool, weights)
        if _is_forbidden_combination(a, b):
            continue
        bindings = find_valid_bindings(a, b)
        if bindings:
            port_a, port_b = rng.choice(bindings)
            return compose(a, b, port_a, port_b)
    return sample_primitive(rng, archetype_pool, weights)


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """
    A single training example.

    input_tokens:  [A tokens] [BIND port_idx_A port_idx_B] [B tokens]
    target_tokens: [port_tokens...] [op_tokens...] [flag_tokens...] [END]
                   (or just [END] for invalid examples)
    is_valid:      ground-truth binding validity label
    entity_a/b:    source entities (used for KV cache probing in Phase 5)
    compound:      composed result (None if invalid)
    a_token_span:  (start, end) slice of input_tokens covering A's serialization
    b_token_span:  (start, end) slice of input_tokens covering B's serialization
    """
    input_tokens: List[int]
    target_tokens: List[int]
    is_valid: bool
    entity_a: Entity
    entity_b: Entity
    compound: Optional[Entity]
    a_token_span: Tuple[int, int]
    b_token_span: Tuple[int, int]


def sample_example(
    rng: random.Random = random,
    invalid_prob: float = 0.2,
    max_depth: int = 2,
    archetype_pool=None,
    weights=None,
    cooccurrence_damp: float = 0.0,
) -> Example:
    """
    weights: per-archetype sampling weights (see balanced_archetype_weights()).
    cooccurrence_damp: probability of rejecting and resampling a valid pair
        where A and B together contain a co-occurring flag pair split across them
        (e.g. A has SCANS, B has BROADCASTS). Set to ~0.7 to break compound-level
        flag co-occurrence biases without eliminating them entirely.
    """
    if rng.random() > invalid_prob:
        return _sample_valid(rng, max_depth, archetype_pool, weights, cooccurrence_damp)
    else:
        return _sample_invalid(rng, max_depth, archetype_pool, weights)


# ---------------------------------------------------------------------------
# Combination holdout — forbidden (archetype_a_id, archetype_b_id) pairs.
# The model sees all archetypes and all token types, but never these specific
# pairings during training. Tests whether it learned abstract composition rules.
# ---------------------------------------------------------------------------

HOLDOUT_COMBINATIONS: frozenset = frozenset([
    (0,  5),   # FLUX→PULSE  +  BLOOM→FLUX
    (1,  3),   # PULSE→WAVE  +  SURGE→TRACE
    (2,  8),   # WAVE→DRIFT  +  TRACE+BLOOM→SURGE
    (3,  1),   # SURGE→TRACE +  PULSE→WAVE
    (4, 11),   # DRIFT→STATIC + PULSE→DRIFT+STATIC
    (5,  7),   # BLOOM→FLUX  +  FLUX+PULSE→WAVE
    (6,  4),   # STATIC→PULSE + DRIFT→STATIC
    (7,  6),   # FLUX+PULSE→WAVE + STATIC→PULSE
    (8,  9),   # TRACE+BLOOM→SURGE + WAVE+DRIFT→FLUX+TRACE
    (9, 11),   # WAVE+DRIFT→FLUX+TRACE + PULSE→DRIFT+STATIC
    (10, 8),   # SURGE+STATIC→BLOOM + TRACE+BLOOM→SURGE
    (11, 10),  # PULSE→DRIFT+STATIC + SURGE+STATIC→BLOOM
])


def _is_forbidden_combination(a: Entity, b: Entity) -> bool:
    """True if (a, b) is a forbidden primitive combination."""
    if not a.is_primitive or not b.is_primitive:
        return False
    return (a.archetype_id, b.archetype_id) in HOLDOUT_COMBINATIONS


def sample_holdout_combination_example(
    rng: random.Random = random,
) -> Example:
    """
    Sample a valid example using a forbidden (a, b) archetype pairing.
    Used for combination generalization testing only — never called during training.
    """
    forbidden = list(HOLDOUT_COMBINATIONS)
    rng.shuffle(forbidden)
    for a_id, b_id in forbidden:
        from .entity import ARCHETYPE_BY_ID
        a = ARCHETYPE_BY_ID[a_id].to_entity()
        b = ARCHETYPE_BY_ID[b_id].to_entity()
        bindings = find_valid_bindings(a, b)
        if not bindings:
            continue
        port_a, port_b = rng.choice(bindings)
        compound = compose(a, b, port_a, port_b)
        a_tokens, a_map = serialize_entity(a)
        b_tokens, b_map = serialize_entity(b)
        rel_a = a_map[port_a]
        rel_b = b_map[port_b]
        a_start, a_end = 0, len(a_tokens)
        b_start = a_end + 3
        b_end = b_start + len(b_tokens)
        return Example(
            input_tokens=a_tokens + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_b)] + b_tokens,
            target_tokens=serialize_compound_output(compound),
            is_valid=True,
            entity_a=a,
            entity_b=b,
            compound=compound,
            a_token_span=(a_start, a_end),
            b_token_span=(b_start, b_end),
        )
    raise RuntimeError("No valid forbidden combination found")


def sample_holdout_example(
    rng: random.Random = random,
    max_depth: int = 1,
) -> Example:
    """
    Sample a valid example that includes at least one holdout archetype.
    Used for generalization testing only — never called during training.
    """
    for _ in range(100):
        # One entity must be a holdout primitive, the other can be anything
        a = rng.choice(HOLDOUT_ARCHETYPES).to_entity()
        b = sample_primitive(rng, ARCHETYPES)  # known archetype
        bindings = find_valid_bindings(a, b)
        if bindings:
            port_a, port_b = rng.choice(bindings)
            compound = compose(a, b, port_a, port_b)
            a_tokens, a_map = serialize_entity(a)
            b_tokens, b_map = serialize_entity(b)
            rel_a = a_map[port_a]
            rel_b = b_map[port_b]
            a_start, a_end = 0, len(a_tokens)
            b_start = a_end + 3
            b_end = b_start + len(b_tokens)
            return Example(
                input_tokens=a_tokens + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_b)] + b_tokens,
                target_tokens=serialize_compound_output(compound),
                is_valid=True,
                entity_a=a,
                entity_b=b,
                compound=compound,
                a_token_span=(a_start, a_end),
                b_token_span=(b_start, b_end),
            )
        # Also try b as holdout
        b = rng.choice(HOLDOUT_ARCHETYPES).to_entity()
        a = sample_primitive(rng, ARCHETYPES)
        bindings = find_valid_bindings(a, b)
        if bindings:
            port_a, port_b = rng.choice(bindings)
            compound = compose(a, b, port_a, port_b)
            a_tokens, a_map = serialize_entity(a)
            b_tokens, b_map = serialize_entity(b)
            rel_a = a_map[port_a]
            rel_b = b_map[port_b]
            a_start, a_end = 0, len(a_tokens)
            b_start = a_end + 3
            b_end = b_start + len(b_tokens)
            return Example(
                input_tokens=a_tokens + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_b)] + b_tokens,
                target_tokens=serialize_compound_output(compound),
                is_valid=True,
                entity_a=a,
                entity_b=b,
                compound=compound,
                a_token_span=(a_start, a_end),
                b_token_span=(b_start, b_end),
            )
    raise RuntimeError("Could not sample a holdout example after 100 attempts")


def _sample_valid(
    rng: random.Random,
    max_depth: int,
    archetype_pool=None,
    weights=None,
    cooccurrence_damp: float = 0.0,
) -> Example:
    for _ in range(50):
        a = sample_entity(rng, max_depth=max_depth, archetype_pool=archetype_pool, weights=weights)
        b = sample_entity(rng, max_depth=max_depth, archetype_pool=archetype_pool, weights=weights)
        if _is_forbidden_combination(a, b):
            continue
        if cooccurrence_damp > 0.0 and _flags_cooccur_across(a, b):
            if rng.random() < cooccurrence_damp:
                continue
        bindings = find_valid_bindings(a, b)
        if not bindings:
            continue
        port_a_idx, port_b_idx = rng.choice(bindings)
        compound = compose(a, b, port_a_idx, port_b_idx)

        a_tokens, a_map = serialize_entity(a)
        b_tokens, b_map = serialize_entity(b)
        rel_a = a_map[port_a_idx]
        rel_b = b_map[port_b_idx]

        # input: A | BIND rel_a rel_b | B
        a_start = 0
        a_end = len(a_tokens)
        b_start = a_end + 3  # BIND + two PORT_IDX tokens
        b_end = b_start + len(b_tokens)

        input_tokens = (
            a_tokens
            + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_b)]
            + b_tokens
        )

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
    raise RuntimeError("Could not sample a valid example after 50 attempts")


def _sample_invalid(rng: random.Random, max_depth: int, archetype_pool=None, weights=None) -> Example:
    for _ in range(50):
        a = sample_entity(rng, max_depth=max_depth, archetype_pool=archetype_pool, weights=weights)
        b = sample_entity(rng, max_depth=max_depth, archetype_pool=archetype_pool, weights=weights)
        if _is_forbidden_combination(a, b):
            continue

        valid_set = set(find_valid_bindings(a, b))
        all_pairs = [(ia, ib) for ia, _ in a.ports for ib, _ in b.ports]
        invalid_pairs = [p for p in all_pairs if p not in valid_set]
        if not invalid_pairs:
            continue

        port_a_idx, port_b_idx = rng.choice(invalid_pairs)
        a_tokens, a_map = serialize_entity(a)
        b_tokens, b_map = serialize_entity(b)

        # For invalid pairs, indices may not be in the maps if they reference
        # non-existent relative positions — clamp to 0 for the token, the
        # model should learn is_valid=False regardless of port token values.
        rel_a = a_map.get(port_a_idx, 0)
        rel_b = b_map.get(port_b_idx, 0)

        a_start, a_end = 0, len(a_tokens)
        b_start = a_end + 3
        b_end = b_start + len(b_tokens)

        input_tokens = (
            a_tokens
            + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_b)]
            + b_tokens
        )

        return Example(
            input_tokens=input_tokens,
            target_tokens=[Token.END],
            is_valid=False,
            entity_a=a,
            entity_b=b,
            compound=None,
            a_token_span=(a_start, a_end),
            b_token_span=(b_start, b_end),
        )
    raise RuntimeError("Could not sample an invalid example after 50 attempts")


# ---------------------------------------------------------------------------
# Attribution examples (Test 2)
# ---------------------------------------------------------------------------

@dataclass
class AttributionExample:
    """
    Input:  [A tokens] [SEP] [B tokens] [SEP] [compound tokens without END]
    Target: one attribution label per compound feature (port, op, flag), then END.

    Labels: ATTR_A, ATTR_B, ATTR_BOTH
      - Ports:  ATTR_A if port came from A, ATTR_B if from B
      - Op types: ATTR_A for A's op(s), ATTR_B for B's op(s)
      - Flags:  ATTR_A if only A has it, ATTR_B if only B has it, ATTR_BOTH if both
    """
    input_tokens: List[int]
    target_tokens: List[int]
    entity_a: Entity
    entity_b: Entity
    compound: Entity


def _attribution_labels(compound: "Entity") -> List[int]:
    """
    Compute attribution labels for each token in the compound output sequence
    (excluding END), in the same order as serialize_compound_output.
    """
    a = compound.component_a
    b = compound.component_b
    b_offset = max(idx for idx, _ in a.ports) + 1

    labels = []

    # Ports — sorted by compound port index
    for idx, port in sorted(compound.ports, key=lambda x: x[0]):
        labels.append(Token.ATTR_A if idx < b_offset else Token.ATTR_B)

    # Op types — A's ops first, then B's (same order as serialize_compound_output)
    for _ in a.op_types:
        labels.append(Token.ATTR_A)
    for _ in b.op_types:
        labels.append(Token.ATTR_B)

    # Flags — sorted by value, same order as serialize_compound_output
    for flag in sorted(compound.flags, key=lambda x: x.value):
        in_a = flag in a.flags
        in_b = flag in b.flags
        if in_a and in_b:
            labels.append(Token.ATTR_BOTH)
        elif in_a:
            labels.append(Token.ATTR_A)
        else:
            labels.append(Token.ATTR_B)

    labels.append(Token.END)
    return labels


def sample_attribution_example(
    rng: random.Random = random,
    max_depth: int = 2,
    archetype_pool=None,
    weights=None,
) -> AttributionExample:
    """Sample one attribution example from a valid compound."""
    for _ in range(50):
        a = sample_entity(rng, max_depth=max_depth, archetype_pool=archetype_pool, weights=weights)
        b = sample_entity(rng, max_depth=max_depth, archetype_pool=archetype_pool, weights=weights)
        bindings = find_valid_bindings(a, b)
        if not bindings:
            continue
        port_a_idx, port_b_idx = rng.choice(bindings)
        compound = compose(a, b, port_a_idx, port_b_idx)

        a_tokens, _ = serialize_entity(a)
        b_tokens, _ = serialize_entity(b)
        compound_tokens = serialize_compound_output(compound)[:-1]  # strip END

        input_tokens = a_tokens + [Token.SEP] + b_tokens + [Token.SEP] + compound_tokens
        target_tokens = _attribution_labels(compound)

        return AttributionExample(
            input_tokens=input_tokens,
            target_tokens=target_tokens,
            entity_a=a,
            entity_b=b,
            compound=compound,
        )
