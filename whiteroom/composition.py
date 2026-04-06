"""
Composition function and binding validator.

Given two entities A and B and a binding (port index on A, port index on B),
computes the compound entity or raises if the binding is invalid.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .entity import Entity, Port, Archetype, ARCHETYPE_BY_ID
from .vocab import PortType, OpType, Flag


# ---------------------------------------------------------------------------
# Binding validation
# ---------------------------------------------------------------------------

class InvalidBinding(Exception):
    pass


def find_valid_bindings(a: Entity, b: Entity) -> List[Tuple[int, int]]:
    """
    Return all (port_idx_on_A, port_idx_on_B) pairs that form a valid binding.
    A valid binding: an output port on A compatible with an input port on B,
    OR an output port on B compatible with an input port on A.
    """
    valid = []
    for ia, pa in a.ports:
        for ib, pb in b.ports:
            if pa.is_output and pb.is_input and pa.compatible_with(pb):
                valid.append((ia, ib))
            elif pa.is_input and pb.is_output and pb.compatible_with(pa):
                valid.append((ia, ib))  # stored as (idx_on_A, idx_on_B) regardless of direction
    return valid


def validate_binding(a: Entity, b: Entity, port_a_idx: int, port_b_idx: int) -> None:
    """
    Raise InvalidBinding if (port_a_idx, port_b_idx) is not a valid binding between A and B.
    Valid binding: one port is output, one is input, and they are type-compatible.
    """
    port_a = dict(a.ports).get(port_a_idx)
    port_b = dict(b.ports).get(port_b_idx)

    if port_a is None:
        raise InvalidBinding(f"Port index {port_a_idx} not found on entity A")
    if port_b is None:
        raise InvalidBinding(f"Port index {port_b_idx} not found on entity B")

    if port_a.is_output and port_b.is_input:
        if not port_a.compatible_with(port_b):
            raise InvalidBinding(
                f"Output {port_a} on A is not compatible with input {port_b} on B"
            )
    elif port_a.is_input and port_b.is_output:
        if not port_b.compatible_with(port_a):
            raise InvalidBinding(
                f"Output {port_b} on B is not compatible with input {port_a} on A"
            )
    else:
        raise InvalidBinding(
            f"Binding requires one input and one output port; "
            f"got {port_a.direction} on A and {port_b.direction} on B"
        )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose(a: Entity, b: Entity, port_a_idx: int, port_b_idx: int) -> Entity:
    """
    Compose A and B via the given binding.

    ports(compound) = (ports(A) ∪ ports(B)) - {bound_port_A, bound_port_B}
    flags(compound) = flags(A) ∪ flags(B)
    op_types: preserved per component — A's op_types + B's op_types

    Port indices are re-numbered in the compound to avoid collisions.
    The original indices are preserved in a stable offset scheme:
        compound port index = original index (A ports keep their index,
        B ports are offset by max(A port indices) + 1)
    """
    validate_binding(a, b, port_a_idx, port_b_idx)

    # Offset B's port indices to avoid collision with A's
    a_indices = {i for i, _ in a.ports}
    b_offset = max(a_indices) + 1 if a_indices else 0

    # Build surviving ports
    surviving = []
    for i, p in a.ports:
        if i != port_a_idx:
            surviving.append((i, p))
    for i, p in b.ports:
        if i != port_b_idx:
            surviving.append((i + b_offset, p))

    return Entity(
        ports=surviving,
        op_types=a.op_types + b.op_types,
        flags=a.flags | b.flags,
        archetype_id=None,
        component_a=a,
        component_b=b,
        binding=(port_a_idx, port_b_idx),
    )
