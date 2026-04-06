"""
Whiteroom entity model and archetype table.

Entities are either primitive (leaf) or compound (two components + one binding).
Compounds can themselves be components — composition trees are variable depth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple
from enum import IntEnum, auto

from .vocab import PortType, OpType, Flag, PORT_TYPES, FLAGS


# ---------------------------------------------------------------------------
# Port
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Port:
    """
    A single port on an entity.

    Output ports produce exactly one type.
    Input ports accept a set of types (size 1 = single-type, size >1 = category).
    """
    direction: str                      # 'in' or 'out'
    type_set: FrozenSet[PortType]       # for 'out', always a singleton

    @staticmethod
    def input(types: FrozenSet[PortType]) -> Port:
        assert len(types) >= 1
        return Port(direction="in", type_set=types)

    @staticmethod
    def output(pt: PortType) -> Port:
        return Port(direction="out", type_set=frozenset({pt}))

    @property
    def is_input(self) -> bool:
        return self.direction == "in"

    @property
    def is_output(self) -> bool:
        return self.direction == "out"

    def compatible_with(self, other: Port) -> bool:
        """
        Returns True if self (output) can bind to other (input).
        Binding rules:
          - self must be output, other must be input
          - Single-type input: type(self) ∈ type_set(other)
          - Category input (|type_set| > 1): type(self) ∈ type_set(other)
          - Category-to-category (both |type_set| > 1): type_set(self) == type_set(other)
            [self is always singleton output so this case is excluded by construction]
        """
        if self.direction != "out" or other.direction != "in":
            return False
        # self.type_set is always a singleton (output port)
        return self.type_set.issubset(other.type_set)

    def __repr__(self) -> str:
        types = ",".join(pt.name for pt in sorted(self.type_set, key=lambda x: x.value))
        return f"Port({self.direction},{types})"


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """
    An entity — either primitive or compound.

    For primitives:
        archetype_id is set, components/binding are None.
    For compounds:
        archetype_id is None, components and binding are set.
        ports and flags are derived from components via composition rules.
    """
    # Port signature: list of (port_index, Port) — index is stable identity
    ports: List[Tuple[int, Port]]

    # Operation type (compound: list, one per surviving external output port cluster)
    # Stored per-component for compounds: op_types[0] = A's op, op_types[1] = B's op
    op_types: List[OpType]

    # Side-behavior flags (union for compounds)
    flags: FrozenSet[Flag]

    # Primitive-only
    archetype_id: Optional[int] = None

    # Compound-only
    component_a: Optional[Entity] = None
    component_b: Optional[Entity] = None
    binding: Optional[Tuple[int, int]] = None  # (port_index_on_A, port_index_on_B)

    @property
    def is_primitive(self) -> bool:
        return self.archetype_id is not None

    @property
    def is_compound(self) -> bool:
        return self.component_a is not None

    def input_ports(self) -> List[Tuple[int, Port]]:
        return [(i, p) for i, p in self.ports if p.is_input]

    def output_ports(self) -> List[Tuple[int, Port]]:
        return [(i, p) for i, p in self.ports if p.is_output]

    def __repr__(self) -> str:
        if self.is_primitive:
            return (f"Primitive(arch={self.archetype_id}, "
                    f"ports={self.ports}, op={self.op_types}, flags={self.flags})")
        return (f"Compound(binding={self.binding}, "
                f"ports={self.ports}, flags={self.flags})")


# ---------------------------------------------------------------------------
# Archetype table
#
# A fixed set of primitive entity templates. Each archetype defines:
#   - port signature (list of Ports)
#   - operation type
#   - flag set
#
# Designed for coverage:
#   - All 8 port types appear as both inputs and outputs across the table
#   - All 3 op types represented
#   - All 6 flags represented
#   - Mix of single-type and category input ports
#   - Enough variety for non-trivial composition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Archetype:
    archetype_id: int
    ports: Tuple[Port, ...]
    op_type: OpType
    flags: FrozenSet[Flag]

    def to_entity(self) -> Entity:
        """Instantiate as an Entity with stable port indices."""
        indexed_ports = [(i, p) for i, p in enumerate(self.ports)]
        return Entity(
            ports=indexed_ports,
            op_types=[self.op_type],
            flags=self.flags,
            archetype_id=self.archetype_id,
        )


T = PortType
O = OpType
F = Flag

ARCHETYPES: List[Archetype] = [
    # 0 — simple converter: FLUX in → PULSE out, void
    Archetype(0,
        ports=(Port.input(frozenset({T.FLUX})), Port.output(T.PULSE)),
        op_type=O.VOID, flags=frozenset()),

    # 1 — amplifier: PULSE in → WAVE out + SURGE out, throttle, shoots
    Archetype(1,
        ports=(Port.input(frozenset({T.PULSE})), Port.output(T.WAVE), Port.output(T.SURGE)),
        op_type=O.THROTTLE, flags=frozenset({F.SHOOTS})),

    # 2 — splitter: WAVE in → DRIFT out + BLOOM out, toggle
    Archetype(2,
        ports=(Port.input(frozenset({T.WAVE})), Port.output(T.DRIFT), Port.output(T.BLOOM)),
        op_type=O.TOGGLE, flags=frozenset()),

    # 3 — scanner: SURGE in → TRACE out, void, scans + illuminates
    Archetype(3,
        ports=(Port.input(frozenset({T.SURGE})), Port.output(T.TRACE)),
        op_type=O.VOID, flags=frozenset({F.SCANS, F.ILLUMINATES})),

    # 4 — broadcaster: DRIFT in → STATIC out, throttle, broadcasts
    Archetype(4,
        ports=(Port.input(frozenset({T.DRIFT})), Port.output(T.STATIC)),
        op_type=O.THROTTLE, flags=frozenset({F.BROADCASTS})),

    # 5 — attractor: BLOOM in → FLUX out, toggle, attracts
    Archetype(5,
        ports=(Port.input(frozenset({T.BLOOM})), Port.output(T.FLUX)),
        op_type=O.TOGGLE, flags=frozenset({F.ATTRACTS})),

    # 6 — spawner: STATIC in → PULSE out, void, spawns
    Archetype(6,
        ports=(Port.input(frozenset({T.STATIC})), Port.output(T.PULSE)),
        op_type=O.VOID, flags=frozenset({F.SPAWNS})),

    # 7 — multi-input relay: {FLUX,PULSE} in → WAVE out, throttle
    Archetype(7,
        ports=(Port.input(frozenset({T.FLUX, T.PULSE})), Port.output(T.WAVE)),
        op_type=O.THROTTLE, flags=frozenset()),

    # 8 — dual-input mixer: TRACE in + BLOOM in → SURGE out, toggle, broadcasts
    Archetype(8,
        ports=(Port.input(frozenset({T.TRACE})), Port.input(frozenset({T.BLOOM})),
               Port.output(T.SURGE)),
        op_type=O.TOGGLE, flags=frozenset({F.BROADCASTS})),

    # 9 — echo: {WAVE,DRIFT} in → FLUX out + TRACE out, void, scans
    Archetype(9,
        ports=(Port.input(frozenset({T.WAVE, T.DRIFT})), Port.output(T.FLUX),
               Port.output(T.TRACE)),
        op_type=O.VOID, flags=frozenset({F.SCANS})),

    # 10 — relay: SURGE in + STATIC in → BLOOM out, throttle, attracts + spawns
    Archetype(10,
        ports=(Port.input(frozenset({T.SURGE})), Port.input(frozenset({T.STATIC})),
               Port.output(T.BLOOM)),
        op_type=O.THROTTLE, flags=frozenset({F.ATTRACTS, F.SPAWNS})),

    # 11 — sink/source: PULSE in → DRIFT out + STATIC out, toggle, shoots + illuminates
    Archetype(11,
        ports=(Port.input(frozenset({T.PULSE})), Port.output(T.DRIFT),
               Port.output(T.STATIC)),
        op_type=O.TOGGLE, flags=frozenset({F.SHOOTS, F.ILLUMINATES})),
]

N_ARCHETYPES = len(ARCHETYPES)
ARCHETYPE_BY_ID = {a.archetype_id: a for a in ARCHETYPES}

# ---------------------------------------------------------------------------
# Holdout archetypes — test-only, never used during training.
# Use HOLDOUT1/HOLDOUT2 port types and flags to test generalization.
# ---------------------------------------------------------------------------

HOLDOUT_ARCHETYPES: List[Archetype] = [
    # 12 — holdout port types only: HOLDOUT1 in → HOLDOUT2 out, void
    Archetype(12,
        ports=(Port.input(frozenset({T.HOLDOUT1})), Port.output(T.HOLDOUT2)),
        op_type=O.VOID, flags=frozenset()),

    # 13 — holdout flag only: FLUX in → PULSE out, throttle, HOLDOUT_FLAG_1
    Archetype(13,
        ports=(Port.input(frozenset({T.FLUX})), Port.output(T.PULSE)),
        op_type=O.THROTTLE, flags=frozenset({F.HOLDOUT1})),

    # 14 — both holdout port + flag: HOLDOUT2 in → SURGE out, toggle, HOLDOUT_FLAG_2
    Archetype(14,
        ports=(Port.input(frozenset({T.HOLDOUT2})), Port.output(T.SURGE)),
        op_type=O.TOGGLE, flags=frozenset({F.HOLDOUT2})),

    # 15 — holdout port as output + known flag: WAVE in → HOLDOUT1 out, void, scans
    Archetype(15,
        ports=(Port.input(frozenset({T.WAVE})), Port.output(T.HOLDOUT1)),
        op_type=O.VOID, flags=frozenset({F.SCANS})),

    # 16 — mix: HOLDOUT1 in → PULSE out, throttle, HOLDOUT_FLAG_1 + broadcasts
    Archetype(16,
        ports=(Port.input(frozenset({T.HOLDOUT1})), Port.output(T.PULSE)),
        op_type=O.THROTTLE, flags=frozenset({F.HOLDOUT1, F.BROADCASTS})),
]

ALL_ARCHETYPES = ARCHETYPES + HOLDOUT_ARCHETYPES
HOLDOUT_ARCHETYPE_IDS = {a.archetype_id for a in HOLDOUT_ARCHETYPES}
