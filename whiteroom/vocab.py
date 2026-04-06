"""
Whiteroom token vocabulary.

Total vocab size is kept small (~30-50 tokens) since this is a structural
reasoning task over short sequences.
"""

from enum import IntEnum, auto


# ---------------------------------------------------------------------------
# Port types — abstract transfer channel labels, no real-world semantics
# ---------------------------------------------------------------------------

class PortType(IntEnum):
    FLUX      = auto()   # T1
    PULSE     = auto()   # T2
    WAVE      = auto()   # T3
    DRIFT     = auto()   # T4
    SURGE     = auto()   # T5
    TRACE     = auto()   # T6
    BLOOM     = auto()   # T7
    STATIC    = auto()   # T8
    HOLDOUT1  = auto()   # T9  — reserved for generalization testing, never used in training
    HOLDOUT2  = auto()   # T10 — reserved for generalization testing, never used in training


PORT_TYPES = list(PortType)
N_PORT_TYPES = len(PORT_TYPES)
TRAINING_PORT_TYPES = [pt for pt in PortType if pt not in (PortType.HOLDOUT1, PortType.HOLDOUT2)]


# ---------------------------------------------------------------------------
# Operation types
# ---------------------------------------------------------------------------

class OpType(IntEnum):
    VOID     = auto()
    THROTTLE = auto()
    TOGGLE   = auto()


OP_TYPES = list(OpType)


# ---------------------------------------------------------------------------
# Side-behavior flags
# ---------------------------------------------------------------------------

class Flag(IntEnum):
    SHOOTS     = auto()
    ILLUMINATES = auto()
    SCANS      = auto()
    BROADCASTS = auto()
    ATTRACTS   = auto()
    SPAWNS     = auto()
    HOLDOUT1   = auto()   # reserved for generalization testing, never used in training
    HOLDOUT2   = auto()   # reserved for generalization testing, never used in training


FLAGS = list(Flag)
TRAINING_FLAGS = [f for f in Flag if f not in (Flag.HOLDOUT1, Flag.HOLDOUT2)]


# ---------------------------------------------------------------------------
# Token vocabulary
#
# Layout:
#   [SPECIAL tokens] [port-in tokens] [port-out tokens] [op tokens] [flag tokens]
#
# Port tokens encode (type, direction) — one token per (type, direction) pair.
# ---------------------------------------------------------------------------

class Token(IntEnum):
    # Special
    PAD      = 0
    COMPOUND = auto()
    BIND     = auto()
    END      = auto()
    SEP      = auto()   # separates A and B sequences in attribution task
    MASK     = auto()   # for masked prediction if needed later

    # Port-in tokens: one per PortType
    PORT_IN_FLUX     = auto()
    PORT_IN_PULSE    = auto()
    PORT_IN_WAVE     = auto()
    PORT_IN_DRIFT    = auto()
    PORT_IN_SURGE    = auto()
    PORT_IN_TRACE    = auto()
    PORT_IN_BLOOM    = auto()
    PORT_IN_STATIC   = auto()
    PORT_IN_HOLDOUT1 = auto()
    PORT_IN_HOLDOUT2 = auto()

    # Port-out tokens: one per PortType
    PORT_OUT_FLUX     = auto()
    PORT_OUT_PULSE    = auto()
    PORT_OUT_WAVE     = auto()
    PORT_OUT_DRIFT    = auto()
    PORT_OUT_SURGE    = auto()
    PORT_OUT_TRACE    = auto()
    PORT_OUT_BLOOM    = auto()
    PORT_OUT_STATIC   = auto()
    PORT_OUT_HOLDOUT1 = auto()
    PORT_OUT_HOLDOUT2 = auto()

    # Op type tokens
    OP_VOID     = auto()
    OP_THROTTLE = auto()
    OP_TOGGLE   = auto()

    # Flag tokens
    FLAG_SHOOTS      = auto()
    FLAG_ILLUMINATES = auto()
    FLAG_SCANS       = auto()
    FLAG_BROADCASTS  = auto()
    FLAG_ATTRACTS    = auto()
    FLAG_SPAWNS      = auto()
    FLAG_HOLDOUT1    = auto()
    FLAG_HOLDOUT2    = auto()

    # Port index tokens — used after BIND to reference ports by relative position
    # within A's or B's serialized port list (0-indexed). 10 positions is sufficient
    # for any entity at the supported composition depth.
    PORT_IDX_0 = auto()
    PORT_IDX_1 = auto()
    PORT_IDX_2 = auto()
    PORT_IDX_3 = auto()
    PORT_IDX_4 = auto()
    PORT_IDX_5 = auto()
    PORT_IDX_6 = auto()
    PORT_IDX_7 = auto()
    PORT_IDX_8 = auto()
    PORT_IDX_9 = auto()

    # Attribution labels — used as targets for the attribution task (Test 2).
    # One label per feature in the compound output, indicating which component
    # it is attributable to.
    ATTR_A    = auto()   # feature comes from component A exclusively
    ATTR_B    = auto()   # feature comes from component B exclusively
    ATTR_BOTH = auto()   # feature present in both (flags only)


VOCAB_SIZE_BASE = len(Token)  # before archetype tokens

MAX_PORT_IDX = 9   # max supported relative port position
MAX_EDIT_POS = 31  # max edit position for compact sibling (appended after VOCAB_SIZE)

# Edit position token IDs for the compact sibling — appended AFTER all archetype
# tokens so they don't shift any existing token IDs.
# These are plain integers, not Token enum members, since they only exist in the
# sibling's extended vocab and must not affect the primary model's vocab layout.
# EDIT_POS_OFFSET is set after VOCAB_SIZE is computed in generator.py.
# Use edit_pos_id(pos) to get the token ID.


def port_idx_token(pos: int) -> Token:
    if pos > MAX_PORT_IDX:
        raise ValueError(f"Port position {pos} exceeds MAX_PORT_IDX={MAX_PORT_IDX}")
    return Token(Token.PORT_IDX_0 + pos)


def edit_pos_id(pos: int, vocab_size: int) -> int:
    """
    Return the token ID for edit position `pos` in the compact sibling vocab.
    Edit position tokens are appended after vocab_size (the primary vocab size
    including archetype tokens), so they don't shift any existing IDs.

    vocab_size: the primary VOCAB_SIZE (from generator.VOCAB_SIZE)
    """
    if pos > MAX_EDIT_POS:
        raise ValueError(f"Edit position {pos} exceeds MAX_EDIT_POS={MAX_EDIT_POS}")
    return vocab_size + pos


def sibling_vocab_size(primary_vocab_size: int) -> int:
    """Total vocab size for the compact sibling: primary + 32 edit position tokens."""
    return primary_vocab_size + MAX_EDIT_POS + 1


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def port_in_token(pt: PortType) -> Token:
    return Token["PORT_IN_" + pt.name]


def port_out_token(pt: PortType) -> Token:
    return Token["PORT_OUT_" + pt.name]


def op_token(op: OpType) -> Token:
    return Token["OP_" + op.name]


def flag_token(flag: Flag) -> Token:
    return Token["FLAG_" + flag.name]


def port_token(pt: PortType, direction: str) -> Token:
    """direction: 'in' or 'out'"""
    if direction == "in":
        return port_in_token(pt)
    elif direction == "out":
        return port_out_token(pt)
    raise ValueError(f"Unknown direction: {direction!r}")
