# Whiteroom — Dataset Spec

## Overview

A synthetic dataset for training a small transformer on compositional reasoning.
Entities are composed from components via typed port bindings. The research goal is to test
whether a model trained on this dataset learns semantic isolation — specifically, whether freezing
one component's cached representations during inference remains valid when the other component
is swapped for a compliant replacement.

---

## Entity Model

Entities are either **primitive** (leaf) or **compound** (two components + one binding, recursive).
A compound can itself be a component of another compound — composition trees are variable depth.

Each entity has:
- A **port signature**: set of typed ports
- An **operation type**: void / throttle / toggle
- A **side-behavior flag set**: 0–2 boolean flags from a shared vocabulary

---

## Port Model

### Port Types

A flat vocabulary of abstract transfer types — exact count TBD during archetype design, target ~8–12.
Types are labels only; no real-world semantics. Called T1, T2, ... Tn (or named during implementation).

Each port is a tuple: `(type_or_type_set, direction)`

- **direction**: `in` or `out`
- **type**: a single type from the vocabulary (output ports always produce one type)
- **type set**: a set of types from the vocabulary (input ports may accept multiple types)

### Binding Compatibility

A binding between output port A and input port B is valid iff:
- `type(A) ∈ type_set(B)` — the output's type is accepted by the input
- For category inputs (set size > 1): output type must be fully contained — `{type(A)} ⊆ type_set(B)`

Category-to-category bindings (both sides have set size > 1) require the sets to be equal:
`type_set(A_out) = type_set(B_in)` — subset relationships introduce dynamic switching behavior
that is out of scope for this model.

### Port Pairing Rules

- Each port is either **paired** (bound, internal, cancelled) or **external** (exposed on the compound)
- Ports pair 1:1 — a port cannot participate in more than one binding
- Fan-in and single-port fan-out are excluded by construction
- One binding pair per compound formation step

### Composition

```
ports(compound) = (ports(A) ∪ ports(B)) - {bound_port_A, bound_port_B}
```

Bound ports cancel and do not appear on the compound's external port signature.

---

## Operation Type

Entity-level property — governs behavior when any output is backlogged:

| Value | Behavior | Signal analogy |
|---|---|---|
| **Void** | Continues consuming input; discards/voids output if backlogged | DC constant |
| **Throttle** | Reduces input consumption proportionally to output clearance | Analog, saturating |
| **Toggle** | Stops consuming when output blocked; resumes when cleared | Digital square wave |

### Backpressure Propagation

When downstream output clogs, backpressure travels upstream through the binding chain:

- **Void**: absorbs backpressure entirely. Nothing propagates upstream.
- **Throttle**: transmits a modulated/eased version of the signal. Averages toward a constant mean given sufficient buffer. Analog — linear at low load, attenuates toward zero at saturation.
- **Toggle**: converts any incoming backpressure to binary toggle upstream (unless already absorbed by void).

### Compound Operation Type

No single reduced value for compounds. Each component's operation type applies independently
to its own surviving external output ports. Backpressure propagates through the binding chain
per the rules above — void absorbs, throttle eases, toggle quantizes.

---

## Side-Behavior Flags

Boolean flags representing behaviors orthogonal to the port graph. Active whenever the entity
is operational. Propagate through compounding by **union** — if either component has a flag,
the compound has it.

- Vocabulary size: ~6–8 flags (TBD)
- Per archetype: 0–2 flags
- Flags are the primary signal for the KV cache locality test (see Research Design)

Examples (to be finalized): `shoots`, `illuminates`, `scans`, `broadcasts`, `attracts`, `spawns`

---

## Token Sequence Representation

Each entity is represented as a token sequence:

```
[archetype_id] [port_token ...] [op_type_token] [flag_token ...]
```

Where:
- `archetype_id`: identifies the primitive type (or COMPOUND for compound entities)
- `port_token`: encodes `(type, direction)` — one token per port
- `op_type_token`: one of {VOID, THROTTLE, TOGGLE}
- `flag_token`: one token per active flag (0–2 tokens)

Compound entities are represented by concatenating component sequences with binding and compound markers:

```
[COMPOUND] [A tokens...] [BIND port_A port_B] [B tokens...] [END]
```

Vocabulary is small — port types × directions + op types + flags + special tokens ≈ 30–50 tokens total.

---

## Generator / Verifier

Both are pure Python programs, no ML required.

**Generator**:
1. Samples two compatible entities (primitive or compound)
2. Finds a valid binding pair (compatible port types)
3. Computes the compound entity by applying composition rules
4. Serializes as token sequences
5. Labels: `is_valid = true`, bound port pair, compound token sequence

For invalid examples: sample incompatible port pairs, label `is_valid = false`, C output masked.

**Verifier**:
Given (A, B, proposed_binding, C, is_valid_prediction):
- Check binding validity via spec rules
- If valid: compute expected C via composition function, compare to predicted C
- If invalid: check is_valid = false, ignore C

Ground truth is entirely programmatic — no labels needed beyond what the spec computes.
Data generation is effectively infinite and free.

---

## Research Design

### Test 1 — Cache Freezing (Primary)

**Setup**: train a model on compound(A, B) → C. Then at inference:
1. Run compound(A, B), cache A's KV tensors
2. Swap B for compliant C (same binding port type, different flags/operation type)
3. Run compound(A, C) with A's KV tensors frozen
4. Measure: does the output still correctly reflect compound(A, C)?

**Perfect result**: output is always correct for any compliant swap. The model never needs to
recompute A when B changes, as long as the binding contract is honored.

**Practical result**: measure how much freezing degrades output quality across many swaps.
Quantify residual coupling — how much does A's representation change when B changes?

**Ground truth**: side-behavior flags. A's flags are determined entirely by A and cannot be
affected by swapping B. Any degradation in flag prediction for A's flags after freezing is
a direct measure of spurious coupling.

### Test 2 — Component Attribution (Secondary)

**Input**: A tokens, B tokens, compound(A, B) tokens
**Output**: for each feature in compound(A, B) — each surviving port, each flag — which component
is it attributable to? {A, B, BINDING}

**Ground truth**: fully deterministic. Surviving ports are traceable to their source component.
Flags are traceable to whichever component(s) carry them (both if shared, one if exclusive).
Bound ports are labelled BINDING (cancelled, internal).

**Relationship to Test 1**: if the model correctly attributes features AND freezing works in
Test 1, the two results reinforce each other — correct attribution is evidence the model has
learned the semantic boundary explicitly, not just implicitly through freezing.

**Natural language connection**: attribution in the whiteroom is the structured analog of
"which parts of this context came from File A vs File B." A model that learns attribution
here has a capability directly transferable to the follow-up research.

### Test 3 — Binding Verification (Tertiary, nearly free)

**Input**: A tokens, B tokens, proposed port_A, proposed port_B
**Output**: `is_valid` (bool) + compound C if valid

Same training data as Test 1, different output head. Generalization test: hold out archetype
combinations during training, test on unseen combinations. Does the model learn the binding
*rules* or memorize specific combinations?

**Extended generalization**: after training, introduce new flag tokens not seen during training.
Binding validity should be unaffected (flags don't affect binding rules). Compound flag
inheritance should work correctly for new flags. Tests whether the model learned structural
independence of flags from port mechanics.

### Research Arc

1. **Test 1**: prove freezing works when binding contracts are explicit → publishable
2. **Test 2**: prove a model can learn to attribute compound features to source components → reinforces Test 1
3. **Test 3**: prove a model can learn to identify valid contracts from examples → nearly free
4. **Future (out of scope)**: transfer attribution and contract-finding to natural language → separate research project

---

## Implementation Notes

- Framework: PyTorch (small transformer, trained from scratch)
- Model size: small — this is a structural reasoning task, not a language task
- Data: generated on the fly during training (infinite, no static dataset needed)
- No pretrained weights, no fine-tuning — train from scratch on synthetic data
- KV cache probing: extract K and V tensors per layer per token position, compare across swap pairs
