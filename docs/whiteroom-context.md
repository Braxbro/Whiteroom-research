# Whiteroom — Research Context

## The Problem

Transformer language models use a **KV cache** during inference. When processing a sequence of
tokens, each layer computes Key (K) and Value (V) matrices for each token. These are cached so
that when generating the next token, the model doesn't recompute them for all previous tokens.

The problem: KV cache invalidation is **purely positional**. If any token at position N changes
or is inserted, everything from position N onward must be recomputed. This is correct for
causally dependent content — but much real-world content isn't causally dependent in the
sequential sense.

**Example**: a context contains two file references, File A and File B, followed by instructions
that use both. If File A is updated with a new rule inserted in the middle, the naive cache
invalidates everything after the insertion point — including all of File B's contribution,
which hasn't changed and has no semantic relationship to the new rule in File A.

**Example**: a calendar context has free time from 2–6PM and a meeting at 3–4PM. The meeting
moves to 4–5PM — still within the free block, no constraint violated. Positional cache
invalidation recomputes everything after the meeting's original position. But the free block's
representation hasn't changed. Nothing about the rest of the day has changed. The invalidation
is spurious.

---

## The Insight

Semantic dependency is not the same as sequential dependency. Two segments of a context can
coexist without interacting — they occupy "free space" relative to each other. Changing one
should not require recomputing the other.

**Hypothesis**: if a model can learn to represent semantic independence — to understand where
the "void" is between segments — then its KV cache representations for independent segments
should remain valid when one segment changes, even if positions shift.

**Corollary**: if you can freeze one component's KV cache and swap the other for a compliant
replacement, and the model's output is still correct, then the model has learned that the
frozen component's representation is sufficient — it didn't need to re-attend to the swapped
component to understand the compound result.

---

## Why Factorio Compound Entities

The researcher is familiar with Factorio's **compound entity** pattern — a modding technique
where two game entities are bound together via a typed connection (fluid pipe, heat pipe, item
transfer, etc.) to produce a single logical entity with combined capabilities.

This maps naturally to the research problem:
- Two components (A and B) with explicit typed interfaces
- A defined binding between them
- The compound's capabilities are the union of A's and B's, minus the bound interfaces
- A's non-binding capabilities are semantically independent of B's non-binding capabilities

The insight: if you swap B for a compliant replacement C (same binding interface, different
other properties), A's contribution to the compound is unchanged. This is exactly the
"change one segment, freeze the other" test for KV cache locality.

---

## Why Not Real Factorio Entities

Real Factorio entities are messy:
- API properties are Lua surface area, not semantic capabilities
- Component field names are overloaded (energy_source means different things on different entities)
- The entity set is too small for a rich training corpus

Instead: a **whiteroom** — a clean synthetic domain inspired by the Factorio compound entity
concept, with all the noise removed and complexity expanded to generate rich training data.

---

## The Whiteroom

A synthetic domain where:
- Entities have **typed ports** (abstract transfer channels, not Factorio-specific)
- Entities have an **operation type** (how they behave when output is backlogged)
- Entities have **side-behavior flags** (capabilities orthogonal to the port graph — "shoots things", "illuminates area", etc.)
- Two entities **compound** by binding one output port to one input port of matching type
- The compound inherits both components' port signatures (minus the bound pair) and all their flags

Everything is deterministic and programmatically verifiable. The "correct" output for any
compound formation is computable from the spec rules — no labeling required.

---

## What We're Testing

**Primary test**: train a model on compound(A, B) → C. At inference, freeze A's KV tensors,
swap B for compliant C', run inference. Does the model still correctly predict compound(A, C')?

If yes: the model learned that A's representation is independent of which compliant B is present.
The binding contract — the port type — is the sufficient abstraction boundary.

If no (partially): measure how much degradation occurs. The residual coupling tells you how far
the model is from perfect semantic isolation.

**Key signal**: side-behavior flags. A's flags are determined entirely by A. If you freeze A
and the model still correctly predicts A's flags in the compound output, the frozen representation
was sufficient. Any error is direct evidence of spurious coupling.

**Perfect result**: freezing A is always valid for any compliant B swap. The model never needs
to recompute A as long as the binding contract is honored.

---

## Secondary Test — Component Attribution

**Input**: A tokens, B tokens, compound(A, B) tokens
**Output**: for each feature in the compound — each port, each flag — which component is it
attributable to? A, B, or the binding (cancelled)?

Ground truth is fully deterministic and free. This is the inverse of the composition task.

**Why it matters**: if the model correctly attributes features AND cache freezing works (Test 1),
the results reinforce each other. Correct attribution means the model has learned the semantic
boundary explicitly — it *knows* which parts of the compound came from which source.

This is also the structured analog of "which parts of this context came from File A vs File B"
in natural language — directly relevant to the follow-up research on unstructured contexts.

## Tertiary Test — Binding Verification (Nearly Free)

**Input**: A, B, proposed binding (port on A + port on B)
**Output**: is this binding valid? If yes, what is compound(A, B)?

Same training data, different output head. Also tests generalization:
- Hold out archetype combinations during training — does the model learn rules or memorize pairs?
- After training, add new flag types — does binding validity remain unaffected? (It should — flags
  are orthogonal to binding mechanics by construction.)

---

## Research Arc

1. **This project**: prove the principle on toy data. Two publishable results for the cost of one implementation.
2. **Follow-up (not this project)**: train a model to identify semantic binding contracts in natural language — where the "free space" is in unstructured text.
3. **Further follow-up**: architectural intervention — if models can learn semantic independence, can you design attention mechanisms that enforce it? (Separator tokens, attention gating, etc.)

Step 1 is the prerequisite discriminator: if the model can't learn semantic independence from
structured toy data, steps 2 and 3 are probably futile. If it can, you have empirical basis
for the harder problems.

---

## What Needs to Be Built

### 1. Generator / Verifier (pure Python)

- Define vocabulary: port types (~8–12), operation types (3), flag vocabulary (~6–8)
- Define archetype sampling: random port signatures + operation type + 0–2 flags
- Implement composition function: given A, B, binding → compound C
- Implement verifier: given predicted output, check against composition function
- Serialize entities as token sequences for model input

### 2. Small Transformer (PyTorch)

- Train from scratch on generated data — no pretrained weights
- Small model: this is a structural reasoning task over short token sequences (~30–50 token vocabulary)
- Two output heads: (a) compound token sequence, (b) is_valid flag
- Data generated on the fly during training — infinite, no static dataset

### 3. KV Cache Probing

- After training, run compound(A, B) and extract K/V tensors per layer per token position
- Run compound(A, C) (compliant swap) with A's tensors frozen
- Measure output quality degradation
- Compare delta in A's tensors vs delta in B's tensors across swap pairs

---

## Technical Notes

- **Framework**: PyTorch. No special libraries needed beyond standard ML stack.
- **Model size**: small — a few layers, modest hidden dimension. The task is simple enough that
  a large model would just overfit and obscure the structural result.
- **Data**: the generator IS the dataset. Generate examples on the fly during training.
  No need for a static train/test split file — just hold out specific archetype combinations.
- **KV cache**: not a file. It's the K and V activation tensors computed during a forward pass.
  To probe them, run a forward pass and extract the tensors from each attention layer.
  PyTorch hooks make this straightforward.

---

## Full Spec

See `whiteroom-spec.md` for the complete formal specification of the entity model, port algebra,
operation type semantics, binding rules, and token representation.
