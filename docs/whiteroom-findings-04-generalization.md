# Whiteroom — Generalization Findings

Two holdout tests probing what the model actually learned: abstract compositional
rules, or token-specific mappings over memorized archetype pairings.

---

## Test 1: Token Holdout

### Setup

Retrained with 2 unseen port types (HOLDOUT1, HOLDOUT2) and 2 unseen flag types
(HOLDOUT1, HOLDOUT2) present in the vocabulary but excluded from all training
archetypes. Five test-only archetypes (IDs 12–16) used these holdout tokens.
Evaluated zero-shot on 500 holdout examples after 50K steps.

**Question**: does the model learn abstract structural rules ("output type = input
type", "flags propagate by union") or token-specific mappings?

### Results

| Condition | n | Seq exact | Port set | Flag exact |
|---|---|---|---|---|
| Standard (known tokens) | 794 | 82.1% | 87.3% | 99.9% |
| Holdout — all | 500 | 0.0% | 42.4% | 10.8% |
| Holdout — flag only (known ports) | 212 | 0.0% | **100%** | 0.0% |
| Holdout — port only (known flags) | 74 | 0.0% | 0.0% | 73.0% |
| Holdout — both | 214 | 0.0% | 0.0% | 0.0% |

### Interpretation

The model did not generalize to unseen token types. The breakdown is informative:

- **Flag-only holdout**: ports are predicted perfectly (known port tokens survive
  correctly), but unseen flag tokens are never emitted — the model cannot output
  a token ID it was never trained to produce.
- **Port-only holdout**: flags from known archetypes are mostly correct (73%),
  but unseen port types in the output are substituted with known port tokens.
- **Both holdout**: complete failure across all metrics.

The model learned **token-specific mappings** ("FLUX maps to FLUX") rather than
abstract structural rules ("output type = input type"). This is expected — the
training objective created no pressure to generalize to novel token IDs.
Composition rules were learned, but grounded in specific token identities rather
than abstract relations over token roles.

---

## Test 2: Combination Holdout

### Setup

Same vocabulary and architecture as Test 1 (vocab size 67). A set of 12 specific
archetype pairings was designated as forbidden and excluded from all training
examples — both valid and invalid compositions. The model saw every individual
archetype and every token type during training, but never the forbidden (A, B)
pairs. Evaluated zero-shot on 500 holdout examples drawn exclusively from
forbidden pairings, compared against 500 valid standard examples.

**Question**: did the model memorize specific (archetype_A, archetype_B) pairings,
or learn compositional rules that transfer to new combinations?

Forbidden combinations verified: 0/5000 training examples contained a holdout
pair (including at compound sub-entity depth).

### Results

| Condition | n | Seq exact | Port set | Flag exact |
|---|---|---|---|---|
| Standard (valid only) | 500 | 83.0% | 88.2% | 100.0% |
| Holdout combinations | 500 | 85.4% | 85.4% | 85.4% |

### Interpretation

The model generalizes strongly to unseen archetype pairings. Sequence exact match
is essentially unchanged (85.4% vs 83.0%). The only meaningful degradation is
flags: 85.4% vs 100.0% — the forbidden combinations produce flag unions the model
never encountered during training, and it handles ~85% of these correctly anyway.

The model learned **compositional rules that are archetype-pair agnostic**. It
knows "take the union of flags, drop bound ports, keep unbound ports from both
sides" as an abstract operation, and applies it correctly to novel (A, B) pairings.

**Note on seq/port/flag equality in holdout**: all three metrics coincide at 85.4%
for holdout, meaning any error in the sequence is simultaneously an error in both
port set and flag set — the model either gets the full compound right or fails
holistically, with no partial credit cases.

---

## Combined Interpretation

The two tests draw a clean line:

| What was withheld | Generalizes? | Why |
|---|---|---|
| Token types (unseen IDs) | **No** — complete failure | Model learned token-specific mappings; cannot emit IDs not seen in training |
| Archetype pairings (unseen combinations) | **Yes** — near-full accuracy | Model learned abstract compositional rules independent of which archetypes are being paired |

The model knows *how to compose* in an archetype-agnostic way, but that knowledge
is grounded in a fixed token vocabulary. Composition rules are abstract over
*pairings* but not over *token identities*.

**Implication for cache isolation (Stages 1–2)**: semantic independence and
attribution are robust behaviours that transfer across unseen archetype pairings
(combination holdout), but are grounded in specific token identities and would not
transfer to unseen token types (token holdout). The isolation is a property of
learned compositional structure, not of abstract symbol manipulation.

**Implication for generalization more broadly**: a different training objective
(e.g., masked token prediction forcing the model to predict token roles rather
than specific IDs) may be needed to induce true token-abstract generalization.
Combination generalization, by contrast, appears to emerge naturally from standard
composition training.

---

## Follow-up

See `whiteroom-findings-stage1.md` and `whiteroom-findings-stage2.md` for the
underlying composition and attribution results these generalization tests build on.

---

*Generated: 2026-03-30. Model: claude-sonnet-4-6. Implementation: PyTorch,
trained from scratch on synthetic data.*
