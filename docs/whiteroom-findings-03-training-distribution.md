# Whiteroom — Training Distribution: Balancing and Co-occurrence Dampening

## Motivation

The original unbalanced training regime produced two dataset biases that
contributed to the 23.4% unsolvable failure rate in the property-append test:

1. **Non-uniform flag rates**: some flags appeared far more often than others
   in compound outputs, creating uneven priors.
2. **Cross-compound co-occurrence**: pairs where co-occurring flags (SCANS+ILLUMINATES,
   ATTRACTS+SPAWNS, SHOOTS+ILLUMINATES) split across A and B. The model learned
   to expect these flags together, so isolating one without the other suppressed
   its emission.

Two interventions were added: `--balance-archetypes` and `--cooccurrence-damp`.

---

## Archetype Structure

12 archetypes, 3 with zero flags (indices 0, 2, 7):

| Archetype | Flags |
|---|---|
| 0 | — |
| 1 | SHOOTS |
| 2 | — |
| 3 | ILLUMINATES, SCANS |
| 4 | BROADCASTS |
| 5 | ATTRACTS |
| 6 | SPAWNS |
| 7 | — |
| 8 | BROADCASTS |
| 9 | SCANS |
| 10 | ATTRACTS, SPAWNS |
| 11 | SHOOTS, ILLUMINATES |

Co-occurring pairs baked into archetypes: (ILLUMINATES, SCANS) in arch 3,
(ATTRACTS, SPAWNS) in arch 10, (SHOOTS, ILLUMINATES) in arch 11.

Each flag appears in exactly 2 archetypes out of the 9 flagged ones.

---

## balanced_archetype_weights

Assigns weight 0.25 to zero-flag archetypes (0, 2, 7) and 1.0 to all flagged
archetypes. This equalizes the **per-entity primitive** flag appearance rate
to ~20.5% for each flag (2 archetypes / total weight 9.75).

**Important limitation**: this equalization holds only at depth=1 (primitive
entities). At depth=2 (compound entities, which compose two primitives), the
compound output flag rate is the union of both entities' flags. The union
operation and nested compounding both amplify flag rates and re-introduce
non-uniformity. The balanced weights do not equalize compound-level flag rates.

---

## Measured Distribution (n=5000 compound examples each)

### Compound-level flag appearance rates

| Condition | SHOOTS | ILLUMINATES | SCANS | BROADCASTS | ATTRACTS | SPAWNS |
|---|---|---|---|---|---|---|
| depth=1, unbalanced | 0.524 | 0.300 | 0.392 | 0.369 | 0.393 | 0.451 |
| depth=1, balanced+damp | 0.634 | 0.347 | 0.395 | 0.536 | 0.516 | 0.671 |
| depth=2, unbalanced | 0.605 | 0.367 | 0.498 | 0.481 | 0.495 | 0.541 |
| depth=2, balanced+damp | 0.694 | 0.445 | 0.469 | 0.626 | 0.564 | 0.728 |

### Cross-compound co-occurrence rate

Fraction of valid compound pairs where A and B have flags from the same
co-occurring pair but split across entities (e.g. A has SCANS, B has ILLUMINATES).

| Condition | Cross-cooccur rate |
|---|---|
| depth=1, unbalanced | 0.182 |
| depth=1, balanced+damp | 0.110 |
| depth=2, unbalanced | 0.264 |
| depth=2, balanced+damp | 0.190 |

---

## Interpretation

**Co-occurrence dampening works as intended**: cross-compound co-occurrence
drops from 26.4% → 19.0% at training depth (depth=2). The damp factor of 0.7
rejects 70% of pairs where co-occurring flags split across A and B, reducing
the model's opportunity to learn that pattern.

**Balancing does not equalize compound-level flag rates**: all flag rates
increase under balanced+damp because flagged archetypes are upweighted, but
the rates remain non-uniform (range 0.445–0.728 at depth=2). ILLUMINATES
remains the rarest; SPAWNS and SHOOTS remain the most common.

**Why this matters for the failure analysis**: the co-occurrence dampening
addresses the primary cause of the 23.4% unsolvable cases (SCANS failing
without BROADCASTS, ILLUMINATES failing without SCANS). But the non-uniform
flag rates are a separate issue — the model still sees some flags far more
often than others, creating unequal priors.

**Note on earlier summary**: a prior analysis reported "all 6 flag rates equal
at 20.5% each" — this referred to the per-entity primitive rate (which is
equalized by the archetype weighting), not the compound output rate used
during training. The compound rates above are the operationally relevant ones.

---

---

## Footnote: Balanced vs Unbalanced — Eval Comparison and a Possible Hybrid

Five balanced seeds and five unbalanced seeds were both fully evaluated (n=300 per seed):

| Metric | Balanced mean ± std | Unbalanced mean ± std |
|--------|--------------------|-----------------------|
| freeze_a acc | 0.939 ± 0.049 | **0.977 ± 0.021** |
| freeze_b acc | 0.959 ± 0.019 | **0.970 ± 0.021** |
| pickup (hybrid) | 0.443 ± 0.185 | **0.550 ± 0.194** |
| attribution | 0.999 ± 0.002 | **1.000 ± 0.000** |

Unbalanced models score higher on every metric. Notably, the pickup variance
is essentially unchanged (0.185 vs 0.194) — balanced training did not stabilise
the seed-to-seed spread in hybrid pickup. Whatever balanced training buys, it
is not reducing that variance.

One possible explanation: the eval pairs are drawn from the same unbalanced
generator, so "better" unbalanced numbers may partly reflect distribution match
rather than strictly stronger representations. The balanced intervention constrains
the training signal, which may limit how strong representations become even as
it corrects co-occurrence artifacts.

**Possible hybrid approach**: train the base model unbalanced (stronger
representations, higher pickup ceiling), then apply a balanced fine-tuning pass
to correct co-occurrence artifacts before Stage 4 curriculum training. This
mirrors large-model practice of pretraining on a broad corpus then fine-tuning
on a curated one. However, given that balanced training did not reduce pickup
variance and Stage 4 is already a fine-tuning pass, the more efficient experiment
is to run Stage 4 on both balanced and unbalanced bases directly and compare —
that isolates the question of whether the base training regime affects curriculum
trainability without adding a separate Stage 2.5.

**Stage 4 may close the gap anyway.** If curriculum training successfully teaches
the decoder to handle frozen context explicitly, it replaces the incidental
variation in pickup (driven by base training luck) with a learned capability.
Both balanced and unbalanced bases would then converge on whatever ceiling the
curriculum can reach. If the gap closes post-Stage 4, the base regime doesn't
matter and the variation seen here is noise in an undertrained decoder. If
unbalanced bases still come out ahead, the base representations are load-bearing
and the hybrid approach is worth pursuing.

---

*Updated: 2026-03-31. Model: claude-sonnet-4-6.*
