# Whiteroom — Sibling Cross-Checkpoint Generalization

## Setup

The sibling trained in Stage 3c was evaluated not just on its training
checkpoint (stage2-attribution-50k, seed 42) but across all available
checkpoints at different training stages. The question: does the sibling's
freeze policy generalize to checkpoints with different internal representations?

Checkpoints evaluated:
- stage1-10k, stage1-30k, stage1-50k  (no attribution, standard training)
- stage2-50k  (with attribution — the sibling's training regime)
- stage3-50k  (combination holdout retrain)

For each checkpoint: freeze_all baseline, full_fresh upper bound, oracle
(brute-force optimal span mask), and sibling-predicted mask. n=300 pairs each.

---

## Results

| Checkpoint | freeze_all | full_fresh | oracle | sibling | sibling vs oracle |
|---|---|---|---|---|---|
| stage1-10k | 0.587 | 0.673 | 0.693 | 0.600 | −0.093 |
| stage1-30k | 0.620 | 0.710 | 0.720 | 0.637 | −0.083 |
| stage1-50k | 0.640 | 0.720 | 0.733 | 0.650 | −0.083 |
| stage2-50k | 0.640 | 0.747 | 0.747 | **0.747** | **0.000** |
| stage3-50k | 0.657 | 0.795 | 0.795 | 0.695 | −0.100 |

---

## Interpretation

### Within-regime: perfect

On stage2-50k — the checkpoint the sibling was trained on — the sibling
matches oracle exactly (74.7%). This reproduces the Stage 3c result. The
sibling learned a perfect freeze policy for this specific checkpoint's
representations.

### Cross-regime: degrades to near-freeze_all

On all other checkpoints, the sibling scores only 1–2pp above freeze_all and
5–10pp below oracle. The sibling is essentially predicting freeze_all for the
cases where a different span policy would help, because those cases have
different representation structure in other checkpoints.

### Stage 3 (combination holdout) side effect

Stage3-50k shows oracle at 79.5% vs stage2-50k oracle at 74.7% — a 5pp
improvement in how well the primary model can benefit from span-selective
freezing. This is an unexpected side effect of the combination holdout
training: the model's representations generalize better to unseen archetype
combinations, making the frozen-context integration problem easier. The
sibling misses this improvement (0.695 vs 0.795 oracle) because it was
not trained on stage3's representation regime.

### What this means for the sibling architecture

The sibling's task is representation-specific. Each checkpoint regime may
need its own sibling, trained on oracle data from that checkpoint. Sharing
a single sibling across checkpoints provides essentially no benefit over
the freeze_all heuristic.

The alternative — training a single sibling on pooled oracle data from
multiple checkpoints — is an open question. If the 7% of cases where
freeze_all is suboptimal share the same structural characteristics across
checkpoints, a pooled sibling could generalize. If those cases are
representation-specific, pooling won't help.

The multi-seed sibling experiment (training one sibling per balanced seed,
evaluating each against its own seed) tests the within-regime version of
this question: does the sibling consistently learn the oracle policy when
trained and evaluated on the same checkpoint?

---

## Connection to the 92.8% freeze_all base rate

The oracle labels are 92.8% freeze_all. A sibling that learns the base rate
perfectly would score 0.928 combo accuracy on val and match freeze_all exactly
downstream. The cross-checkpoint sibling scoring ~freeze_all+1pp is consistent
with it having learned the base rate but failing on the 7.2% edge cases — which
it can only identify reliably for the checkpoint it was trained on.

---

*Generated: 2026-03-30. Model: claude-sonnet-4-6.*
