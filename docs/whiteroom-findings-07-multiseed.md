# Whiteroom — Multi-Seed Analysis and What It Reveals

## Overview

Five seeds (1–5) were trained under the balanced-archetype + co-occurrence-damped
regime (Stage 2, 50k steps). Three eval tasks were run on every final checkpoint:
cache freeze, attribution, and property-append. A strategy benchmark was then run
comparing sibling-guided freeze against static patterns and oracle.

This document covers what the multi-seed results mean for the research arc as a
whole, including a significant limitation of the property-append experiment that
only became apparent at this stage.

---

## Training

All seeds: `--steps 50000 --batch-size 64 --balance-archetypes --cooccurrence-damp 0.7 --seed N`

Final loss (step 50k):

| Seed | Total | Seq | Attr |
|------|-------|-----|------|
| 1 | 0.3271 | 0.0906 | 0.0180 |
| 2 | 0.3432 | 0.0914 | 0.0175 |
| 3 | 0.3363 | 0.0928 | 0.0196 |
| 4 | 0.3408 | 0.0908 | 0.0192 |
| 5 | 0.3367 | 0.0903 | 0.0196 |

Mean seq loss: 0.0912 ± 0.0010. Training is stable and low-variance across seeds.

---

## Cache Freeze Results (n=300 per seed)

### A-frozen (A's encoder output frozen, B+BIND from fresh encode)

| Seed | Frozen seq acc | Degradation | Cosine sim |
|------|---------------|-------------|------------|
| 1 | 0.8667 | +0.033 | 0.742 |
| 2 | 0.9900 | −0.007 | 0.881 |
| 3 | 0.9633 | +0.017 | 0.823 |
| 4 | 0.9133 | +0.013 | 0.834 |
| 5 | 0.9633 | +0.020 | 0.837 |
| **mean** | **0.939 ± 0.049** | **+0.015 ± 0.015** | **0.823 ± 0.051** |

### B-frozen (B's encoder output frozen, A+BIND from fresh encode)

| Seed | Frozen seq acc | Degradation | Cosine sim |
|------|---------------|-------------|------------|
| 1 | 0.9333 | 0.000 | 0.810 |
| 2 | 0.9733 | 0.000 | 0.857 |
| 3 | 0.9700 | −0.010 | 0.765 |
| 4 | 0.9433 | 0.000 | 0.843 |
| 5 | 0.9733 | 0.000 | 0.826 |
| **mean** | **0.959 ± 0.019** | **−0.002 ± 0.005** | **0.820 ± 0.036** |

### Interpretation

Cache freeze is the most reproducible result. B-frozen is near-zero degradation
across all seeds (±0.005 on degradation, essentially zero). A-frozen shows a small
positive degradation with more variance — seed 1 is the outlier (0.867 frozen acc,
0.742 cosine sim), suggesting its representations are less geometrically stable.

**This finding is robust.** The model learns semantic isolation between components
as a consequence of the composition training objective, not as an artifact of a
single initialization.

---

## Attribution Results (n=300 per seed)

| Seed | Seq exact match | Token accuracy |
|------|----------------|----------------|
| 1 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 1.0000 |
| 3 | 0.9967 | 0.9995 |
| 4 | 1.0000 | 1.0000 |
| 5 | 1.0000 | 1.0000 |
| **mean** | **0.999 ± 0.002** | **1.000 ± 0.000** |

Near-perfect and essentially zero variance across seeds. Attribution is the most
stable capability — once the architecture can learn it, every initialization learns
it perfectly. Seed 3 (0.997) is a rounding artifact on 300 samples (1 error).

---

## Property-Append Results (n=300 per seed)

### Hybrid pickup and full-fresh rates

| Seed | Hybrid pickup | Full-fresh |
|------|--------------|------------|
| 1 | 0.627 | 0.703 |
| 2 | 0.340 | 0.423 |
| 3 | 0.213 | 0.297 |
| 4 | 0.637 | 0.657 |
| 5 | 0.400 | 0.457 |
| **mean** | **0.443 ± 0.185** | **0.507 ± 0.169** |

### Flag preservation in hybrid condition

| Metric | Mean | ±Std |
|--------|------|------|
| A-flags preserved | 0.937 | 0.016 |
| B-flags preserved | 0.932 | 0.017 |

### Interpretation

Hybrid pickup has extremely high variance (±0.185 on a 0–1 scale). The
single-run result reported in Stage 3a (69.2%) was an optimistic seed — the
multi-seed mean is 44.3%, with seed 3 as low as 21.3%.

This is not a flaw in the experiment design — it is information. The property-append
behaviour is an **emergent side effect** of composition training, not a learned
capability. Some initializations happen to produce representations tolerant of
frozen-context-plus-novel-token; others do not. The variance is architectural noise.

Flag preservation (existing A/B flags surviving the hybrid decode) is low-variance
and high (0.93+), meaning the frozen context is stable — the failure mode is
specifically **failure to pick up the new token**, not corruption of the old state.

---

## What the Property-Append Experiment Actually Measures

This is the critical limitation that only became clear at multi-seed scale.

In the hybrid condition:
1. The extended sequence `[A | BIND | B | extra_flag]` is **fully encoded** by the
   encoder with bidirectional attention across all positions.
2. The extra_flag's memory vector at position L has attended to A, BIND, and B
   during encoding — it is a contextually-rich representation.
3. This vector is then appended to the frozen base cache.

**In a real frozen KV cache, step 1 is not available.** The encoder ran over
positions 0..L-1 and cached the results. When a new token arrives, it cannot
re-attend to A and B — the encoder is not called again. The new token's
representation would have to be produced either in isolation, or with only a
narrow re-encoding window.

What the experiment actually measures: *can the decoder integrate a
contextually-encoded extra_flag appended to a stale base context?*

This is a strictly easier problem than the true frozen-cache scenario, where the
new token would only have access to its own embedding and positional encoding.
The 44.3% hybrid pickup rate is therefore an **upper bound** on what a real
frozen-cache editing mechanism could achieve without architectural changes.

**The Stage 1 cache freeze result is unaffected by this limitation.** Freezing
A's or B's span during the composition of a different compound is a legitimate
test — the fresh encode of `[A|BIND|C]` is a genuine fresh encode, and the
frozen A positions are taken from a genuinely different prior context. That
finding (zero degradation, low variance) stands.

---

## Strategy Benchmark: Sibling vs Static Patterns (n=300 per seed)

Note: the sibling used here was trained on a single prior checkpoint
(stage2-attribution-50k, seed 42) and evaluated on the 5 balanced seeds —
a cross-checkpoint generalization test. Per-seed results from training one
sibling per seed are in whiteroom-findings-sibling-xcheckpoint.md and
the pending multi-seed sibling experiment.

The sibling was trained on oracle data from a single prior checkpoint
(stage2-attribution-50k, seed 42). It was evaluated against the 5 balanced
multi-seed checkpoints — a cross-checkpoint generalization test.

| Strategy | Mean | ±Std | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|
| full_fresh | 0.489 | 0.175 | 0.690 | 0.390 | 0.280 | 0.650 | 0.433 |
| freeze_all | 0.419 | 0.183 | 0.587 | 0.327 | 0.187 | 0.623 | 0.373 |
| freeze_A_only | 0.461 | 0.173 | 0.637 | 0.370 | 0.243 | 0.637 | 0.420 |
| freeze_B_only | 0.471 | 0.182 | 0.663 | 0.383 | 0.250 | 0.657 | 0.403 |
| freeze_A_B | 0.443 | 0.172 | 0.590 | 0.363 | 0.217 | 0.637 | 0.407 |
| oracle | 0.501 | 0.183 | 0.710 | 0.403 | 0.280 | 0.670 | 0.440 |
| sibling | 0.424 | 0.185 | 0.600 | 0.337 | 0.187 | 0.623 | 0.373 |

### Key findings

**The sibling does not generalize across checkpoints.** Trained on a single
prior checkpoint, the sibling scores 0.424 mean — indistinguishable from
freeze_all (0.419) and far below oracle (0.501). It matched oracle exactly
on the checkpoint it was trained on (Stage 3c); that precision does not
transfer to new checkpoints.

**The oracle barely beats full_fresh.** oracle=0.501 vs full_fresh=0.489 —
a gap of 1.2pp. The span-level freeze policy is nearly worthless: once
you're re-encoding the extra_flag position (which all strategies do), the
marginal value of optimally freezing vs re-encoding A and B spans is tiny.
The bottleneck is not the freeze policy — it is the decoder's ability to
integrate the novel token at all, which varies enormously by seed.

**freeze_B_only is the best static heuristic** (0.471), slightly beating
freeze_A_only (0.461) and substantially beating freeze_all (0.419). This
is consistent with the cache freeze results: B-frozen has lower degradation
than A-frozen, suggesting B's representations are more positionally stable.

**The variance dominates everything.** All strategies show ±0.17–0.185 std.
No strategy — including oracle — reduces the seed-to-seed variance. This
confirms the bottleneck is in the primary model's representations, not in
the freeze policy.

---

## Revised Research Arc

| Stage | Finding | Variance | Status |
|-------|---------|----------|--------|
| 1 | Cache freeze: zero degradation when A or B frozen | Low (±0.019 B-frozen) | **Robust** |
| 2 | Attribution: near-perfect, no interference | Near-zero (±0.002) | **Robust** |
| 3a | Property-append: hybrid pickup 44.3% mean | **High (±0.185)** | Partially revised |
| 3b/c | Sibling learns perfect policy on training checkpoint | N/A (single checkpoint) | Holds within regime |
| Multi-seed | Sibling does not generalize; oracle ≈ full_fresh | High | New finding |

### What is and isn't solid

**Solid**: The model learns compositional semantics that naturally produce
encoder representations where component spans are geometrically independent.
Freezing one component's cache while re-encoding a new compound produces
essentially zero degradation. This is a reliable emergent property.

**Solid**: Attribution is perfectly learnable and stable across seeds.

**Revised**: The property-append result (69.2% from Stage 3a) was a single
lucky seed. The multi-seed mean is 44.3% ± 18.5%. The capability is real but
unreliable. More importantly, the experiment measures an easier version of the
true frozen-cache problem — in practice, the new token would not have had the
benefit of encoding with full attention to the existing context.

**New**: The span freeze policy (sibling, oracle) adds almost no value over
full_fresh or a simple static heuristic. The bottleneck is the primary model's
decoder tolerance for novel tokens in frozen context, not the freeze policy.

---

## Implications for Stage 4 (Curriculum Fine-tuning)

The curriculum approach becomes more motivated, and more precisely targeted, in
light of these findings.

The goal of Stage 4 is to explicitly train decoder tolerance for frozen context.
The curriculum as designed (detach A+BIND+B positions, train on extra_flag
integration) addresses the right thing.

However, to be faithful to the real frozen-cache problem, the curriculum should
encode the extra_flag **in isolation or with a narrow window** rather than with
full sequence attention. If the extra_flag position is encoded with access to
A and B, the curriculum is training on an easier version of the problem — the
same as the Stage 3a measurement artifact.

A faithful curriculum: embed the extra_flag token with positional encoding only
(no cross-attention to A/B), then train the decoder to integrate it from the
frozen base context. This is harder and more realistic.

The experimental comparison then becomes:
- **Stage 2** (n=5 seeds): emergent tolerance, 44.3% ± 18.5%
- **Stage 4a** (full-attention curriculum): upper bound — how well can tolerance
  be trained under favourable conditions?
- **Stage 4b** (isolated-token curriculum): realistic conditions — how well can
  the decoder integrate a truly uncontextualised new token?

If Stage 4a improves mean and reduces variance substantially, tolerance is
trainable. If Stage 4b cannot close the gap to Stage 4a, the bottleneck shifts
to the encoder — you need some form of re-encoding to make the new token useful,
and frozen KV caches can't provide it without architectural changes.

---

## Per-Seed Sibling Benchmark (n=300 per seed, with latency)

Each seed's sibling trained exclusively on oracle data from its own checkpoint.
Both full format `[old | SEP | new]` and compact format `[old | SEP | pos_tok | flag_tok]` evaluated.

### Benchmark results

| Strategy | Mean | ±Std | ms/pair | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|---|
| freeze_all | 0.419 | 0.183 | 18ms | 0.587 | 0.327 | 0.187 | 0.623 | 0.373 |
| full_fresh | 0.489 | 0.175 | 18ms | 0.690 | 0.390 | 0.280 | 0.650 | 0.433 |
| oracle | 0.501 | 0.183 | **195ms** | 0.710 | 0.403 | 0.280 | 0.670 | 0.440 |
| sibling_full | 0.501 | 0.183 | 19ms | 0.710 | 0.403 | 0.280 | 0.670 | 0.440 |
| sibling_compact | 0.498 | 0.184 | 19ms | 0.710 | 0.400 | 0.280 | 0.667 | 0.433 |

### Seed-dependency: does each sibling match oracle on its own seed?

| Seed | Oracle | Full gap | Compact gap | freeze_all gap |
|------|--------|----------|-------------|----------------|
| 1 | 0.710 | +0.000 | +0.000 | −0.123 |
| 2 | 0.403 | +0.000 | −0.003 | −0.077 |
| 3 | 0.280 | +0.000 | +0.000 | −0.093 |
| 4 | 0.670 | +0.000 | −0.003 | −0.047 |
| 5 | 0.440 | +0.000 | −0.007 | −0.067 |

**Sibling (full format) matches oracle exactly on every seed** — gap=0.000 across all 5.
Compact format is within 0–7 thousandths. Both strongly outperform freeze_all (−0.047 to −0.123 gap).

Val accuracy at convergence (2000 steps, 2000 training samples):
- Seeds 1–4 full: 1.000. Seed 5 full: 0.987 (did not converge — the one exception).
- Seeds 1, 3–4 compact: 1.000. Seed 2 compact: 0.997. Seed 5 compact: 0.999 (post-patch).
- Compact converges faster: seed 1 hit 1.000 at step 1100 vs step 1500 for full.

### Key findings

**The sibling learns a perfect span policy when trained on its own checkpoint.**
This confirms the cross-checkpoint failure seen earlier is not a capacity issue —
the model can represent the correct policy, but the policy is checkpoint-specific.
Each primary model's oracle distribution is shaped by that checkpoint's particular
representations, and 2000 samples is enough to learn it exactly.

**Compact format is competitive with full format** at a fraction of the input length
(~15 tokens vs ~26). It converges faster and scores within measurement noise.
The compact format encodes the edit description (position + token), not the full new
sequence — discarding redundant tokens appears to sharpen the learning signal.

**Sibling latency: 19ms vs oracle 195ms** — 10× faster. The sibling replaces
brute-force search over 8 span combinations (each requiring encoder + decoder pass)
with a single forward pass.

---

## Scaling Hypothesis: When Does the Span Policy Matter?

At the current scale, the span freeze policy contributes almost nothing to accuracy:

- oracle − full_fresh = 1.2pp (0.501 vs 0.489)
- sibling = oracle (gap 0.000)
- sibling − full_fresh ≈ 1.2pp

The policy is learned correctly (sibling = oracle) but the correct policy barely
helps over re-encoding everything. The bottleneck at this scale is **decoder
tolerance** — the primary model's ability to integrate a novel token in frozen
context — not the freeze decision itself.

This changes at scale for two independent reasons:

**1. Re-encode cost grows with sequence length.**
At this scale, old and new sequences are 13–26 tokens; a full re-encode is ~18ms.
At realistic KV-cache operating scales (hundreds to thousands of cached tokens),
re-encoding the full sequence is the expensive operation. A 19ms sibling pass
that selects which spans genuinely need refreshing becomes valuable precisely
because it avoids that cost. The compute savings are proportional to sequence
length, while the sibling cost stays roughly constant (it only sees the diff description).

**2. Oracle gap may widen if decoder tolerance improves.**
If Stage 4 curriculum training raises decoder tolerance — the primary model
learns to integrate frozen context with novel tokens — the cases where span
selection actually matters (freeze A, re-encode B vs freeze both) will become
more distinguishable. Currently, neither option works well enough for the
difference to show. With a more capable decoder, the optimal freeze mask has
higher value, and the gap oracle − full_fresh could grow substantially.

**Summary**: at this scale the experiment answers "does the sibling learn the
right policy?" (yes). The question "does the right policy matter?" requires
either a larger primary model or Stage 4 improvements to decoder tolerance.
Both are on the research arc.

---

*Updated: 2026-03-31. Model: claude-sonnet-4-6.*
