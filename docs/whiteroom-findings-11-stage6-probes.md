# Findings 11 — Stage 6: Memory Manipulation Probes

## Overview

Stage 6 is a zero-cost analysis pass on existing checkpoints (Stage 2, Stage 5). All tests operate directly on frozen encoder memory tensors — no new training. Five probes:

1. **Memory splice swap** (position independence) — documented in Findings 10
2. **Duplicate A** — replace B's positions with A's representations
3. **Shuffle A** — randomly permute token order within A's encoder segment
4. **Content corruption** — add scaled Gaussian noise to A's representations
5. **Binding direction** — split freeze tests by binding direction (A.out→B.in vs A.in←B.out)
6. **Cross-attention patterns** — capture decoder cross-attention weights under normal vs spliced memory

Together these probes characterize *how* the decoder reads frozen representations: what it depends on, what it ignores, and how it routes attention.

---

## Probe 1: Duplicate A

### Setup

For each equal-length (A, B) pair:
- Normal decode: `[A|BIND|B]` → `compound(A,B)`
- Duped decode: replace B's encoder output positions with A's representations → decode

Since the target is still `compound(A,B)`, duped accuracy measures how often A+A accidentally matches the correct compound.

### Results (Stage 5, 5 seeds, n=300 each)

| Seed | normal_acc | duped_acc | agreement | cost |
|------|-----------|-----------|-----------|------|
| 1 | 1.000 | 0.317 | 0.317 | +0.683 |
| 2 | 0.947 | 0.233 | 0.233 | +0.713 |
| 3 | 0.903 | 0.187 | 0.283 | +0.717 |
| 4 | 0.903 | 0.103 | 0.200 | +0.800 |
| 5 | 0.990 | 0.087 | 0.087 | +0.913 |
| **mean** | **0.949** | **0.185** | **0.224** | **+0.765** |

Stage 2 (5 seeds, n=300 each):

| Seed | normal_acc | duped_acc | agreement | cost |
|------|-----------|-----------|-----------|------|
| 1 | 0.850 | 0.000 | 0.150 | +0.850 |
| 2 | 0.947 | 0.000 | 0.053 | +0.947 |
| 3 | 0.897 | 0.000 | 0.053 | +0.897 |
| 4 | 0.897 | 0.000 | 0.103 | +0.897 |
| 5 | 0.947 | 0.000 | 0.053 | +0.947 |
| **mean** | **0.908** | **0.000** | **0.082** | **+0.908** |

### Interpretation

Accuracy collapses to **0.09–0.32** (mean 0.185) from the ~0.95 baseline. The decoder cannot compose A with a duplicate of itself to produce `compound(A,B)`.

The residual duped_acc (~0.185) is not the decoder "doing its best" — note that `agreement ≈ duped_acc`, meaning whenever the model produces the correct answer with duplicated A, it's producing the *same* output as normal. These are cases where `compound(A,A)` accidentally overlaps with `compound(A,B)` in output space (e.g., when A and B share flags). The model isn't partially solving the problem; it's consistently producing the A+A compound, which occasionally happens to match the target.

**Implication**: the decoder identifies which entity is in each slot by the semantic content of the representations, not by position. Putting the same entity in both slots is not a graceful degradation — it's a fundamentally different input that the model handles consistently (wrong, but consistently so).

---

## Probe 2: Shuffle A Tokens Within Segment

### Setup

Randomly permute the token positions within A's encoder output span (B and BIND untouched). Run on pairs of any length (not just equal-length).

### Results (Stage 5, 5 seeds, n=300 each)

| Seed | normal_acc | shuffle_acc | agreement | cost |
|------|-----------|-------------|-----------|------|
| 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| 2 | 0.973 | 0.973 | 1.000 | 0.000 |
| 3 | 0.940 | 0.940 | 1.000 | 0.000 |
| 4 | 0.950 | 0.950 | 1.000 | 0.000 |
| 5 | 0.990 | 0.990 | 1.000 | 0.000 |
| **mean** | **0.971** | **0.971** | **1.000** | **0.000** |

### Interpretation

**Zero cost. Perfect agreement.** Token ordering within A's segment is completely irrelevant to the decoder.

This reveals a key architectural fact: the bidirectional encoder distributes A's identity across all of its token positions simultaneously. Each position in A's encoder output contains information about the entire A entity (not just its local token), because self-attention has already pooled bidirectionally over the full A segment. The decoder then aggregates over these positions — since each one contains the full entity's information, their ordering doesn't matter.

Combined with the cross-attention findings (see below), this means the decoder reads A as a *distributed semantic cloud*, not as a sequential token stream.

---

## Probe 3: Content Corruption

### Setup

Add Gaussian noise to A's encoder output positions, scaled to each token's L2 norm:
`noise = N(0, σ² · ||repr||²)` per token. Sweep σ ∈ {0, 0.1, 0.25, 0.5, 1.0, 2.0}.

### Results (Stage 5, mean across 5 seeds, n=300 each)

| σ | mean accuracy |
|---|---------------|
| 0.0 | 0.953 |
| 0.1 | 0.879 |
| **0.25** | **0.011** |
| 0.5 | 0.000 |
| 1.0 | 0.000 |
| 2.0 | 0.000 |

Per-seed accuracy at σ=0.25:

| Seed | acc at σ=0.25 |
|------|--------------|
| 1 | 0.000 |
| 2 | 0.013 |
| 3 | 0.013 |
| 4 | 0.017 |
| 5 | 0.010 |

### Interpretation

The degradation curve is a **hard cliff**, not a ramp. From σ=0.1 to σ=0.25 — a factor of 2.5× in noise magnitude — accuracy drops from ~88% to ~1%. By σ=0.5 it's zero.

This means the representations are **high-precision and brittle**: they carry information in exact numeric values, not in coarse feature directions. The decoder is reading precise coordinates in representation space, not robust abstract features.

This stands in apparent tension with the shuffle result (Probe 2), which shows zero sensitivity to token order, and the splice result (Findings 10), which shows zero sensitivity to position. The decoder is *spatially* flexible (it doesn't care where representations live or in what order) but *numerically* rigid (it requires exact representation values).

The combined picture: the encoder has learned to encode entity identity as a high-dimensional point in representation space. The decoder has learned to read that point precisely. The *location* of that point in the memory tensor is irrelevant; the *value* of the point is everything.

---

## Probe 4: Binding Direction

### Setup

Filter (A, B, C) freeze triplets by the direction of the A-B binding:
- **Forward**: A has output port, B has input port (A pushes to B)
- **Reverse**: A has input port, B has output port (B pushes to A)

Run A-frozen test in both cases. Compare freeze accuracy and encoder cosine similarity.

### Results (Stage 5, 5 seeds)

| Seed | forward frozen | reverse frozen | forward cos | reverse cos |
|------|---------------|---------------|-------------|-------------|
| 1 | 1.000 | 1.000 | 0.900 | 0.934 |
| 2 | 0.952 | 1.000 | 0.899 | 0.916 |
| 3 | 0.945 | 0.974 | 0.829 | 0.884 |
| 4 | 0.903 | 0.974 | 0.822 | 0.880 |
| 5 | 1.000 | 1.000 | 0.918 | 0.900 |
| **mean** | **0.960** | **0.990** | **0.874** | **0.903** |

### Interpretation

**Reverse bindings freeze better than forward** (0.990 vs 0.960) and produce more stable encoder representations (cos sim 0.903 vs 0.874).

When A is the "sink" (input port, receiving from B), A's encoding is less entangled with B's specific content — it's describing what A accepts, not what A provides. This makes A's representation more stable across different B partners, which means the frozen cache is more accurate.

Seeds 3 and 4 show **negative freeze cost on reverse** (frozen_acc > normal_acc). The frozen cache is *more* accurate than the fresh encoding for these seeds. One interpretation: the frozen representation from a clean AB encoding is slightly more precise than the fresh representation from an AC encoding with a different partner, because the frozen cache saw the full training distribution of AB pairs.

---

## Probe 5: Cross-Attention Patterns

### Setup

Capture decoder cross-attention weights per layer per decoding step using forward hooks on `model.transformer.decoder.layers[i].multihead_attn`. Compare attention mass on A-region vs B-region between normal and position-spliced memory.

**content_follow_score** per layer: fraction of (A-region + B-region) attention mass that lands on whichever positions now contain A's representations.
- 1.0 = fully semantic (attention follows A's content to its new position)
- 0.0 = fully positional (attention stays at A's original positions regardless)

### Results — Stage 5

| Seed | Layer 0 | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| 1 | 0.615 | 0.729 | 0.795 |
| 2 | 0.591 | 0.899 | 0.553 |
| 3 | 0.540 | 0.630 | 0.916 |
| 4 | 0.486 | 0.763 | 0.922 |
| 5 | 0.670 | 0.739 | 0.683 |
| **mean** | **0.580** | **0.752** | **0.774** |

### Results — Stage 2

| Seed | Layer 0 | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| 1 | 0.788 | 0.824 | 0.799 |
| 2 | 0.740 | 0.880 | 0.794 |
| 3 | 0.645 | 0.817 | 0.900 |
| 4 | 0.634 | 0.751 | 0.928 |
| 5 | 0.722 | 0.867 | 0.845 |
| **mean** | **0.706** | **0.828** | **0.853** |

### Key observation: attention maps flip exactly

The core finding is not the content_follow_score itself but what underlies it: **`spliced_old_a_mass == normal_b_mass` and `spliced_old_b_mass == normal_a_mass` to the decimal, for every seed, every layer, every stage.** The attention map is performing an exact flip — mass that went to A positions in normal decoding now goes to B positions (where A's content lives) in spliced decoding, and vice versa.

The content_follow_score < 1.0 is entirely explained by attention mass on BIND tokens and other positions that aren't in the A+B budget — not by any residual positional bias within the A/B regions.

### Stage 2 vs Stage 5 comparison

Stage 2 has **higher content_follow scores** (mean ~0.80 vs ~0.70 for Stage 5). The freeze curriculum in Stage 5 introduces a slight positional bias absent in Stage 2's purely-compositional training. When the model is never taught to think about "A is always in positions 0:N," it learns to route by content more purely.

### Layer depth trend

Content_follow generally increases with layer depth, especially in Stage 5:
- Layer 0: most positionally mixed (0.49–0.79 across seeds/stages)
- Layers 1–2: more purely semantic (0.55–0.93)

Early decoder layers appear to do some mixed routing; later layers converge on content-based identification. This suggests a rough two-phase decoding process: early layers orient (position-influenced), later layers read (semantically).

---

## Cross-Stage Comparison (Stage 2 vs Stage 5)

### Duplicate A

| Stage | mean duped_acc | mean agreement |
|-------|---------------|----------------|
| Stage 2 | **0.000** | 0.082 |
| Stage 5 | 0.185 | 0.224 |

Stage 2 duped_acc is exactly zero across all 5 seeds — A+A never accidentally produces the correct compound(A,B) output. Stage 5 has ~18.5% residual overlap. This likely reflects that Stage 5's curriculum training produces representations where A's encoding is slightly more "averaged" over its composition partners, allowing occasional accidental overlap with the target. Stage 2's sharper representations produce a completely disjoint A+A output.

The agreement rate being nonzero (~8% for Stage 2) means the model still consistently outputs the same (wrong) thing with duplicated A — it just never overlaps with the correct answer.

### Shuffle A

Identical across both stages: cost=0.000, agreement=1.000 for all 10 seeds. This property does not depend on training stage or curriculum.

### Content Corruption

Mean accuracy at σ=0.25:

| Stage | mean acc at σ=0.25 |
|-------|-------------------|
| Stage 2 | 0.043 |
| Stage 5 | 0.011 |

Stage 2 is slightly more noise-tolerant at the cliff point. Both stages hit zero by σ=0.5. The cliff is sharp in both cases but Stage 5's representations are slightly more brittle — the curriculum training produces more precise but less robust representations.

### Binding Direction

| Stage | forward frozen | reverse frozen | direction gap |
|-------|---------------|---------------|---------------|
| Stage 2 (mean) | 0.946 | 0.955 | +0.009 |
| Stage 5 (mean) | 0.960 | 0.990 | +0.030 |

The reverse > forward pattern is present in Stage 2 but the gap is small (0.009). Stage 5's curriculum amplifies it substantially (0.030). The freeze curriculum specifically trains for accuracy under frozen conditions, and appears to benefit the "A=sink" case more — possibly because sink representations are more stable across compositions to begin with, so the curriculum can push them closer to perfect.

---

## Summary

| Probe | Finding |
|-------|---------|
| Duplicate A | Collapses to ~0.19 accuracy; decoder identifies entities by content — same entity twice is not a valid composition |
| Shuffle A | Zero cost, perfect agreement — intra-segment token order is entirely irrelevant |
| Corruption | Hard cliff at σ=0.25; representations are precise numeric points, not robust coarse features |
| Binding direction | Reverse bindings (A=sink) freeze better; frozen cache occasionally outperforms fresh encoding |
| Cross-attention | Attention maps flip exactly with memory splice; decoder routes by semantic content from Stage 2 onward |

**Unified picture**: the encoder maps each entity to a high-dimensional point in representation space. The decoder reads those points by semantic content, not by position or token order. The points are precise — small noise destroys them — but their location in memory is irrelevant. Semantic routing is a fundamental property of compositional training, present from Stage 2, slightly diluted but not broken by the freeze curriculum.

---

## Appendix: Raw attention map data

Full per-layer, per-head attention tensors for 3 example pairs per checkpoint are stored in:
- `_agent/cache/runs/probe_cross_attention_stage5.json`
- `_agent/cache/runs/probe_cross_attention_stage2.json`

These can be visualized as heatmaps (decoder step × source position) to inspect individual attention head behavior.
