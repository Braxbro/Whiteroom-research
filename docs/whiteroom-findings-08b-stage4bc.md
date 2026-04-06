# Findings 08b — Stage 4b & 4c: Extended Curriculum Fine-Tuning

## Overview

Stage 4 (Findings 08) established that post-hoc curriculum fine-tuning massively improved pickup (+33pp) without hurting freeze accuracy. However, seeds 2 and 3 remained significantly weaker than the rest (pickup 0.693 and 0.480 vs 0.880–0.917 for seeds 1, 4, 5). Stage 4b tried a two-phase curriculum to push further. Stage 4c applied even longer training to the two laggard seeds.

---

## Stage 4b: Two-Phase Curriculum

### Motivation

Stage 4 used a single undifferentiated curriculum pass. Stage 4b introduced a two-phase approach:
- **Phase 1 (partial freeze, 10k steps)**: curriculum batches with partial freeze — decoder sees a mix of frozen and live encoder positions, forcing it to integrate across context boundaries
- **Phase 2 (full freeze, 10k steps)**: curriculum batches with full freeze — decoder must predict the compound entirely from a frozen base cache

Hypothesis: phasing from partial to full freeze gives the decoder a gentler learning trajectory — first learning to handle the boundary condition, then hardening to the full frozen-cache scenario.

All 5 seeds. Fine-tuned from each seed's Stage 2 `checkpoint_final.pt`. lr=1e-4, curriculum_prob=0.5.

### Results

| Seed | A-frozen | B-frozen | cos-A | pickup | full-fresh | attr |
|------|----------|----------|-------|--------|------------|------|
| 1 | 0.910 | 0.953 | 0.817 | 0.930 | 0.943 | 1.000 |
| 2 | 0.990 | 0.937 | 0.919 | 0.713 | 0.780 | 1.000 |
| 3 | 0.930 | 0.880 | 0.856 | 0.553 | 0.650 | 1.000 |
| 4 | 0.967 | 0.907 | 0.872 | 0.963 | 0.970 | 0.997 |
| 5 | 0.990 | 0.927 | 0.890 | 0.960 | 0.990 | 1.000 |
| **mean** | **0.957** | **0.921** | **0.871** | **0.824** | **0.867** | **0.999** |
| **std** | 0.036 | 0.028 | 0.038 | 0.183 | 0.143 | 0.001 |

### Stage 4 → 4b delta

| Metric | Stage 4 | Stage 4b | Δ |
|--------|---------|---------|---|
| A-frozen | 0.951 | 0.957 | +0.006 |
| B-frozen | 0.923 | 0.921 | −0.002 |
| pickup | 0.773 | **0.824** | **+0.051** |
| full-fresh | — | 0.867 | — |
| attribution | 0.999 | 0.999 | ≈0 |

### Interpretation

Stage 4b improves mean pickup by +5pp over Stage 4. Seeds 1, 4, and 5 all push into the 0.93–0.96 range — essentially saturating what post-hoc curriculum can achieve for these seeds. Seeds 2 and 3 show only marginal improvement (seed 2: 0.693 → 0.713, seed 3: 0.480 → 0.553).

**The variance does not close.** std on pickup is essentially unchanged (0.187 → 0.183). The two-phase structure helps seeds that were already moving in the right direction, but doesn't break the structural ceiling for laggard seeds. This suggests the bottleneck for seeds 2 and 3 is not decoder tolerance (which the curriculum trains) but encoder geometry — specifically, the entanglement pattern that formed during Stage 2 training and is not accessible to post-hoc fine-tuning.

**Seed 4 anomaly appears**: Stage 4b seed 4 reaches 0.963 pickup, the highest of any seed. This checkpoint will become notable later — Stage 5 from-scratch training fails to reproduce it (see Findings 09), suggesting seed 4 found an unusual attractor during Stage 2 that the Stage 4b curriculum locked in.

---

## Stage 4c: Extended Curriculum for Laggard Seeds

### Motivation

Seeds 2 and 3 appear stuck. Stage 4c tests whether the ceiling is a function of training time: run a much longer curriculum on only these two seeds using the same two-phase structure but at 3× the duration — 30k steps partial freeze + 10k steps full freeze = 40k total.

### Setup

- Seeds 2 and 3 only (targeted at the laggards)
- Phase 1: partial freeze, 30k steps, curriculum_prob=0.5, lr=1e-4 with cosine decay
- Phase 2: full freeze, 10k steps, curriculum_prob=0.5, lr=5e-5 with cosine decay
- Fine-tuned from each seed's Stage 2 `checkpoint_final.pt` (not from Stage 4b)
- ~76 min per seed (sequential, not parallel)

### Results

| Seed | A-frozen | B-frozen | cos-A | pickup | full-fresh | attr |
|------|----------|----------|-------|--------|------------|------|
| 2 | 0.990 | 0.923 | 0.893 | 0.733 | 0.800 | 1.000 |
| 3 | 0.913 | 0.880 | 0.813 | 0.527 | 0.573 | 0.997 |

### Stage 4b → 4c delta (seeds 2 and 3 only)

| Seed | 4b pickup | 4c pickup | Δ | 4b A-frozen | 4c A-frozen | Δ |
|------|-----------|-----------|---|-------------|-------------|---|
| 2 | 0.713 | 0.733 | +0.020 | 0.990 | 0.990 | 0.000 |
| 3 | 0.553 | 0.527 | **−0.026** | 0.930 | 0.913 | −0.017 |

### Interpretation

2× more compute (40k vs 20k steps) produced essentially zero improvement for seed 2 (+0.020) and a **regression for seed 3** (−0.026 pickup, −0.017 A-frozen). Longer training makes seed 3 worse.

This is the clearest evidence yet that the seeds 2 and 3 ceiling is architectural, not computational. The encoder geometry entangled during Stage 2 training cannot be unwound by additional decoder-focused curriculum. Stage 4c is training the decoder on an encoder that was never organized for clean component separation — more training pressure just destabilizes what's already there.

**Lesson from 4b/4c**: post-hoc curriculum has a hard ceiling set by the base encoder geometry. For seeds that happened to develop separable representations during Stage 2 (seeds 1, 4, 5), the curriculum rapidly unlocks the capability. For seeds that didn't (seeds 2, 3), no amount of curriculum fine-tuning recovers it.

This conclusion directly motivates Stage 5: if the encoder geometry is the binding constraint, the only fix is to let the encoder and decoder co-evolve under freeze conditions from the start.

---

## Full Arc: Stage 4 → 4b → 4c

| Stage | Seeds | Steps | mean pickup | pickup std |
|-------|-------|-------|-------------|------------|
| Stage 2 (base) | 5 | 50k | 0.443 | 0.185 |
| Stage 4 | 5 | +20k | 0.773 | 0.187 |
| Stage 4b | 5 | +20k | 0.824 | 0.183 |
| Stage 4c | 2 only | +40k | 0.630\* | — |

\*Seeds 2 and 3 only; not comparable to full-seed mean.

Pickup improves rapidly from Stage 2 → Stage 4 → Stage 4b, then stops. The std is essentially frozen at 0.183–0.187 throughout — every stage lifts the floor but doesn't collapse the variance. The distribution of seed quality is stable across the entire post-hoc curriculum arc, confirming it as a structural property of Stage 2 initialization.
