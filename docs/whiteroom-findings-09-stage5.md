# Findings 09 — Stage 5 & 5b: From-Scratch Adaptive Curriculum + Unfrozen Fine-Tuning

## Overview

Stage 5 trains all 5 seeds from scratch with an adaptive freeze curriculum baked in from the start, using a shared multiprocessing data pool (one SharedDataServer feeding all 5 seed processes simultaneously). Stage 5b then applies unfrozen curriculum fine-tuning on top of Stage 5 checkpoints, testing whether the decoder can improve pickup without disturbing the encoder geometry.

---

## Stage 5: From-Scratch Adaptive Curriculum

### Motivation

Stage 4 post-hoc curriculum fine-tuning showed a ceiling: encoder representations entangled during Stage 2 training couldn't be fully detangled afterward. Seeds 2 and 3 were persistently stubborn (pickup 0.55–0.73 across Stage 4/4b/4c) despite strong freeze accuracy. Stage 4c's longer partial-freeze pass even regressed seed 3. Conclusion: two-stage training has a fundamental limitation — the encoder converges to a solution before the curriculum ever runs.

Stage 5 hypothesis: if encoder and decoder co-evolve under freeze conditions from step 1, the encoder will find a naturally disentangled solution.

### Setup

- Architecture: same WhiteroomTransformer (379,844 params, d_model=64, 3 enc/dec layers)
- Adaptive curriculum: Phase 1 (partial freeze) until curr loss slope plateaus → Phase 2 (full freeze) until plateau → done
- Plateau detection: linear regression slope over 10-interval rolling window, threshold 5e-5
- Shared data: one SharedDataServer (16 CPU workers, 3 queues) feeds all 5 seed processes
- 5 seeds trained in parallel on one GPU; GPU utilization: 99%

### Training trajectory

All 5 seeds converged cleanly together, curr loss dropping from ~0.55 → ~0.005 (phase 1 plateau) → ~0.002 (phase 2 plateau).

Phase 1→2 transitions (step):

| Seed | Transition step | Final step | Final curr |
|------|----------------|------------|------------|
| 1 | ~28,500 | 28,500 | 0.0027 |
| 2 | ~30,000 | 35,500 | 0.0031 |
| 3 | ~18,500 | 30,000 | 0.0036 |
| 4 | ~31,000 | 33,000 | 0.0027 |
| 5 | ~30,500 | 36,000 | 0.0025 |

Seed 3 transitioned earliest but with the highest final curr — plateaued faster rather than converging harder. Seeds 2 and 5 took longest. All seeds finished in 28.5–36k steps.

**Efficiency**: Stage 5 total ≈ 50k steps (Stage 5 ~30–35k + Stage 5b 20k) vs Stage 2+4b ≈ 70k (50k + 20k). Roughly 30% fewer steps for equivalent or better results.

### Eval results

| Seed | A-frozen | B-frozen | cos-A | pickup | full-fresh | attr |
|------|----------|----------|-------|--------|------------|------|
| 1 | 1.000 | 0.983 | 0.921 | 0.903 | 0.933 | 0.970 |
| 2 | 0.953 | 0.930 | 0.905 | 0.687 | 0.763 | 0.977 |
| 3 | 0.977 | 0.910 | 0.866 | 0.667 | 0.730 | 0.933 |
| 4 | 0.953 | 0.940 | 0.861 | 0.700 | 0.813 | 0.973 |
| 5 | 1.000 | 0.970 | 0.905 | 0.893 | 0.940 | 0.983 |
| **mean** | **0.977** | **0.947** | **0.892** | **0.770** | **0.836** | **0.967** |
| **std** | 0.023 | 0.030 | 0.026 | 0.118 | 0.097 | 0.020 |

**Freeze** is dramatically better than Stage 2 baseline (~0.82 A-frozen) and better than Stage 4b for most seeds. Near-zero degradation across the board.

**Pickup variance remains high** (±0.118). Seeds 1 and 5 are excellent (0.90+), seeds 2, 3, 4 are moderate (0.67–0.70). From-scratch training equalized the seeds somewhat (seed 3 recovered from 0.527 in Stage 4c to 0.667) but didn't eliminate variance.

**Seed 4 anomaly**: dropped from 0.963 pickup in Stage 4b to 0.700 in Stage 5. Stage 4b seed 4 likely found an unusually favorable attractor from its Stage 2 initialization that the from-scratch curriculum didn't reproduce.

### Cross-stage comparison (Stage 4b → Stage 5)

Relative seed ranking is **not preserved** on pickup — seed 4 collapsed from top to mid-pack, seed 3 recovered from worst to mid-pack. This implies the solution space is stochastic and the two training approaches sample different attractors, not that seeds have inherent fixed quality levels.

Freeze results are more consistent: mostly positive or neutral deltas across seeds, suggesting the from-scratch curriculum reliably improves geometric separation regardless of initialization.

---

## Stage 5b: Unfrozen Curriculum Fine-Tuning

### Motivation

Stage 5 models have strong freeze geometry. Hypothesis: the adaptive curriculum may have left the decoder in a slightly constrained state (it never trained on fully unfrozen encoder output). An unfrozen fine-tuning pass might let the decoder clean up and improve pickup while the encoder geometry remains stable — since the encoder has no gradient pressure to entangle (high freeze accuracy means the frozen-span structure is already near-optimal for the curriculum loss).

### Setup

- Start from Stage 5 `checkpoint_final.pt` for each seed
- `finetune_curriculum.py` without `--partial-freeze`: full model trainable
- 20k steps, lr=1e-4, curriculum_prob=0.5, 3 workers per seed, all 5 seeds parallel
- **Bug fixed**: `DataPrefetcher.get_comp()` / `get_attr()` put-back used unbounded `queue.put()`, causing deadlock when queue was full. Fixed with `timeout=0.1` + drop on Full.

### Eval results

| Seed | A-frozen | B-frozen | cos-A | pickup | full-fresh | attr |
|------|----------|----------|-------|--------|------------|------|
| 1 | 0.980 | 0.990 | 0.900 | 0.883 | 0.937 | 0.990 |
| 2 | 0.997 | 0.967 | 0.903 | 0.653 | 0.767 | 1.000 |
| 3 | 0.947 | 0.933 | 0.851 | 0.597 | 0.680 | 0.997 |
| 4 | 0.947 | 0.950 | 0.848 | 0.690 | 0.793 | 0.997 |
| 5 | 0.970 | 0.940 | 0.851 | 0.777 | 0.813 | 0.997 |
| **mean** | **0.968** | **0.956** | **0.871** | **0.720** | **0.798** | **0.996** |
| **std** | 0.022 | 0.023 | 0.028 | 0.112 | 0.093 | 0.004 |

### Stage 5 → 5b delta

| Metric | Stage 5 | Stage 5b | Δ |
|--------|---------|---------|---|
| A-frozen | 0.977 | 0.968 | -0.009 |
| B-frozen | 0.947 | 0.956 | +0.009 |
| cos sim (A) | 0.892 | 0.871 | -0.021 |
| pickup | 0.770 | 0.720 | **-0.050** |
| full-fresh | 0.836 | 0.798 | -0.038 |
| attribution | 0.967 | **0.996** | **+0.029** |

### Interpretation

Unfrozen fine-tuning **improved attribution to near-perfect (0.996)** but **hurt pickup across almost every seed**. Freeze accuracy held roughly steady but cosine similarity drifted down, suggesting the encoder representations shifted slightly. Even small shifts in the encoder geometry appear to be enough to reduce pickup, despite not meaningfully changing freeze accuracy.

This suggests attribution and pickup are somewhat in tension under unfrozen training:
- Attribution rewards accurate regular composition, pulling encoder toward cleaner general representations
- Pickup requires specific geometric separation that the unfrozen curriculum erodes slightly, even when freeze accuracy is maintained

The hypothesis that "the decoder would clean up without touching the encoder" did not hold. The encoder does move, and the movement hurts pickup even when it helps attribution.

### Lesson

For models that already have strong freeze geometry (Stage 5), unfrozen fine-tuning is counterproductive for pickup. If the goal is higher pickup, the approach should either (a) freeze the encoder and only train the decoder, or (b) include explicit geometric regularization. The Stage 5 checkpoint is the better starting point for downstream use.

---

## Compute summary

| Stage | Steps (per seed) | Cumulative |
|-------|-----------------|------------|
| Stage 2+4b | 50k + 20k = 70k | 70k |
| Stage 2+4c | 50k + 40k = 90k | 90k |
| Stage 5 only | ~30–35k | ~32k |
| Stage 5+5b | ~30–35k + 20k | ~52k |

Stage 5+5b uses ~25% less compute than Stage 2+4b with better freeze results, comparable attribution, and slightly lower pickup mean (but much better seed 3 recovery). Stage 5 alone is the most compute-efficient checkpoint with strong freeze and acceptable pickup.

---

## Stage 5c: Frozen-Encoder Curriculum Fine-Tuning

### Motivation

Stage 5b showed that unfrozen fine-tuning improved attribution but hurt pickup — the encoder drifted slightly, eroding the geometric separation needed for frozen-cache decoding. Stage 5c tests the cleaner version: freeze the encoder entirely (`src_embed` + `transformer.encoder` weights set `requires_grad=False`) and train only the decoder + heads on the curriculum. No encoder gradient = no geometry drift possible.

### Setup

- Start from Stage 5 `checkpoint_final.pt`
- Encoder fully frozen; only decoder, heads, and tgt_embed trainable
- 20k steps, lr=1e-4, curriculum_prob=0.5, 3 workers per seed, all 5 seeds parallel
- ~75 min wall time (comparable to 5b's ~74 min — encoder backward skip offset by slightly slower step throughput)

### Eval results

| Seed | A-frozen | B-frozen | cos-A | pickup | full-fresh | attr |
|------|----------|----------|-------|--------|------------|------|
| 1 | 1.000 | 0.990 | 0.921 | 0.830 | 0.863 | 0.993 |
| 2 | 1.000 | 0.967 | 0.905 | 0.677 | 0.750 | 1.000 |
| 3 | 0.977 | 0.960 | 0.866 | 0.567 | 0.663 | 0.990 |
| 4 | 0.960 | 0.977 | 0.861 | 0.647 | 0.707 | 0.990 |
| 5 | 1.000 | 0.983 | 0.905 | 0.773 | 0.803 | 0.997 |
| **mean** | **0.987** | **0.975** | **0.892** | **0.699** | **0.757** | **0.994** |
| **std** | 0.018 | 0.012 | 0.026 | 0.104 | 0.079 | 0.004 |

### Three-way comparison: Stage 5 → 5b → 5c

| Metric | Stage 5 | Stage 5b | Stage 5c | Best |
|--------|---------|---------|---------|------|
| A-frozen | 0.977 | 0.968 | **0.987** | 5c |
| B-frozen | 0.947 | 0.956 | **0.975** | 5c |
| cos sim | 0.892 | 0.871 | **0.892** | 5 / 5c |
| pickup | **0.770** | 0.720 | 0.699 | 5 |
| full-fresh | **0.836** | 0.798 | 0.757 | 5 |
| attribution | 0.967 | **0.996** | 0.994 | 5b / 5c |

### Interpretation

Freezing the encoder successfully protected the geometry — A-frozen and B-frozen accuracy both improved over Stage 5 and 5b, and cosine similarity held at Stage 5 levels. Attribution also held near-perfect (0.994). However, **pickup dropped further** (0.770 → 0.699), continuing the trend from 5b.

This confirms that pickup improvement requires the encoder to move. The decoder alone — whether trained unfrozen (5b) or with a hard-frozen encoder (5c) — cannot improve pickup. The improvement from 5b→5c is in freeze and attribution quality, not in pickup.

### Lesson

The three fine-tuning variants tell a consistent story:
- **Unfrozen (5b)**: encoder drifts → attribution improves, freeze holds roughly, pickup drops
- **Frozen encoder (5c)**: geometry perfectly preserved, attribution near-perfect, pickup drops most
- **Stage 5 base**: best pickup, because the encoder and decoder co-evolved together under freeze conditions

The implication: pickup is a joint encoder-decoder property that emerges from co-evolution under the curriculum. Post-hoc fine-tuning of any kind degrades pickup because it decouples what the curriculum joined together. **Stage 5 is the canonical checkpoint** for downstream use.

---

## Compute summary (updated)

| Stage | Steps (per seed) | Cumulative |
|-------|-----------------|------------|
| Stage 2+4b | 50k + 20k = 70k | 70k |
| Stage 2+4c | 50k + 40k = 90k | 90k |
| Stage 5 only | ~32k | ~32k |
| Stage 5+5b | ~32k + 20k | ~52k |
| Stage 5+5c | ~32k + 20k | ~52k |
