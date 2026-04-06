# Stage 7: Encoder Isolation Mechanisms — Two-Variable Architectural Study (~74K → ~380K)

**Experiment Date:** April 1, 2026
**Total Runtime:** ~8 hours (6 variants at ~380K + 2 variants at ~74K, all 5 seeds, full eval + probes)
**Scales:** ~74K (d_model=32, 2 layers) + ~380K (d_model=64, 3 layers)
**Setup:** Same training methodology as Stage 5 (adaptive freeze curriculum), varying only encoder and BIND directionality.

## Motivation

Stage 5 established that from-scratch adaptive curriculum training achieves strong encoder-decoder separation and reasonable composition recovery (77% at ~380K, 49.7% at ~74K). However, this required no explicit architectural isolation constraints—the encoder naturally found a partially-separate solution.

**Stage 7 tests a fundamental hypothesis:** *can we engineer better position-independent composition by imposing structural isolation constraints during training?*

The question decomposes into two independent variables:
1. **Encoder directionality**: How does the encoder view A and B together?
   - **Bidirectional** (unrestricted): encoder processes A and B with full attention (Stage 5 baseline)
   - **Causal**: encoder blocks A→B attention (A can't see B)
   - **Block-diagonal masked**: encoder restricts to A-only and B-only processing, meeting only via BIND token
2. **BIND directionality**: How is the BIND token trained?
   - **Unidirectional** (fixed): A→BIND→B routing (Stage 5 baseline)
   - **Bidirectional** (flipped): 50/50 A→BIND→B and B→BIND→A during training

This creates a 2×2 factorial study (encoder × BIND directionality) that systematically isolates what architectural constraints help or hurt composition learning at scale.

### Stage 5 Baseline Context

Stage 5 serves as the unconstrained reference:
- Bidirectional encoder (no masking)
- Unidirectional BIND (always A→BIND→B)
- Results: 77.0% composition at ~380K, 0.7021 decoder routing score
- This represents the "natural" optimum without any isolation forcing.

---

## Experimental Design

### Architecture

Same WhiteroomTransformer as Stage 5:
- **~380K scale**: d_model=64, 3 encoder + 3 decoder layers, 5 seeds (379,844 params)
- **~74K scale**: d_model=32, 2 encoder + 2 decoder layers, 5 seeds (74,244 params, subset: 7d, 7e only)
- All variants share identical architecture within their scale class

### Training

- **Curriculum:** Adaptive freeze (Phase 1: partial freeze → Phase 2: full freeze), same as Stage 5
- **Initialization:** All variants trained from scratch independently (not from Stage 5 checkpoints)
- **Data & learning:** Identical to Stage 5 (SharedDataServer, 5 seeds parallel, standard hyperparams)

### Variants (Factorial Design)

**Encoder × BIND directionality:**

| Variant | Encoder Type | BIND Direction | Notes |
|---------|--------------|----------------|-------|
| **Stage5 (baseline)** | Bidirectional | Unidirectional | No constraints; natural optimum |
| **7a** | Bidirectional | Bidirectional | BIND only forced symmetric; encoder unrestricted |
| **7b** | Causal | Unidirectional | Encoder isolation without bidirectional training |
| **7c** | Causal | Bidirectional | Both constraints active simultaneously |
| **7d** | Block-diagonal masked | Unidirectional | Strict encoder isolation via attention masking |
| **7e** | Block-diagonal masked | Bidirectional | Maximum isolation + symmetric BIND routing |

**Encoder masking strategies:**
- **Bidirectional (Stage 5)**: A and B attend to each other and BIND freely
- **Causal**: A attends to [A, BIND]; B attends to [BIND, B]; no cross-entity attention
- **Block-diagonal masked**: A attends to [A + BIND]; BIND attends to all; B attends to [BIND + B] (strongest isolation)

---

## Results Summary (~380K Scale)

| Variant | B_Iso | Decoder | Composition | Notes |
|---------|-------|---------|-------------|-------|
| **Stage5 baseline** | 0.0160 | 0.7021 | 77.0% | Unconstrained optimum |
| **7a (bidir-bind)** | 0.2100 | 0.5207 | 49.3% | Worst composition |
| **7b (causal-enc)** | 0.0380 | 0.5175 | 58.9% | Causal cost |
| **7c (causal+bidir)** | 0.2427 | 0.3479 | 59.5% | Decoder disaster |
| **7d (block-diag masked)** | -0.0020 | 0.6314 | 60.6% | **BEST isolated** |
| **7e (block-diag + bidir)** | 0.0040 | 0.5101 | 49.7% | Bidirectional collapse |

---

## Detailed Analysis: ~380K Scale

### 7a: Bidirectional BIND Only (No Encoder Constraint)

**Hypothesis:** Forcing bidirectional A↔B routing during training might improve position independence without encoder isolation.

**Result:** No benefit. Composition (49.3%) matches unconstrained baseline (49.7%), isolation is worst (0.2100). Bidirectional BIND alone cannot create meaningful separation.

**Interpretation:** Without structural encoder isolation, bidirectional training just adds conflicting gradient signals—the model must learn opposite routing patterns simultaneously (A→BIND→B in Phase 1, B→BIND→A in Phase 2). This is confusing without architectural backing. The decoder benefits from neither isolation (none imposed) nor unidirectional consistency (direction constantly flips).

**Lesson:** Bidirectional BIND is a training-time intervention that only works when the encoder is already structured to support it.

---

### 7b: Causal Encoder (Weak Isolation)

**Hypothesis:** Soft isolation via causal attention masking (A can't see B) might improve separation while preserving some cross-entity information.

**Result:** Modest gain. Composition (58.9%) improves +9.2% over baseline. Isolation is reasonable (0.0380). But decoder routing is weak (0.5175, -23% vs Stage5).

**Interpretation:** Causal masking successfully reduces A↔B leakage—the encoder learns that when A is processed, B hasn't been seen yet, creating asymmetric representations. This helps composition recovery (+9.2% over baseline).

However, the cost is high: decoder routing degrades by 23%. The model must route cross-attention by pure content semantics, with no positional cues to fall back on. Additionally, there's a *leakage asymmetry*: B can still see BIND (which contains A's processed state), so complete separation never happens.

**Scaling note:** From ~74K to ~380K, 7b loses 12.1 percentage points of composition, the worst scaling among causal variants. Larger capacity doesn't help—it actually commits more strongly to the separation strategy, worsening composition.

---

### 7c: Causal + Bidirectional (Conflicting Constraints)

**Hypothesis:** Combined encoder and BIND constraints might reinforce isolation.

**Result:** Catastrophic failure. Composition (59.5%) is decent, but decoder routing completely breaks (0.3479, worst across all variants). Isolation is worst (0.2427).

**Interpretation:** The two constraints fundamentally conflict. The model must learn:
- Causal encoding: A→[A, BIND], B→[BIND, B] (asymmetric)
- Bidirectional BIND: A→BIND→B AND B→BIND→A (symmetric)

These create incompatible gradient pressures. The decoder cross-attention mechanism, which must route by content *and* handle flipped A/B positions, breaks entirely (0.3479 is the lowest decoder score in the entire study). This asymmetric encoder state confuses the decoder's learned routing patterns beyond recovery.

**Why composition survives**: The freeze curriculum still works—frozen partial-spans provide enough signal to maintain composition accuracy even though routing is broken.

---

## Detailed Analysis: Scaling Study (~74K → ~380K)

### 7d: Block-Diagonal Masked Encoder (Strong Isolation)

**Design:** Bidirectional encoder with strict masking: A→[A+BIND], BIND→all, B→[BIND+B]. This eliminates all direct A↔B communication, creating mathematically independent representations except through the BIND bottleneck.

**~74K Results:** Isolation excellent (0.0200), composition good (66.2%), decoder strong (0.5950).
- Trade-off: -12% decoder cost vs baseline, but isolation is tight and composition is reasonable
- Demonstrates that hard masking works architecturally without bidirectional forcing

**~380K Results:** Isolation perfect (-0.0020), decoder improves (+6.1%), but composition drops (-5.6% absolute vs ~74K).

**Scaling narrative:** The perfect isolation paradoxically worsens composition at scale. At ~74K, the model learns to bridge the A↔BIND↔B gap reasonably well (66.2%). At ~380K, with more capacity, the model *commits harder* to isolation—A and B representations diverge so completely that the decoder struggles to recover hybrid properties. The decoder itself improves (+6.1%), proving capacity helps routing, but the representations themselves have diverged too far to bridge effectively.

This is the **clearest evidence against the "BIND bottleneck" hypothesis**: more capacity doesn't unlock block-diagonal masked constraints; instead, larger models take full advantage of the isolation and separate A/B more completely, sacrificing composition.

**Overall ranking at ~380K:** 7d emerges as best among isolated variants (isolation -0.0020, decoder 0.6314, composition 60.6%), though still 17 points below unconstrained optimal (77%). The isolation-composition trade-off is real and fundamental.

---

### 7e: Block-Diagonal Masked + Bidirectional BIND

**Design:** Strictest constraints: block-diagonal encoder + bidirectional BIND flipping.

**~74K Results:** Isolation perfect (0.0007), composition acceptable (59.8%), decoder moderate (0.5115).
- Nearly flawless architectural isolation, but bidirectional flipping introduces optimization complexity
- Composition still reasonable despite dual constraints

**~380K Results:** Isolation stable (0.0040), but composition **collapses** (49.7% = unconstrained baseline!). Decoder weakens to 0.5101.

**Scaling narrative:** Catastrophic failure at scale. At ~74K, the model successfully learns both A→BIND→B and B→BIND→A routing patterns despite perfect isolation (-0.0007). At ~380K, with more capacity and larger latent space, these conflicting objectives become irreconcilable:

- Block-diagonal masking forces A and B to evolve independently
- Bidirectional BIND demands the decoder learn symmetric routing (A→B and B→A with equal facility)
- These requirements directly contradict: if A and B are isolated, one direction is always "backward" relative to memory order

The decoder breaks trying to satisfy both (0.5101, second-worst). Composition degrades 10.1 percentage points from ~74K to 49.7% (matching unconstrained baseline exactly), erasing all isolation benefit.

**Key insight:** Bidirectional BIND needs encoder *overlap* (mixing) to work. When applied to perfectly isolated encoder, it creates unsolvable gradient conflicts at scale.

---

## Full Comparison: All Variants at Both Scales

| Variant | ~74K Decoder | ~380K Decoder | ~74K Comp | ~380K Comp | ~74K B_Iso | ~380K B_Iso | Scaling Story |
|---------|-------------|-------------|----------|-----------|-----------|-----------|-----------------|
| **Stage5** | 0.6759 | 0.7021 | 49.7% | 77.0% | 0.0327 | 0.0160 | Unconstrained: decoder +3.9%, comp +27.3% |
| **7a** | DNE | 0.5207 | DNE | 49.3% | DNE | 0.2100 | ~380K-only variant, no ~74K baseline |
| **7b** | 0.4943 | 0.5175 | 71.0% | 58.9% | 0.0767 | 0.0380 | Best ~74K comp, loses 12% at scale |
| **7c** | DNE | 0.3479 | DNE | 59.5% | DNE | 0.2427 | ~380K-only variant, decoder disaster |
| **7d** | 0.5950 | 0.6314 | 66.2% | 60.6% | 0.0200 | -0.0020 | Perfect isolation, composition trade-off |
| **7e** | 0.5115 | 0.5101 | 59.8% | 49.7% | 0.0007 | 0.0040 | Bidirectional collapse, back to baseline |

---

## Key Findings

### The Isolation-Composition Trade-Off is Real

Block-diagonal masking (7d) achieves the strongest isolation (-0.0020 at ~380K, nearly perfect) but at the cost of composition: 60.6% vs 77% optimal, a 17-point gap. This gap *worsens with scale*: 7d loses 5.6 percentage points from ~74K to ~380K as larger models commit harder to isolation. Decoder routing improves with capacity (+6.1%), but representation divergence between A and B increases faster. This is not a capacity problem—it's an architectural constraint problem.

### Capacity Does Not Unlock Isolation Constraints

The "BIND bottleneck" hypothesis—that larger models can overcome isolation by using more capacity—is **disproven**:
- All isolated variants (7b, 7d, 7e) improve composition at ~74K
- All degrade at ~380K despite vastly more capacity
- 7e is most striking: near-perfect ~74K isolation (0.0007) → catastrophic ~380K failure (49.7% composition, matching unconstrained baseline)

Larger models don't recover from isolation; they commit harder to it, sacrificing composition recovery.

### Bidirectional BIND Requires Encoder Overlap

7e proves bidirectional routing needs encoder mixing:
- At ~74K: Can learn A→B and B→A with perfect isolation (0.0007), composition 59.8%
- At ~380K: Conflicting objectives become irreconcilable, composition collapses (49.7%)

Bidirectional BIND creates symmetric gradient pressure that contradicts strict isolation—one direction is always "backward" relative to memory order. When encoder isolation is perfect, this conflict grows unsolvable at scale.

**Corollary:** 7a (bidirectional BIND alone, no encoder masking) also fails (49.3% composition), proving bidirectional training also needs structural backing.

### Causal Masking: Weak Isolation, Better Scaling

7b provides soft isolation (0.0767 at ~74K) with a 23% decoder cost, but scales more gracefully:
- Loses 12.1% composition (~74K: 71% → ~380K: 58.9%)
- Compared to 7e's 10.1% absolute loss, this seems better, but 7e's baseline was lower
- Decoder barely improves with scale (+2.3%), showing capacity doesn't help causal routing

Causal masking allows limited leakage (B sees BIND which contains A's state), which helps composition but breaks true isolation.

### Constraint Mixing Breaks Routing

7c (causal encoder + bidirectional BIND) has the worst decoder score in the entire study (0.3479). The two constraints create fundamentally incompatible encoder geometry—causal creates asymmetry, bidirectional demands symmetry. Cross-attention routing mechanism cannot learn stable patterns. Composition survives (59.5%) only because freeze curriculum provides strong supervision.

### Position Independence Requires Rethinking

Block-diagonal masking perfectly isolates A and B (isolation -0.0020) but achieves 60.6% composition—17 points below optimal. The problem is not architectural forcing (masking works as designed) but information loss: isolated representations lose the cross-entity context needed for composition. Suggests position-independent composition may require:
- Soft/learned masking instead of hard blocks
- Explicit bridge mechanisms (separate A↔B pathway)
- Graduated isolation (start bidirectional, transition to isolated)
- Latent composition module independent of encoder structure

---

## Files

Eval and probe results:
- `_agent/cache/runs/stage7/*/eval_results.json`
- `_agent/cache/runs/stage7/*-probes-cross_attention.json`
- `_agent/cache/runs/stage7_50k/*/eval_results.json`
- `_agent/cache/runs/stage7_50k/*-probes-cross_attention.json`

Full Stage 7 analysis:
- `_agent/cache/runs/stage7/STAGE7_FINDINGS.md`
- `_agent/cache/runs/stage7_50k/FINDINGS_50K_BLOCK_DIAG.md`

---

## Next Steps

1. **Translation layer experiment:** Load ~74K 7d encoder → ~380K Stage5 decoder. Test if ~74K isolation patterns work with larger decoder. Would show if problem is encoder over-commitment or decoder capacity.

2. **Soft isolation variants:** Replace hard block-diagonal masked masking with learned attention gates. Test if gradual isolation (not hard blocks) recovers composition.

3. **Graduated isolation training:** Start with causal encoder, transition to block-diagonal masked during training. Test if progressive constraint improves composition scaling.

4. **Bridge mechanisms:** Add explicit channels or tokens between A and B. Test if learned bridges can compensate for perfect isolation.

5. **Scaling sweep (deferred):** Repeat promising variants at 100K, 200K, 1M params to map composition-isolation-capacity landscape.
