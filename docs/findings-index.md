# Findings Index

A skimmable map of the research arc. Each entry summarizes what was tested, what was found, and where to read more.

The short version: we set out to prove O(1) KV cache composition via semantic isolation. We spent most of the project trying to teach models to append new information to a frozen cache. At the end, we found that the O(1) swap property was already present in the very first model we trained — it's intrinsic to composition training. The isolation work produced real architectural insights but was solving a different problem than we thought.

---

## Stage 1 — Domain & Baseline Composition
**[whiteroom-findings-01-stage1.md](whiteroom-findings-01-stage1.md)**

Establishes the Whiteroom synthetic domain: typed-port entities, deterministic composition rules, 379K-parameter encoder-decoder transformer. Trained 50K steps from scratch. Achieves strong basic composition and attribution. Confirms the domain is learnable and the evaluation machinery works.

→ See also: [whiteroom-spec.md](whiteroom-spec.md) for domain definition and [whiteroom-context.md](whiteroom-context.md) for research problem context.

---

## Stage 2 — Attribution + Training Distribution
**[whiteroom-findings-02-stage2.md](whiteroom-findings-02-stage2.md)** · **[whiteroom-findings-03-training-distribution.md](whiteroom-findings-03-training-distribution.md)**

Adds joint attribution training (which compound tokens came from which entity). Identifies and corrects two training distribution biases (unbalanced flag rates, flag co-occurrence). Multi-seed run (5 seeds) establishes reliable baselines.

→ Results: [results/multiseed/](../results/multiseed/), [results/multiseed-unbalanced/](../results/multiseed-unbalanced/)

---

## Stage 3 (Findings 04–07) — Generalization, Freeze Tests, Sibling
**[whiteroom-findings-04-generalization.md](whiteroom-findings-04-generalization.md)** · **[whiteroom-findings-05-sibling.md](whiteroom-findings-05-sibling.md)** · **[whiteroom-findings-06-sibling-xcheckpoint.md](whiteroom-findings-06-sibling-xcheckpoint.md)** · **[whiteroom-findings-07-multiseed.md](whiteroom-findings-07-multiseed.md)**

Token and combination holdout tests confirm the model learns abstract compositional rules, not token-specific mappings. The property-append freeze test is introduced: can the decoder produce the correct compound when given a frozen base cache plus one new appended token? A "sibling" span-predictor model is trained to select which cache positions to freeze. Multi-seed analysis exposes a key limitation: post-hoc curriculum fine-tuning has a ceiling because the encoder already converged before curriculum training began.

→ Results: [results/multiseed/](../results/multiseed/), [results/multiseed-unbalanced/](../results/multiseed-unbalanced/), [results/siblings-multiseed-compact/](../results/siblings-multiseed-compact/)

---

## Stage 4 — Curriculum Fine-Tuning (Post-Hoc)
**[whiteroom-findings-08-stage4.md](whiteroom-findings-08-stage4.md)** · **[whiteroom-findings-08b-stage4bc.md](whiteroom-findings-08b-stage4bc.md)**

Mixed curriculum fine-tunes the Stage 2 checkpoints to improve property-append pickup. Achieves +33pp improvement on most seeds. Two-phase curriculum (Stage 4b) pushes further. Laggard seeds (2 and 3) remain stubborn — confirmed by Stage 4c extended training. Conclusion: post-hoc curriculum cannot fix encoder representations that entangled during pretraining.

**Checkpoint released**: stage4b-seed4 — this seed achieved anomalously high composition performance that motivated most of the subsequent research. Included in the main repository. Exact reproduction is not expected. See [model-specs.md](model-specs.md).

→ Results: [results/stage4/](../results/stage4/), [results/stage4b/](../results/stage4b/), [results/stage4c/](../results/stage4c/)

---

## Stage 5 — From-Scratch Adaptive Curriculum
**[whiteroom-findings-09-stage5.md](whiteroom-findings-09-stage5.md)**

Trains all 5 seeds from scratch with the freeze curriculum active from step 1. The encoder and decoder co-evolve under isolation pressure, producing naturally disentangled representations. Achieves ~77% composition at 380K params, ~49.7% at 74K, near-perfect attribution. Best free-trained baseline in the project.

→ Results: [results/stage5/](../results/stage5/), [results/stage5b/](../results/stage5b/), [results/stage5c/](../results/stage5c/)

---

## Stage 6 — Semantic Position Independence (Analysis Pass)
**[whiteroom-findings-10-stage6-analysis.md](whiteroom-findings-10-stage6-analysis.md)** · **[whiteroom-findings-11-stage6-probes.md](whiteroom-findings-11-stage6-probes.md)**

No new training. Probes existing checkpoints (Stage 2, Stage 5) to characterize how the decoder reads frozen encoder memory. Memory splice swap test: swap A and B's encoder representations in the frozen cache — does decoding break? It doesn't. The decoder routes by content semantics, not position. Duplicate-A, shuffle, noise, and cross-attention probes all confirm the same picture.

→ Results: [results/probe_cross_attention_stage2.json](../results/probe_cross_attention_stage2.json), [results/probe_cross_attention_stage5.json](../results/probe_cross_attention_stage5.json), [results/probe_memory_ops_results.json](../results/probe_memory_ops_results.json)

---

## Stage 7 — Encoder Isolation Mechanisms (Architectural Variants)
**[whiteroom-findings-12-stage7.md](whiteroom-findings-12-stage7.md)**

2×2 factorial study over encoder attention pattern (bidirectional / causal / block-diagonal) × BIND directionality (uni / bidir). Trained at both 74K and 380K scales. **7d (block-diagonal / sawtooth encoder) is the winner**: near-perfect isolation at 363K steps, best composition retention (60.6%) among isolation variants. Key finding: isolation and composition trade off at scale — more capacity commits harder to isolation, not composition recovery.

→ Results: [results/stage7/](../results/stage7/), [results/stage7_50k/](../results/stage7_50k/)

---

## Stage 8 — Frozen Enc + Frozen Dec + Learned Projection
**[whiteroom-findings-13-stage8.md](whiteroom-findings-13-stage8.md)**

Freeze 7d (best isolation) and Stage5 (best composition). Train only a small projection layer between them. **Linear projection: 62.3% composition, 95.1% attribution, perfect isolation. MLP: 64.5% / 95.0%.** Best isolation+composition numbers in the project. Also surfaces a numerical stability issue in frozen dropout during eval — fixed by `fix_stage9_checkpoints.py`.

→ Results: [results/stage8/](../results/stage8/) · Script: [scripts/fix_stage9_checkpoints.py](../scripts/fix_stage9_checkpoints.py)

---

## Stage 9 — 3-Stage Architecture (Free Enc + Adaptation + Free Dec)
**[whiteroom-findings-14-stage9.md](whiteroom-findings-14-stage9.md)**

Replaces the frozen+projection approach with an end-to-end 3-stage architecture (encoder → adaptation layer → decoder), all trained jointly with block-diagonal masking. Catastrophic failure: 23.5% composition, attribution collapses to 6.5%, isolation perfects. Root cause: the adaptation layer is captured by whichever component learns fastest, preventing co-adaptation between encoder and decoder.

→ Results: [results/stage9/](../results/stage9/), [results/stage9b/](../results/stage9b/), [results/stage9c/](../results/stage9c/)

---

## Stage 10 — Asymmetric WhiteroomTransformer
**[whiteroom-findings-15-stage10.md](whiteroom-findings-15-stage10.md)**

Drops the adaptation layer. Uses the base encoder-decoder transformer with block-diagonal masking and asymmetric layer counts (shallow enc, deep dec). Key findings: **decoder depth is the primary composition lever; encoder depth ≥2 hurts isolation; enc:dec ratio ~1:10 is the sweet spot.** Best models: **10e "The Club"** (d=32, 2+21, ~55.3% composition) and **10g** (d=64, 1+11). 10i (same as 10h but no mask) is the ablation confirming the mask matters. `valid_weight=0.25` reduces variance and improves composition ~5pp.

→ Results: [results/stage10/](../results/stage10/)

---

## Stage 11 — Cross-Pair Memory Swap (O(1) Confirmation)
**[whiteroom-findings-16-stage11-swap.md](whiteroom-findings-16-stage11-swap.md)**

The central question: can pre-computed encoder representations be spliced across entity pairs without re-encoding? Sample (A, B, C, D) where A+B and C+D are both valid, then decode A+D using C's cached B-slot representations (and vice versa). **Result: all tested models maintain 95–100% of fresh-encode accuracy.** Stage 2 (no curriculum, no masking) achieves 100.6% — the capability is intrinsic to composition training and was present from the start. Stage 10 block-diagonal models achieve left_cos_sim=1.000 (structural invariance). **O(1) KV cache composition is empirically confirmed.**

→ Results: [results/stage5/eval_swap_results.json](../results/stage5/eval_swap_results.json), [results/stage10/10e-d32-2enc-21dec/eval_swap_results.json](../results/stage10/10e-d32-2enc-21dec/eval_swap_results.json), [results/multiseed/eval_swap_results.json](../results/multiseed/eval_swap_results.json)

---

## Bonus: Stage 12 — Two-BIND Zero-Shot
*(Not a separate findings file — covered in the full summary)*

Zero-shot test of two-BIND composition (A+B+C) on models trained only on single-BIND. Flag accuracy 96–99% despite never seeing the format. Fails only on port counting (one hallucinated port). Suggests the compositional representation is structurally extensible.

→ Results: [results/stage10/10e-d32-2enc-21dec/eval_twobin_results.json](../results/stage10/10e-d32-2enc-21dec/eval_twobin_results.json), [results/multiseed/eval_twobin_results.json](../results/multiseed/eval_twobin_results.json)

---

## Further Reading

- [whiteroom-full-summary.md](whiteroom-full-summary.md) — narrative summary of the full arc
- [whiteroom-spec.md](whiteroom-spec.md) — domain specification and entity model
- [whiteroom-context.md](whiteroom-context.md) — project context and motivation
- [whiteroom-orientation-00-litreview.md](whiteroom-orientation-00-litreview.md) — literature orientation
- [whiteroom-future-enc-enc-dec-dec.md](whiteroom-future-enc-enc-dec-dec.md) — future directions
- [frontier-model-predictions.md](frontier-model-predictions.md) — predictions logged before key experiments
- [model-specs.md](model-specs.md) — checkpoint architecture reference
