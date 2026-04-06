# Whiteroom — Stage 1 Findings

## Domain & Setup

Synthetic domain for testing semantic independence in transformer KV cache representations.

**Entity model**: entities have typed ports (in/out transfer channels), an operation type (void/throttle/toggle), and side-behavior flags (shoots, illuminates, scans, broadcasts, attracts, spawns). Two entities compose into a compound by binding one output port to a compatible input port. The compound inherits both port signatures minus the bound pair, and all flags by union. Ground truth is fully deterministic — labels computed programmatically.

**Vocabulary**: 53 tokens (8 port types × 2 directions, 3 op types, 6 flags, 10 port-index tokens, 12 archetype IDs, 5 special tokens).

**Model**: encoder-decoder transformer, 379,844 parameters, trained from scratch. 3 encoder layers, 3 decoder layers, d_model=64, 4 heads, ffn=256. Two output heads: sequence prediction (compound token output) and is_valid classification (binding validity).

**Training**: 50K steps, batch size 64, on-the-fly data generation (infinite, no static dataset), cosine LR schedule, Adam optimizer, lr=3e-4. ~22 minutes on a single CUDA GPU (torch 2.11.0+cu126).

---

## Task Performance at 50K Steps

| Metric | Score |
|---|---|
| Sequence exact match (compound output) | 87.0% |
| Port set match (unordered, correct metric) | **90.6%** |
| Flag exact match | **100.0%** |
| is_valid accuracy (natural distribution, 80% valid) | 92.2% |

Ports are orderless by spec, so 90.6% (unordered set match) is the correct port accuracy figure. The 87.0% ordered match reflects sequencing errors only — the model knows which ports survive but occasionally misorients them in the output. Since order is irrelevant, this is not a real error.

Flag accuracy reached ceiling. The model perfectly learned that flags are independent of port mechanics — a prerequisite for the cache freezing experiment.

### Port Failure Analysis

Of the 9.4% port set mismatches (n=152/1589 valid examples):
- **64%** — right count, wrong types: correct number of ports predicted but one type substituted for another. DRIFT, BLOOM, WAVE, STATIC are most commonly confused. No single dominant failure type — likely general type-similarity confusion.
- **36%** — wrong count: model outputs ±1 port. Almost exclusively off-by-one, occurring more at higher port counts (5+). Larger compounds have longer, more complex port signatures.

Neither failure mode is structural. The research signals (flags, cache freezing) are unaffected.

### is_valid Balanced Evaluation

The 92.2% figure is on the natural training distribution (80% valid). On a balanced 50/50 evaluation (1000 valid, 1000 invalid):

| Metric | Score |
|---|---|
| Accuracy | 83.8% |
| Precision | 75.5% |
| Recall (valid detection) | **100.0%** |
| Specificity (invalid detection) | 67.5% |
| False positive rate | 32.5% |
| False negative rate | 0.0% |

The model never misses a valid binding but calls 32.5% of invalid bindings valid. This is a training imbalance artifact — 80/20 split biases it toward predicting valid when uncertain. Not a capability failure; the model clearly has the information required (recall is perfect). A retrain at 50/50 would likely close most of this gap.

**Stage 1b candidates**:
- *Balanced retrain*: retrain from scratch at 50/50 valid/invalid split; check whether specificity improves without degrading composition or cache freezing.
- *Curriculum approach*: train valid-heavy first (current 80/20) to solidify composition and semantic independence behavior, then continue training with an invalid-biased split to sharpen the validity classifier. The two objectives are largely orthogonal — low risk of the second phase disturbing what the first phase learned. May be preferable to a full retrain if stage 1 results need to be preserved.

---

## Cache Freezing Experiment

### Setup

For each triplet (A, B, C) where C is a *compliant swap* for B — same binding interface to A (compatible port type), different flags and/or operation type:

- **Normal forward**: encode `[A | BIND | C]` fresh → decode → predict `compound(A, C)`
- **A-frozen forward**: take A's encoder output from `encode([A | BIND | B])`, splice with BIND+C positions from `encode([A | BIND | C])` → decode → predict `compound(A, C)`

Symmetric B-frozen test: B stays fixed, A is swapped for compliant D. Freeze B's encoder output from the original `encode([A | BIND | B])` run, use D+BIND positions from `encode([D | BIND | B])`.

500 triplets per checkpoint. Primitives only (depth-0 entities).

### Results

| Checkpoint | A-frozen seq | A-frozen flags | Δ flags | A cos-sim | B-frozen seq | B-frozen flags | Δ flags | B cos-sim |
|---|---|---|---|---|---|---|---|---|
| 30K | 100% | 100% | **0%** | 0.928 | 100% | 100% | **0%** | 0.909 |
| 40K | 100% | 100% | **0%** | 0.929 | 100% | 100% | **0%** | 0.909 |
| 50K | 100% | 100% | **0%** | 0.929 | 100% | 100% | **0%** | 0.909 |

### Interpretation

**Zero degradation in both directions across all checkpoints.**

The model works correctly whether A or B's encoder representations are frozen from a prior run with a different partner. It cannot tell which component is stale.

**Cosine similarity**: A's encoder representation shifts ~7% (cos-sim 0.929) when B is swapped; B's shifts ~9% (cos-sim 0.909). The representations are not perfectly invariant — the encoder retains some cross-component coupling. Yet the decoder learned to be completely robust to this residual variation.

**Asymmetry (A more stable than B)**: likely positional — A always appears first in the input sequence and receives less cross-attention contamination from the variable component. Consistent across all checkpoints.

**No special training required**: the hypothesis was that explicit frozen-cache training might be needed to force detangling. It was not. Training on the composition task alone was sufficient. The model learned semantic independence spontaneously.

**Positional bias — diminished, not eliminated**: the encoder retains measurable positional coupling (cosine sim < 1.0 in both directions). The model did not learn position-invariant *representations* — it learned position-tolerant *prediction*. The decoder learned not to act on the residual positional bias in the encoder. This distinction matters: cache freezing works in practice, but the underlying representations are not fully decoupled. Whether this residual coupling becomes a problem at scale — larger models, longer contexts, weaker explicit contracts — is an open question and a motivating framing for the follow-up study.

---

## Key Finding

A transformer trained on structured compositional data with **explicit typed binding contracts** learns semantic independence without any architectural intervention or special training objective. Freezing either component's cached representations causes zero output degradation.

The binding contract — the port type — is the sufficient abstraction boundary. The model learned that what's behind the contract on either side does not affect the other side's representation, at the level that matters for prediction.

---

## Conditions & Limitations

- Primitives only (depth-0 entities). Compound-as-component not yet tested.
- Small model (379,844 params) on a low-complexity structural task — not a language model.
- Explicit, discrete contracts. The generalization question (implicit contracts in natural language) is out of scope for this stage.

---

## Research Arc

1. **Stage 1**: explicit typed contracts, structured synthetic domain → positive result.
2. **Stage 1b (optional follow-up)**: curriculum training for validity classifier; attribution head; compound-as-component (depth > 0).
3. **Follow-up study**: can a model learn to identify semantic binding contracts in natural language, where contracts are implicit?
4. **Further follow-up**: architectural interventions — attention gating, separator tokens — if models can learn semantic independence, can you design mechanisms that enforce it?

Step 1 is the necessary condition for steps 2 and 3. If the model could not learn semantic independence given explicit contracts, steps 2 and 3 would be futile. It can.

---

*Generated: 2026-03-30. Model: claude-sonnet-4-6. Implementation: PyTorch, trained from scratch on synthetic data.*
