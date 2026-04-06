# Whiteroom Model Specs

All models share: `vocab_size=67`, `dropout=0.1`, `dim_feedforward=4×d_model`.

Each stage ran 5 seeds unless otherwise noted. Checkpoints available as release assets except where noted.

---

## Baseline / Early Stages

Standard symmetric encoder-decoder architecture. `d_model=64`, `nhead=4`, `enc=3`, `dec=3`, `ff=256`.

| Run | Steps | Notes |
|---|---|---|
| stage2/multiseed | 50K | Balanced training distribution |
| stage2/multiseed-unbalanced | 50K | Unbalanced distribution — confirms finding is not distribution-dependent |
| stage2-holdout-tokens | 50K | Token holdout generalization probe |
| stage3-combination-holdout | 50K | Combination holdout generalization probe |
| stage4 | 20K | Frozen-encoder curriculum |
| stage4b-seed4 | 10K | **See below** |
| stage4c | 10K | Frozen-encoder curriculum variant |
| stage5 | ~25–30K | Adaptive freeze curriculum — best free-trained baseline |
| stage5b | 20K | Curriculum variant |
| stage5c | 20K | Curriculum variant |

### stage4b-seed4

Same architecture as all other stage4 runs (`d=64`, `3+3`), stopped at step 10K. This seed produced anomalously high composition performance that motivated the remainder of the research. Included in the main repository rather than as a release asset. Exact reproduction is not expected — it depended on hardware, RNG state, and training noise that were not logged.

---

## Stage 7 — Encoder Isolation Variants

Base architecture: `d_model=64`, `nhead=4`, `enc=3`, `dec=3`, `ff=256`. Trained to ~363K parameters.

| Run | Encoder mask | BIND direction | Notes |
|---|---|---|---|
| 7a | None (bidirectional) | Bidirectional | Bidir-BIND baseline |
| 7b | Causal | Unidirectional | A cannot attend to B |
| 7c | Causal | Bidirectional | |
| 7d | Block-diagonal (sawtooth) | Unidirectional | Best isolation variant |
| 7e | Block-diagonal (sawtooth) | Bidirectional | |

Also trained at 50K-parameter scale (stage7_50k) for scaling comparison.

---

## Stage 8 — Frozen Encoder + Frozen Decoder + Learned Projection

Encoder and decoder weights are frozen; only the projection layer is trained.

- **Encoder**: frozen stage7/7d (`d=64`, `3+3`, block-diagonal mask)
- **Decoder**: frozen stage5 (`d=64`, `3+3`)

| Run | Projection architecture |
|---|---|
| 8d (linear) | LayerNorm(64) → Linear(64→64) |
| 8e (mlp) | LayerNorm(64) → Linear(64→256) → ReLU → Linear(256→64) |

Note: stage8 checkpoints use a different format (`checkpoint_translation.pt`); config is stored under `encoder_config`, `decoder_config`, and `projection_config` keys.

---

## Stage 9 — Free 3-Stage

Three-stage training with adaptation layer. `d_model=64`, `nhead=4`, `enc=3`, `dec=3`, `ff=256`.

| Run | Steps | Notes |
|---|---|---|
| stage9 | ~25K | Baseline 3-stage |
| stage9b | ~60K | Variant |
| stage9c | ~48K | Variant |

---

## Stage 10 — Asymmetric WhiteroomTransformer

Asymmetric enc/dec depth with block-diagonal encoder mask (except 10i ablation). All runs use 5 seeds.

| Run | d_model | nhead | Enc layers | Dec layers | ff | Mask | Notes |
|---|---|---|---|---|---|---|---|
| 10a | 64 | 4 | 1 | 5 | 256 | ✓ | |
| 10b | 64 | 4 | 1 | 5 | 256 | ✓ | valid_weight=0.25 |
| 10c | 64 | 4 | 2 | 4 | 256 | ✓ | |
| 10d | 32 | 2 | 2 | 10 | 128 | ✓ | |
| 10e "The Club" | 32 | 2 | 2 | 21 | 128 | ✓ | Best d=32 result |
| 10f | 64 | 4 | 2 | 6 | 256 | ✓ | |
| 10g | 64 | 4 | 1 | 11 | 256 | ✓ | Best d=64 result |
| 10h | 64 | 4 | 3 | 9 | 256 | ✓ | |
| 10i | 64 | 4 | 3 | 9 | 256 | ✗ | Ablation — no mask; paired with 10h |
