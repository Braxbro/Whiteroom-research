# Whiteroom — Future Direction: Encoder-Encoder-Decoder-Decoder Architecture

## Motivation

The current single-encoder architecture relies on emergent representation
isolation — the model must learn, as a side effect of composition training,
to keep component spans geometrically separable in a shared encoder output.
This works for most seeds but is not guaranteed: seeds 2 and 3 show entangled
representations that limit curriculum training effectiveness. At harder tasks
or larger scales, relying on emergence becomes increasingly fragile.

The enc-enc-dec-dec architecture makes cache isolation explicit and architectural
rather than a training outcome.

---

## Architecture

Four components in series:

    Encoder 1 → Encoder 2 → Decoder 2 → Decoder 1

Each stage produces an independently cacheable intermediate representation:

- **V1**: Encoder 1 output — raw token-level representation
- **V2**: Encoder 2 output — abstract compound-level representation (attends over V1)
- **D2**: Decoder 2 output — intermediate decoded representation (attends over V2)
- **Final output**: Decoder 1 output (attends over D2)

All of V1, V2, and D2 are freezable independently. Three cacheable intermediate
layers instead of one.

---

## Cache Policy Space

The sibling's job expands from span-mask-over-tokens to a structured policy
over 3 layers × component granularity:

| Change type | Freeze V1 | Freeze V2 | Freeze D2 |
|-------------|-----------|-----------|-----------|
| Flag append only | ✓ | ✓ | ✗ (re-run) |
| Binding change | ✓ | ✗ | ✗ |
| Component A replacement | ✗ (A spans) | ✗ | ✗ |
| Full recompute | ✗ | ✗ | ✗ |

Each layer can also be frozen at component granularity — freeze A's contribution
to V2 while re-running B's, for example. The policy space is strictly richer
than span masking over a flat token sequence, but the decisions are more
semantically meaningful and therefore potentially easier to learn.

---

## Sibling in This Architecture

The sibling gets more expressive output (structured 3-layer policy) but inference
cost remains a function of input sequence length and parameter count — the output
head is trivially small. The sibling may also operate on intermediate
representations (diff between V1_old/V1_new etc.) rather than raw tokens,
giving a smaller and more informative input than the current `[old | SEP | new]`
format.

Net effect: richer policy, same or cheaper inference cost, potentially better-
conditioned input. The sibling scales better in this architecture than in the
current one.

---

## Why Not Now

This is conjecture until tested. Open questions before it's worth building:

1. **Does the current architecture have more runway?** Stage 4b partial freeze
   results may push seed 2/3 pickup significantly. If all seeds converge near
   0.9+ pickup, the emergent isolation approach is sufficient at this scale and
   the architectural change is only motivated by harder tasks.

2. **Does series enc-enc actually improve representation separability?** The
   hypothesis is that encoder 2 can build compound representations from already-
   formed component representations in V1, making separation more explicit. But
   without ablations this is unverified.

3. **Does the richer sibling policy actually need the richer input?** The current
   sibling already matches oracle with a simple transformer over token sequences.
   Whether the 3-layer policy is learnable and whether it adds value over the
   current span mask is an empirical question.

The natural trigger for pursuing this: current architecture hits a ceiling on
pickup that curriculum training can't break through, particularly for seeds with
entangled base representations. That would indicate the bottleneck is structural
rather than trainable, and an architectural fix is warranted.

---

*Generated: 2026-03-31. Model: claude-sonnet-4-6.*
