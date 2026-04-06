# What is Whiteroom?

I (with assistance from Claude Sonnet and Opus, plus moral support from Claude Haiku) trained a number of standard encoder-decoder transformers on a synthetic composition task inspired by the compound entity implementation paradigm often seen in Factorio mod development. (We also trained encoder-bridge-decoder transformers, however they did not offer any benefit over encoder-decoder transformers) In doing so, we found that models trained on well-defined composition tasks can swap out precomputed encoder representations (KV cache segments) from different contexts with zero accuracy loss. Current research consensus, and four frontier models additionally used as consensus aggregators, predicted that this would fail.

It didn't.

As for the name Whiteroom itself, it came from the fact that I wanted to build a white room parallel to the compound entity paradigm as my toy problem due to personal familiarity. Claude (both Opus and Sonnet) adopted it as a name "Whiteroom" and I never bothered correcting them. The name stuck. Sorry if the name overlaps with anyone else. I'm bad with names.

**In short:** Compositional portability is an inherent property of encoder-decoder transformers trained on well-defined composition tasks, without any special architecture required. You can encode an input once, store the representation, and then substitute it in and out without recomputing it.

## Repo Structure

- `checkpoints/` - Model checkpoints I don't expect to be easily reproduceable. Currently only includes Stage 4b, seed 4 - the 'jackpot seed' that motivated many further stages of research chasing append-to-frozen capability.
- `docs/` - A number of writeups, including records of frontier model predictions, an early lit review from Opus, the initial research context and specifications, and 16 findings files. (Sorry. I got carried away. It was a fun week.) 
  - `model-specs.md` can also be found here, containing the hyperparameters of all the models trained.
  - For reading the findings, `findings-index.md` would be a good first stop.
- `results/` - Raw JSON eval results.
- `scripts/` - The scripts Sonnet made for model training, evaluation, and debugging.
- `whiteroom/` - The main Whiteroom Python package implementing the synthetic domain and common utilities between each stage.

## How to Reproduce

Requires PyTorch. Training scripts are in `scripts/stage{n}/`. Each stage (except for 10d-10i) trains in a couple of hours on my RTX 2070 SUPER, fed data generated on an Intel i9-10900K CPU. No dataset download is provided, since the training dataset was generated on-demand. (and thus does not exist *to* download)

Eval scripts are in `scripts` and `scripts/eval/` - see `docs/model-specs.md` for exact hyperparams per stage.

## AI Use Acknowledgement

The code used for this project, as well as the preliminary findings docs you will find here, were generated using Claude Sonnet 4.6 using Claude Code. 

Research methodology was designed alongside suggestions from Claude Opus 4.6 using Claude.ai. 

This research was spurred by a discussion with Claude Haiku using Claude.ai after I expressed confusion why prefix caching was standard, which led to a tangent into theoretical segmented/partitioned caching.

Which then led to this. (Well, it led to the start of this. Then this led to this.)
