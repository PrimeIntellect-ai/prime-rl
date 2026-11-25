# Goal-conditioned supervised learning on achieved goals

This optional example learns from a failed request when its actual actions reached another valid goal. Run one CPU update from the repository root.

```sh
uv run python examples/extra/hindsight/train.py
```

An environment supplies a `FixedHorizonWorld` for each completed v1 episode. Its deterministic transitions, horizon, and allowed actions must stay the same when the requested goal changes. Provide the set of goals that may be requested for this starting state, the fixed legal token alphabet, and a renderer that reconstructs every action prefix from the goal and earlier actions. The original verifier reward must equal the indicator that the terminal state matches the requested goal.

Pass episodes and worlds to `make_hindsight_batch`. Each supported achieved goal yields its rendered prefixes and the actions originally selected. An unsupported goal leaves an empty slot. Original rewards and model-call evidence remain on the factual episode. Prime-RL's stock wire format omits sampler support, so the learner trusts the environment's declared legal actions when only the wire record is available. On a live record with captured support, reconstruction checks it against that declaration at every sampled action. Imitation requires no behavior log probabilities or model-version snapshot.

Teacher-force the retained prefixes through the student and gather its logits over the declared legal actions at the training temperature. Call `hindsight_loss(legal_logits, action_columns, len(episodes))`. This sums the action losses over each eligible episode and divides by the original episode count. Dividing again by the fixed horizon gives the per-decision reduction. The optional metrics dictionary counts original episodes, unsupported outcomes, successes with a single supported goal and eligible misses. The example uses an explicit caller. Distributed launcher integration and timing depend on the environment and serving stack.

For weighted or centered updates, see the [experimental hindsight estimators](posterior.md).
