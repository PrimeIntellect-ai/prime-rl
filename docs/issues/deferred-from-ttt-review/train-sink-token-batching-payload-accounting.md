# Issue: token batching counts traces rather than the shipped payload

## Status

Deferred from the TTT review. Reproduce on current `main` before implementation.

## Summary

Token-batched `TrainSink` accounting uses rollout/trace token totals. The trainer receives
`TrainingSample` objects, whose number and length can differ from the source trace after graph
conversion, sample expansion, filtering, or auxiliary loss routing.

## Why the counts can differ

- One graph may yield multiple training samples.
- Shared or untrainable branches may be omitted.
- A sample carries full context even when only a subset of tokens has loss weight.
- An algorithm may attach additional CE-routed samples.
- Future transformations may split or combine samples independently of the trace graph.

As a result, a configured token batch can be materially above or below its intended trainer
payload. This is primarily a batching and resource-predictability concern; it is not evidence
that ordinary count-batched training is incorrect.

## Change considered on the TTT branch

The branch recomputed pending tokens from `sum(len(sample.token_ids) for sample in
rollout.samples)` and included group-owned auxiliary samples exactly once. Batch cutting used
the same payload measure.

## Why it was removed or gated in TTT

Changing the unit for every token-batched job is a general Prime-RL semantic change. It can
alter batch boundaries, optimizer-step composition, memory use, and reproducibility. The TTT
ScaleSWE experiment uses rollout-count batching, so a universal change is not required for the
experiment.

If TTT retains exact accounting for configurations that add Q&A/meta samples, it should be
explicitly TTT-gated or TTT token batching should be rejected until a general policy is chosen.

## Design questions

- Does `token_batch_size` mean wire tokens, trainable loss tokens, or model-forward tokens?
- Should multimodal payload cost be represented separately?
- Should oversized first samples ship alone or be rejected?
- Is exact recomputation acceptable, or should each rollout cache its final payload cost?
- When filters run after a cut, should the sink backfill to the target?

## Suggested tests

- A linear rollout where trace and sample sizes match preserves existing boundaries.
- A forked graph producing multiple samples uses the documented unit.
- Auxiliary CE samples are counted exactly once.
- Post-filter removal has defined underfill behavior.
- Overflow remains stable across repeated calls and mixed environments.

## Relevant code

- `src/prime_rl/orchestrator/train_sink.py`
- `src/prime_rl/orchestrator/trajectories.py`
- `src/prime_rl/transport/types.py`
