# Issue: all-rejected cohorts can leave token batching without progress

## Status

Deferred from the TTT review. No general behavior change is included here.

## Summary

A completed group may produce no trainer samples because every rollout errored or filters
removed all survivors. In token-batched mode, zero samples contribute zero tokens, so the sink
may never reach its threshold. Repeated dead cohorts can accumulate without a trainer step or a
clear terminal signal.

## Observable consequences

- A run appears alive but stops producing training batches.
- Rejected rollouts may never be persisted in the normal batch output.
- Per-rollout resources owned outside the sink cannot learn that the cohort is conclusive.
- Operators see no bounded failure when every new group is unusable.

The TTT review exposed the resource aspect because rejected rollouts can own adapter
checkpoints, but the liveness question exists independently of TTT.

## Change considered on the TTT branch

The branch emitted lifecycle-only batches with an empty `samples` list, persisted rejected
rollouts to an append-only diagnostic stream, counted consecutive empty batches, and aborted
after a bound unless viable survivors remained buffered.

## Why this belongs outside TTT

Empty-batch semantics affect the orchestrator state machine for every environment. An abort
threshold that is suitable for one workload may be wrong for another, and emitting batches
with no trainer payload changes assumptions held by callers. TTT needs a way to release its own
resources, but it should not silently define the general liveness policy.

## Design questions

- Should a dead cohort be an event, a persisted record, or a `TrainBatch` with no samples?
- Is the failure threshold global, per environment, or time-based?
- What counts as progress when healthy groups are partially buffered?
- Should errored and policy-filtered cohorts have different thresholds?
- How should checkpoint/resume represent pending rejected observations?

## Suggested tests

- Repeated all-error groups terminate or report according to the selected policy.
- Alternating dead and productive groups do not trigger a false abort.
- Partial healthy groups count as progress only under a documented rule.
- Count- and token-batched modes behave consistently.
- Rejected observations are persisted exactly once.

## Relevant code

- `src/prime_rl/orchestrator/train_sink.py`
- `src/prime_rl/orchestrator/orchestrator.py`
- `src/prime_rl/orchestrator/types.py`
- `src/prime_rl/orchestrator/utils.py`
