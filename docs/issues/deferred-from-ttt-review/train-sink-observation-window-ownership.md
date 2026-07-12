# Issue: a ready batch can report an unrelated partial-group arrival

## Status

Deferred from the TTT review. The scenario should be reproduced against current `main` before
changing the batch contract.

## Summary

`TrainSink` maintains both trainer-ready survivors and a broader arrival window used for
metrics and persistence. When an older overflow cohort is already ready, the arrival that
triggers its emission can belong to a new, incomplete group. Returning the entire arrival
window can then associate that partial-group rollout with the older batch.

## Example

1. Group A finalizes with enough survivors for more than one trainer batch.
2. One A batch ships; A overflow remains ready.
3. The first rollout from group B arrives, but B is not complete and has no samples ready.
4. The sink emits A overflow and returns every arrival since the last shipment, including B.
5. B is persisted/counts in A's window, then may be missing when B's own samples eventually
   ship because the observation window was reset.

This is primarily an ownership/observability bug. Trainer samples can still contain the right A
cohort while metrics and persisted traces describe a different set.

## Change considered on the TTT branch

The branch tracked exact cohort IDs, conclusively rejected arrivals, and explicit carry from
earlier empty lifecycle batches. A normal shipment selected only observations owned by that
cohort and left unrelated partial groups for a later window.

## Why it is deferred

The meaning of `TrainBatch.rollouts` is shared Prime-RL API behavior. Tightening it can change
metrics denominators, persistence timing, filter telemetry, and any downstream code that relied
on the broader arrival-window interpretation. TTT checkpoint GC needs exact ownership, but that
can be isolated without redefining non-TTT batches.

## Suggested tests

- Ready overflow from A followed by one partial B arrival reports only A.
- B appears exactly once when B's own cohort ships.
- Rejected arrivals remain observable without being duplicated.
- Empty-batch carry does not sweep unrelated in-progress groups.
- Metrics and persisted rollout IDs match the documented ownership set.

## Relevant code

- `src/prime_rl/orchestrator/train_sink.py`
- `src/prime_rl/orchestrator/types.py`
- rollout persistence in `src/prime_rl/orchestrator/orchestrator.py`
