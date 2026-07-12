# Issue: Echo assumes one sample per trainable branch

## Status

Deferred from the TTT review. Existing Echo behavior should remain unchanged until this is
specified independently.

## Summary

`EchoAlgorithm` pairs `rollout.samples` with trainable branches. That assumption can fail if
trajectory conversion emits more than one sample for a branch or removes/duplicates branch
entries. A TTT version split is one example, but future graph transformations could create the
same mismatch without TTT.

Observation-token CE weighting also needs a rule when one branch is represented by several
samples: copying every observation span into every sample would multiply its loss.

## Change considered on the TTT branch

The branch introduced a shared iterator producing exact `(branch, mask, version)` entries,
asserted strict sample alignment, and assigned each observation span to the adapter version of
the following sampled response (or the preceding response for a trailing span).

## Why it is deferred

That logic runs for every Echo rollout and changes how shared or unusual branch structures are
weighted. The TTT ScaleSWE experiment does not use Echo. The minimal TTT branch can reject or
document Echo+TTT instead of redesigning Echo's objective.

## Design questions

- Should Echo operate on graph nodes before flattening rather than reconstructing alignment?
- Which sample owns observation text between responses produced by different policies/adapters?
- Should strict mismatch raise, skip Echo CE, or fall back to branch order?
- How do shared sampled prefixes interact with Echo advantages?

## Suggested tests

- Existing linear and multi-turn Echo cases remain unchanged.
- One branch producing multiple samples has an explicit observation ownership rule.
- Forked branches and shared prefixes do not multiply CE unexpectedly.
- A mismatch fails with enough identifiers to diagnose the responsible conversion.

## Relevant code

- `src/prime_rl/orchestrator/algo/echo.py`
- `src/prime_rl/orchestrator/trajectories.py`
- `tests/unit/orchestrator/test_algorithms.py`
