# Issue: forked traces can train one sampled node more than once

## Status

Resolved independently on Prime-RL `main` by PR #2975, commit `b453f40d8`
(`fix(orchestrator): train shared sampled nodes exactly once across branches`). This note is
retained because it records the rationale and alternatives that led to the fix; no additional
TTT-branch implementation is required.

## Summary

A verifier trace is a graph, while `trace_to_samples` flattens root-to-leaf branches. If two
branches share a model-sampled prefix, the same sampled node appears with a true loss mask in
both branch samples. The trainer no longer has node identity, so it cannot recognize that the
two token spans came from one sampling event.

## Potential effect

If a sampled node lies on N branches, its tokens may contribute N policy-gradient terms. Each
sample also carries an advantage, so this is not necessarily equivalent to a benign duplicate
context. The shared prefix can receive disproportionate gradient weight.

Affected graph shapes may include subagents, tree search, best-of-N exploration, token-drift
forks, and feature-specific side branches. Linear traces are unaffected.

## Change considered on the TTT branch

The branch walked branches in stable order and granted each sampled node a trainable mask only
on its first branch. Later branches retained the node as masked context. The same iterator also
split TTT samples by adapter version.

## Why it was kept out of the TTT patch

Per-branch masks currently answer “was this token sampled on this path?” Changing flattening to
“train this sampling event once globally” is plausible, but it changes the effective objective
for every branching algorithm. Some algorithms may intentionally want path-weighted shared
prefixes. That policy should be decided in the core trajectory design, not by TTT.

The policy was therefore reviewed and landed as its own Prime-RL change. Current `main`
implements first-branch ownership for a shared sampled node and keeps Echo's branch/sample
alignment in the same core trajectory layer.

## Design questions

- Is the unit of credit a sampled node, a path, or a leaf outcome?
- If branches have different advantages, which advantage should own a shared prefix?
- Should shared-prefix weight be averaged, summed, or assigned to the first branch?
- Can explicit node IDs be carried to the trainer instead of resolving policy in conversion?

## Suggested tests

- Linear traces remain byte-for-byte unchanged.
- A two-leaf graph makes the chosen shared-prefix weighting explicit.
- Branch-order changes do not accidentally change the objective unless order is the policy.
- Echo and other algorithms with observation loss receive aligned samples.
- Multi-turn and nested forks are covered.

## Relevant code

- `src/prime_rl/orchestrator/trajectories.py`
- `src/prime_rl/orchestrator/algo/echo.py`
- `verifiers.v1` trace/branch graph semantics
