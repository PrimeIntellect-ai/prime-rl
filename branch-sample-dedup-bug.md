# Bug: branches sharing a sampled prefix double-count those tokens in training

**Status**: latent on `main` (no in-tree harness currently triggers it); fixed on the TTT
experiment branch (`sebastian/ttt-2026-07-06`, commit `1a821a800`) — this note is the
standalone description for an upstream PR, independent of the TTT work.

## Where

`prime-rl`, `src/prime_rl/orchestrator/trajectories.py::trace_to_samples` (as of upstream
`df2acf487`). The verifiers side (`Branch.sampled_mask`) is *not* wrong per se — it
faithfully reports "was this token model-sampled?" per branch — the bug is in how the
consumer turns overlapping branches into training samples.

## The bug

A v1 `Trace` is a message **graph**; `trace.branches` is the set of root→leaf paths, and
`trace_to_samples` builds **one training sample per branch**, using `branch.sampled_mask`
as the trainable-token mask:

```python
for branch in trace.branches:
    mask = branch.sampled_mask       # True on every model-sampled token IN THIS PATH
    ...
    samples.append(TrainingSample(token_ids=branch.token_ids, mask=mask, ...))
```

Branches are *paths*, so any node shared by several branches appears in several samples.
For **input** nodes (system prompt, user/tool messages) that's correct and intended — they
are context (mask `False`) everywhere. But for **sampled** nodes (assistant completions,
mask `True`) it means: a sampled token that lies on N branches is trainable in N samples.
The trainer normalizes the loss by global token counts and has no cross-sample dedup, so
those tokens receive **N× the gradient weight** of tokens on a single branch. Advantages
are also stamped per sample, so the same completion tokens contribute N advantage-weighted
policy-gradient terms.

## Why it doesn't fire today

Every in-tree harness produces branches that share only **input** nodes:

- linear harnesses → one branch, no sharing;
- the compaction-style harnesses (`compact` example, the new `compacting`) rebuild the
  prompt as `[system, user(summary)]` → branches share at most the system node
  (mask all-`False`);
- renderer-level token-drift forks (`graph._commit_turn`'s token-identity tightening)
  fork *at* the divergence, and the reused prefix before the fork is per-node reused —
  but the drifted turn re-samples, creating a new node; the shared prefix nodes'
  sampled tokens DO end up in both branches' samples. This is the one existing path
  that can already trigger the bug in the wild — it needs a retokenization break
  mid-rollout (BPE drift, a template that drops `<think>` across user turns without
  the bridge applying), which is rare but real.

Any future harness with genuine mid-trajectory forks — **subagents** branching off a
shared sampled prefix, tree-search / best-of-n exploration recorded on one trace, or the
TTT Q&A side-generations that motivated finding this — makes the bug acute: with k forks
off a leaf, the entire shared trajectory's completion tokens get (k+…)× weight.

## The fix

One rule in `trace_to_samples`: **a sampled node is trainable exactly once across the
trace.** Walk branches in their stable order (`trace.branches` follows node creation
order, so "first branch containing a node" is deterministic); the first branch containing
a sampled node keeps that node's mask, every later branch re-carries the node's tokens as
pure context (mask `False`):

```python
samples: list[TrainingSample] = []
trained_nodes: set[int] = set()  # id(node) of sampled nodes already granted their mask
for branch in trace.branches:
    mask: list[bool] = []
    for node in branch.nodes:
        if node.sampled and any(node.mask) and id(node) in trained_nodes:
            mask.extend([False] * len(node.mask))   # context here; trained elsewhere
        else:
            if node.sampled and any(node.mask):
                trained_nodes.add(id(node))
            mask.extend(node.mask)
    if not any(mask):
        continue
    ...
```

Properties:

- **No behavior change for current traces**: with input-only sharing, the computed mask
  equals `branch.sampled_mask` exactly (verified by the existing test suite passing
  unchanged).
- Every sampled token still receives gradient exactly once, in the branch where it first
  appears — losing no signal, removing only the duplication.
- Later branches still *see* the shared completion as context, so their own new tokens
  train against the correct conditioning.
- `id(node)` is a safe key here: branches are views over the same `trace.nodes` list, so
  a shared node is the same object in every branch.

Alternative placements considered: (a) fix in `Branch.sampled_mask` — wrong layer, the
property's per-branch semantics ("was this sampled?") are correct and other consumers
(e.g. `num_output_tokens`) rely on them; (b) dedup in the trainer — too late, samples
are packed/split across micro batches and the node identity is gone. The sample builder
is the single point that turns the graph into flat sequences, so it owns the invariant.

## Test

`tests/unit/ttt/test_replay.py::test_shared_sampled_prefix_trained_once` on the branch
(a two-fork trace off a sampled leaf; asserts the shared tokens are mask-True in exactly
one of the two samples, and each branch's own tail stays trainable). For an upstream PR
this test should move next to the other `trace_to_samples` coverage and drop the TTT
naming — the repro needs nothing TTT-specific.
