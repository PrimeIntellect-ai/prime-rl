"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything here is
attribute access — no dicts. The trace is a message graph (`trace.nodes`); each `trace.branches`
entry (a root→leaf path) is first-class and carries its own flat token sequence
(`branch.token_ids` / `branch.sampled_mask` / `branch.logprobs`), so a branch yields one
training sample directly. Token-length readers (`completion_len`, `total_tokens`, `num_turns`)
live on `vf.Trace` itself.

RL/OPD rollouts come from the renderer client, so every node already carries its tokens. SFT
rolls out against a teacher over plain chat-completions (the oai client returns messages, not
tokens), so `backfill_trace` re-renders the conversation to populate the nodes before
`trace_to_samples` runs.
"""

from __future__ import annotations

import verifiers.v1 as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def backfill_trace(trace: vf.Trace, renderer) -> None:
    """Populate per-node tokens for a trace whose rollout client returned none (SFT against a
    teacher served over plain chat-completions). Re-renders each branch's messages with the
    student renderer and attributes the tokens back onto the graph nodes by per-message span
    (leading template scaffold folds into the following message), trainable where the renderer
    marks them model-sampled (`sampled_mask` — assistant content only). SFT trains on the
    teacher's tokens, not its logprobs, so logprobs stay empty. Mutates the trace in place;
    nodes already carrying tokens are left untouched. Tools aren't re-supplied, so this targets
    text turns."""
    from verifiers.v1.clients.renderer import message_to_wire

    for branch in trace.branches:
        nodes = branch.nodes
        rendered = renderer.render([message_to_wire(node.message) for node in nodes])
        ids, spans, sampled = rendered.token_ids, rendered.message_token_spans(), rendered.sampled_mask
        prev = 0
        for i, node in enumerate(nodes):
            if node.token_ids:  # shared prefix node already filled by an earlier branch
                prev += len(node.token_ids)
                continue
            span = spans[i] if i < len(spans) else None
            end = span[1] if span is not None else prev
            node.token_ids = ids[prev:end]
            node.mask = list(sampled[prev:end]) if sampled else [False] * (end - prev)
            node.logprobs = []
            prev = end


def trace_to_samples(trace: vf.Trace, *, env_name: str = "") -> list[TrainingSample]:
    """Convert a v1 `Trace` into `TrainingSample`s — one per branch.

    Each `trace.branches` entry is already a flat token sequence (`branch.token_ids` /
    `branch.sampled_mask` / `branch.logprobs`), so a sample carries it directly: `mask` marks
    the trainable (model-sampled) tokens, the context tokens between completions stay masked
    out. On a rollout error the whole completion is masked out. Branches with no sampled tokens
    (e.g. an openai client carrying none) yield nothing.
    """
    has_error = trace.has_error
    samples: list[TrainingSample] = []
    for branch in trace.branches:
        mask = branch.sampled_mask
        if not any(mask):
            continue
        samples.append(
            TrainingSample(
                token_ids=branch.token_ids,
                mask=[m and not has_error for m in mask],
                logprobs=branch.logprobs,
                temperatures=[],  # filled by TrainSink.process_group
                teacher_logprobs=None,
                advantage=None,
                env_name=env_name,
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples
