"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything
here is attribute access — no dicts. The trajectory is segmented into `branches`
(recomputed on the consumer; a branch is a maximal linear run of turns whose context
only grew), each turn carrying `tokens` (`prompt_ids` / `completion_ids` /
`completion_logprobs`, from the renderer client). Token-length readers
(`completion_len`, `total_tokens`, `has_response`) live on `vf.Trace` itself.
"""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1 import graph

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def backfill_rollout_tokens(trace: vf.Trace, tokenizer) -> None:
    """No-op under the message-graph trace: training is renderer-only, so every turn already
    carries token ids (the graph stores them per message). The old SFT-via-chat-client
    backfill (rendering token-less turns) is unsupported — `trajectory` is now a read-only
    view over the graph, so it can't write tokens back."""
    return


def trace_to_samples(trace: vf.Trace, *, env_name: str = "") -> list[TrainingSample]:
    """Convert a v1 `Trace` into `TrainingSample`s — one per branch.

    Walks the message graph: each branch (a leaf→root path) is a flat token sequence built
    by concatenating its nodes' `token_ids`/`sampled_mask`/`logprobs` (`graph
    .branch_token_sequences`). The prompt is everything up to the first model-sampled token;
    the completion is the rest, trainable where `sampled_mask` is True (the per-turn context
    tokens between completions stay masked out). On a rollout error the whole completion is
    masked out. Branches with no sampled tokens (e.g. an openai client carrying none) yield
    nothing.
    """
    has_error = trace.has_error
    samples: list[TrainingSample] = []
    for ids, sampled_mask, logprobs in graph.branch_token_sequences(trace):
        if not any(sampled_mask):
            continue
        first = sampled_mask.index(True)  # split prompt | completion at the first sampled token
        prompt_ids = ids[:first]
        samples.append(
            TrainingSample(
                prompt_ids=prompt_ids,
                prompt_mask=[False] * len(prompt_ids),
                completion_ids=ids[first:],
                completion_mask=[m and not has_error for m in sampled_mask[first:]],
                completion_logprobs=logprobs[first:],
                completion_temperatures=[],  # filled by TrainSink.process_group
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
