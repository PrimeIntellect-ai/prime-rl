"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything here is
attribute access — no dicts. The trace is a message graph (`trace.nodes`); each `trace.branches`
entry (a root→leaf path) is first-class and carries its own flat token sequence
(`branch.token_ids` / `branch.sampled_mask` / `branch.logprobs`), so a branch yields one
training sample directly. Token-length readers (`completion_len`, `total_tokens`, `num_turns`)
live on `vf.Trace` itself.

Training is renderer-only across every mode (RL/OPD student, SFT teacher), so every node
always carries its tokens — no backfill needed.
"""

from __future__ import annotations

import verifiers.v1 as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


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
