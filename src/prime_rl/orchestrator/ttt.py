"""TTT (test-time training) orchestrator helpers.

This module hosts compaction-aligned TTT helpers, separate from the
chunked TTT path so the two strategies can be developed and tested
independently. See ``docs/ttt-implementation-plan.md`` for the design.

Today the only piece that's self-contained — and therefore useful in
isolation before the rest of the TTT machinery (learner service,
transport types, trainer-side replay) lands — is the trajectory parser
that detects compaction events on RLM-harness rollouts. The parser is
pure (a function over a list of dicts), depends on nothing else in
prime-rl, and is consumed by the (future) compaction-aligned update
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CompactionEvent:
    """One compaction event detected in an RLM-harness trajectory.

    Attributes:
        step_index: Index into the trajectory at which the compaction
            occurred — i.e., the step whose ``prompt`` is the
            post-compaction reset (``[system, user(framing + summary)]``).
        pre_compaction_message_count: Number of messages in the
            accumulated conversation immediately before this compaction
            fired. The dismissed slice is messages
            ``[0:pre_compaction_message_count)`` of the pre-compaction
            conversation; downstream consumers use this to recover the
            token range to train the LoRA on.
    """

    step_index: int
    pre_compaction_message_count: int


def detect_compaction_events(
    trajectory: Iterable[dict[str, Any]],
) -> list[CompactionEvent]:
    """Detect compaction events in an RLM-harness rollout trajectory.

    Structural signal: a trajectory step whose ``prompt`` (message
    list) is *shorter* than the conversation accumulated up to that
    point can only happen when the harness reset the conversation to
    ``[system, user(framing + summary)]``. See
    ``deps/verifiers/.../envs/experimental/composable/harnesses/rlm.py``
    (``render_completion_with_branches``) for the reference detection
    on the rendering side — this function mirrors that logic for the
    training-side consumer.

    The detection is purely structural: ``COMPACTION_BOUNDARY_MARKER``
    in the harness is a debug aid for human-readable rendering, not a
    contract we depend on.

    Sub-agent calls (``X-RLM-Depth > 0``) are elided upstream by the
    harness's ``keep_trajectory_step`` filter before the trajectory
    reaches the orchestrator, so this function only sees parent-agent
    steps. Parent-level compaction events are what we want to train on.

    Args:
        trajectory: Iterable of trajectory steps as produced by
            verifiers. Each step is a mapping with ``prompt`` and
            ``completion`` keys whose values are message lists.

    Returns:
        Compaction events in trajectory order. Empty list when the
        trajectory contains no compaction — either short rollouts that
        never exceeded ``summarize_at_tokens``, or non-RLM rollouts
        whose harness doesn't compact at all.
    """
    events: list[CompactionEvent] = []
    steps = list(trajectory)
    if not steps:
        return events

    first = steps[0]
    prev_len = len(first["prompt"]) + len(first["completion"])

    for i, step in enumerate(steps[1:], start=1):
        prompt_len = len(step["prompt"])
        if prompt_len < prev_len:
            events.append(
                CompactionEvent(
                    step_index=i,
                    pre_compaction_message_count=prev_len,
                )
            )
        prev_len = prompt_len + len(step["completion"])

    return events
