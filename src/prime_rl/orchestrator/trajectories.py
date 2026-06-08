"""Turn a vf-nano `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything
here is attribute access — no dicts. The trajectory is segmented into `branches`
(recomputed on the consumer; a branch is a maximal linear run of turns whose context
only grew), each turn carrying `tokens` (`prompt_ids` / `completion_ids` /
`completion_logprobs`, from the renderer client). The readers pull what the
sinks/eval need straight off the Trace.
"""

from __future__ import annotations

import verifiers.nano as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def _last_turn(trace: vf.Trace) -> vf.Turn | None:
    return trace.trajectory[-1] if trace.trajectory else None


def _turn_token_count(turn: vf.Turn | None, ids_attr: str, usage_attr: str) -> int:
    """Token count for one turn — prefer the engine token ids, fall back to the
    response usage (the openai/eval client carries usage but no token ids)."""
    if turn is None:
        return 0
    if turn.tokens and getattr(turn.tokens, ids_attr):
        return len(getattr(turn.tokens, ids_attr))
    usage = turn.response.usage
    return int(getattr(usage, usage_attr)) if usage else 0


def trace_completion_len(trace: vf.Trace) -> int:
    """Completion tokens generated in the final turn (for length metrics)."""
    return _turn_token_count(_last_turn(trace), "completion_ids", "completion_tokens")


def trace_total_tokens(trace: vf.Trace) -> int:
    """Final-turn context size (prompt + completion) — used for token batching."""
    last = _last_turn(trace)
    return _turn_token_count(last, "prompt_ids", "prompt_tokens") + _turn_token_count(
        last, "completion_ids", "completion_tokens"
    )


def trace_has_response(trace: vf.Trace) -> bool:
    """Whether the final assistant turn produced non-empty content."""
    last = _last_turn(trace)
    return bool(last and last.response.message.content)


def trace_to_samples(trace: vf.Trace, *, env_name: str = "") -> list[TrainingSample]:
    """Convert a vf-nano `Trace` into `TrainingSample`s — one per branch.

    Stitch each branch's turns into one token sequence: the branch's first-turn
    prompt, then for each turn the new context tokens since the running prefix (mask
    `False`, the model didn't generate them) followed by that turn's completion tokens
    (mask `True`, with logprobs). On a rollout error the whole completion is masked
    out. Branches whose turns carry no token ids (e.g. an openai client) yield nothing.
    """
    has_error = trace.error is not None
    samples: list[TrainingSample] = []
    for branch in trace.branches:
        turns = branch.turns
        if not turns or any(turn.tokens is None for turn in turns):
            continue

        prompt_ids = list(turns[0].tokens.prompt_ids)
        completion_ids: list[int] = []
        completion_mask: list[bool] = []
        completion_logprobs: list[float] = []
        prefix_len = len(prompt_ids)  # running prompt+completion length of the branch

        for turn in turns:
            tokens = turn.tokens
            new_prompt = list(tokens.prompt_ids[prefix_len:])
            completion_ids.extend(new_prompt)
            completion_mask.extend([False] * len(new_prompt))
            completion_logprobs.extend([0.0] * len(new_prompt))

            turn_completion = list(tokens.completion_ids)
            completion_ids.extend(turn_completion)
            completion_mask.extend([not has_error] * len(turn_completion))
            logprobs = list(tokens.completion_logprobs)
            if len(logprobs) != len(turn_completion):
                logprobs = (logprobs + [0.0] * len(turn_completion))[: len(turn_completion)]
            completion_logprobs.extend(logprobs)

            prefix_len = len(tokens.prompt_ids) + len(turn_completion)

        if not completion_ids:
            continue
        samples.append(
            TrainingSample(
                prompt_ids=prompt_ids,
                prompt_mask=[False] * len(prompt_ids),
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_logprobs=completion_logprobs,
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
