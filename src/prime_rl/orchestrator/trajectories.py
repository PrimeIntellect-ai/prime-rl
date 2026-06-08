"""Turn a vf-nano `Trace` (the env server's native output) into training data.

The orchestrator consumes the `Trace` dict directly — no legacy RolloutOutput
shape. The trajectory is already segmented into `branches` (a branch is a maximal
linear run of turns whose context only grew), each turn carrying `tokens`
(`prompt_ids` / `completion_ids` / `completion_logprobs`, from the renderer
client). The readers below pull what the sinks/eval need straight off the Trace.
"""

from __future__ import annotations

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def _last_turn(trace: dict) -> dict | None:
    trajectory = trace.get("trajectory") or []
    return trajectory[-1] if trajectory else None


def _turn_token_count(turn: dict | None, ids_key: str, usage_key: str) -> int:
    """Token count for one turn — prefer the engine token ids, fall back to the
    response usage (the openai/eval client carries usage but no token ids)."""
    if turn is None:
        return 0
    tokens = turn.get("tokens") or {}
    ids = tokens.get(ids_key)
    if ids:
        return len(ids)
    usage = (turn.get("response") or {}).get("usage") or {}
    return int(usage.get(usage_key) or 0)


def trace_completion_len(trace: dict) -> int:
    """Completion tokens generated in the final turn (for length metrics)."""
    return _turn_token_count(_last_turn(trace), "completion_ids", "completion_tokens")


def trace_total_tokens(trace: dict) -> int:
    """Final-turn context size (prompt + completion) — used for token batching."""
    last = _last_turn(trace)
    return _turn_token_count(last, "prompt_ids", "prompt_tokens") + _turn_token_count(
        last, "completion_ids", "completion_tokens"
    )


def trace_has_response(trace: dict) -> bool:
    """Whether the final assistant turn produced non-empty content."""
    last = _last_turn(trace)
    return bool(last and (last["response"]["message"].get("content")))


def trace_to_samples(raw: dict, *, env_name: str = "") -> list[TrainingSample]:
    """Convert a vf-nano Trace dict into `TrainingSample`s — one per branch.

    Within a branch, turn `i`'s prompt extends turn `i-1`'s prompt+completion, so
    we append each turn's new prompt tokens (mask `False`, the model didn't
    generate them) then its completion tokens (mask `True`, with logprobs). On a
    rollout error the whole completion is masked out so it isn't trained on.
    Branches whose turns carry no token ids (e.g. an openai client) yield nothing.
    """
    has_error = raw.get("error") is not None
    samples: list[TrainingSample] = []
    for branch in raw.get("branches") or []:
        turns = branch.get("turns") or []
        token_lists = [t.get("tokens") for t in turns]
        if not turns or any(tk is None for tk in token_lists):
            continue

        first = token_lists[0]
        prompt_ids = list(first["prompt_ids"])
        completion_ids: list[int] = []
        completion_mask: list[bool] = []
        completion_logprobs: list[float] = []
        prefix_len = len(prompt_ids)  # running prompt+completion length of the branch

        for tokens in token_lists:
            new_prompt = list(tokens["prompt_ids"][prefix_len:])
            completion_ids.extend(new_prompt)
            completion_mask.extend([False] * len(new_prompt))
            completion_logprobs.extend([0.0] * len(new_prompt))

            turn_completion = list(tokens["completion_ids"])
            completion_ids.extend(turn_completion)
            completion_mask.extend([not has_error] * len(turn_completion))
            logprobs = list(tokens.get("completion_logprobs") or [])
            if len(logprobs) != len(turn_completion):
                logprobs = (logprobs + [0.0] * len(turn_completion))[: len(turn_completion)]
            completion_logprobs.extend(logprobs)

            prefix_len = len(tokens["prompt_ids"]) + len(turn_completion)

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
            f"No trainable samples (error={raw.get('error') is not None}, "
            f"stop={raw.get('stop_condition')}, num_turns={len(raw.get('trajectory') or [])})."
        )
    return samples
