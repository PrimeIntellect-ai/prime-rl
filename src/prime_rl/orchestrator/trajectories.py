"""Adapt vf-nano `Trace` dicts into the orchestrator's training data.

The env server returns a `Trace` (as a JSON dict) with the trajectory already
segmented into `branches` (a branch is a maximal linear run of turns whose context
only grew) and each turn carrying `tokens` (`prompt_ids` / `completion_ids` /
`completion_logprobs`, populated by the renderer client). This module turns that
into:

  - `trace_to_output`: a RolloutOutput-shaped dict (adds `example_id`,
    `completion`, `token_usage`) the rest of the orchestrator already reads.
  - `trace_to_samples`: one `TrainingSample` per branch — branching is done
    server-side, so we just lay out prompt + completion token ids, masking the
    intervening prompt tokens the model didn't generate and keeping per-token
    logprobs.
"""

from __future__ import annotations

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


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


def trace_to_output(trace: dict, example_id: int) -> dict:
    """Adapt a vf-nano `Trace` dict into the RolloutOutput-shaped dict the sink /
    monitor / eval consume. Keeps the Trace fields (`trajectory`, `branches`,
    `reward`, `metrics`, `error`, `is_truncated`) and adds the ones they read."""
    out = dict(trace)
    out["example_id"] = example_id
    trajectory = trace.get("trajectory") or []
    last = trajectory[-1] if trajectory else None
    # `completion` is used for no-response detection + sample logging: the last
    # assistant message, or [] when there was no (non-empty) response.
    content = last["response"]["message"].get("content") if last else None
    out["completion"] = [last["response"]["message"]] if content else []
    out["token_usage"] = {
        "input_tokens": sum(_turn_token_count(t, "prompt_ids", "prompt_tokens") for t in trajectory),
        "output_tokens": sum(_turn_token_count(t, "completion_ids", "completion_tokens") for t in trajectory),
        "final_input_tokens": _turn_token_count(last, "prompt_ids", "prompt_tokens"),
        "final_output_tokens": _turn_token_count(last, "completion_ids", "completion_tokens"),
    }
    return out


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
            f"No trainable samples for example {raw.get('example_id')} "
            f"(error={raw.get('error') is not None}, stop={raw.get('stop_condition')})."
        )
    return samples
