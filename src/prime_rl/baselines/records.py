from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _message_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages.strip()
    if not messages:
        return ""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        elif content is not None:
            parts.append(str(content))
    return "\n".join(parts).strip()


def _json_or_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _usage(output: Mapping[str, Any]) -> tuple[float, float]:
    token_usage = output.get("token_usage")
    if isinstance(token_usage, Mapping):
        return (
            float(token_usage.get("input_tokens") or 0.0),
            float(token_usage.get("output_tokens") or 0.0),
        )

    prompt_tokens = 0.0
    completion_tokens = 0.0
    for step in output.get("trajectory") or []:
        if not isinstance(step, Mapping):
            continue
        response = step.get("response")
        usage = None
        if isinstance(response, Mapping):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if isinstance(usage, Mapping):
            prompt_tokens += float(usage.get("prompt_tokens") or usage.get("input_tokens") or 0.0)
            completion_tokens += float(usage.get("completion_tokens") or usage.get("output_tokens") or 0.0)
    return prompt_tokens, completion_tokens


def parse_answer(env: Any, completion: Any) -> str | None:
    parser = getattr(env, "parser", None)
    if parser is None:
        return None
    parse = getattr(parser, "parse_answer", None)
    if parse is None:
        return None
    answer = parse(completion)
    return None if answer is None else str(answer)


def _latest_judge_decision(output: Mapping[str, Any]) -> dict[str, Any] | None:
    decision = output.get("judge_decision_last")
    if isinstance(decision, Mapping):
        return dict(decision)
    decisions = output.get("judge_decision")
    if not isinstance(decisions, Mapping) or not decisions:
        return None
    latest = next(reversed(decisions.values()))
    return dict(latest) if isinstance(latest, Mapping) else None


def _hard_correct_from_judge_decision(decision: Mapping[str, Any] | None) -> bool | None:
    if decision is None:
        return None
    p_correct = decision.get("p_correct")
    if not isinstance(p_correct, int | float):
        return None
    support = decision.get("support")
    policy = support.get("policy") if isinstance(support, Mapping) else None
    threshold = 0.5
    if isinstance(policy, Mapping) and isinstance(policy.get("threshold"), int | float):
        threshold = float(policy["threshold"])
    return float(p_correct) > threshold


def flatten_output(
    *,
    output: Mapping[str, Any],
    env: Any,
    run_id: str,
    env_id: str,
    protocol: str,
    dataset: str,
    model: str,
    seed: int,
    trial_index: int,
    success_threshold: float,
) -> dict[str, Any]:
    completion = output.get("completion")
    input_tokens, output_tokens = _usage(output)
    reward = float(output.get("reward") or 0.0)
    timing = output.get("timing") if isinstance(output.get("timing"), Mapping) else {}
    info = _json_or_value(output.get("info"))
    judge_decision = _latest_judge_decision(output)
    posterior_correct = None
    if isinstance(judge_decision, Mapping) and isinstance(judge_decision.get("p_correct"), int | float):
        posterior_correct = float(judge_decision["p_correct"])
    hard_correct = _hard_correct_from_judge_decision(judge_decision)

    return {
        "run_id": run_id,
        "env_id": env_id,
        "protocol": protocol,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "example_id": str(output.get("example_id")),
        "trial_index": trial_index,
        "task": output.get("task"),
        "target": output.get("answer"),
        "parsed_answer": parse_answer(env, completion),
        "response": _message_text(completion),
        "reward": reward,
        "correct": hard_correct if hard_correct is not None else reward >= success_threshold,
        "is_completed": output.get("is_completed"),
        "is_truncated": output.get("is_truncated"),
        "stop_condition": output.get("stop_condition"),
        "error": output.get("error"),
        "judge_response": output.get("judge_response"),
        "judge_decision": output.get("judge_decision"),
        "judge_decision_last": output.get("judge_decision_last"),
        "posterior_correct": posterior_correct,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "generation_ms": timing.get("generation_ms"),
        "scoring_ms": timing.get("scoring_ms"),
        "total_ms": timing.get("total_ms"),
        "metrics": output.get("metrics") or {},
        "info": info,
    }


def token_summary_row(
    records: list[dict[str, Any]], *, model: str, protocol: str, dataset: str, seed: int
) -> dict[str, Any]:
    input_tokens = sum(float(r.get("input_tokens") or 0.0) for r in records)
    output_tokens = sum(float(r.get("output_tokens") or 0.0) for r in records)
    return {
        "model_name": model,
        "model_short": model.split("/")[-1],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "reasoning_tokens": None,
        "cache_read": None,
        "run_model": model.split("/")[-1],
        "effort": None,
        "protocol": protocol,
        "seed": seed,
        "dataset": dataset,
    }
