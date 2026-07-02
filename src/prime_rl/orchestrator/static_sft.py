"""Static SFT replay: dataset rows -> tokenized rollouts, no env server or model client.

A row's messages are rendered with the policy renderer; the renderer's
``sampled_mask`` marks the assistant emission tokens as the CE targets, exactly
as if a teacher had sampled them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf
from verifiers.v1.dialects.chat import message_to_wire, parse_message
from verifiers.v1.graph import MessageNode

from prime_rl.configs.algorithm import StaticDatasetConfig
from prime_rl.orchestrator.types import Rollout
from prime_rl.utils.chat_template import normalize_messages

if TYPE_CHECKING:
    from datasets import Dataset
    from renderers.base import RenderedTokens, Renderer


def load_static_sft_dataset(config: StaticDatasetConfig) -> Dataset:
    """Load the dataset the env replays; ``task_idx`` indexes it directly (rows
    stay Arrow-backed — the dispatcher only ever reads one row at a time)."""
    from datasets import load_dataset

    dataset = load_dataset(config.name, config.subset, split=config.split)
    if config.max_examples is not None:
        dataset = dataset.take(min(config.max_examples, len(dataset)))
    if len(dataset) == 0:
        raise ValueError(f"Static SFT dataset {config.name!r} split {config.split!r} loaded no rows.")
    return dataset


def row_to_rollout(row: dict, task_idx: int, config: StaticDatasetConfig, renderer: Renderer) -> Rollout:
    """Render one dataset row into a completed, tokenized rollout."""
    messages = [parse_message(m) for m in _row_messages(row, config)]
    rendered = renderer.render([message_to_wire(m) for m in messages], tools=_row_tools(row, config))
    if not any(rendered.sampled_mask):
        raise ValueError("Static SFT row rendered no assistant emission tokens to train on.")
    return Rollout(
        task=vf.Task(idx=task_idx, prompt=None),
        nodes=_nodes_from_rendered(messages, rendered),
        is_completed=True,
        stop_condition="static_dataset",
    )


def _row_messages(row: dict, config: StaticDatasetConfig) -> list[dict[str, Any]]:
    if row.get(config.messages_column) is not None:
        messages = normalize_messages(_maybe_json(row[config.messages_column]), default_role="assistant")
    elif config.prompt_column in row and config.completion_column in row:
        prompt = normalize_messages(_maybe_json(row[config.prompt_column]), default_role="user")
        completion = normalize_messages(_maybe_json(row[config.completion_column]), default_role="assistant")
        messages = prompt + completion
    else:
        raise ValueError(
            "Static SFT rows must have either "
            f"'{config.messages_column}' or both '{config.prompt_column}' and '{config.completion_column}'."
        )
    return [_normalize_tool_calls(m) for m in messages]


def _normalize_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Coerce stored tool calls to the OAI wire shape the typed ``ToolCall``
    expects: datasets carry ``arguments`` as parquet structs/dicts (it must be
    a raw JSON string) and sometimes flatten ``function`` or drop ``id``."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    normalized = []
    for i, call in enumerate(calls):
        function = call.get("function") or {"name": call.get("name"), "arguments": call.get("arguments")}
        if not isinstance(function.get("arguments"), str):
            function = {**function, "arguments": json.dumps(function.get("arguments"))}
        normalized.append({**call, "id": call.get("id") or f"call_{i}", "function": function})
    return {**message, "tool_calls": normalized}


def _row_tools(row: dict, config: StaticDatasetConfig) -> list[dict[str, Any]] | None:
    """Tool definitions in OAI form. Accepts the verifiers ``tool_defs`` column
    and bare ``{name, description, parameters}`` shapes, mirroring the offline
    SFT trainer's normalization."""
    raw = row.get(config.tools_column, row.get("tool_defs"))
    if raw is None:
        return None
    parsed = _maybe_json(raw)
    if not isinstance(parsed, list):
        raise TypeError(f"Static SFT tools must be a list, got {type(parsed).__name__}")
    return [
        t
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t
        else {"type": "function", "function": {k: t.get(k) for k in ("name", "description", "parameters")}}
        for t in parsed
    ] or None


def _nodes_from_rendered(messages: vf.Messages, rendered: RenderedTokens) -> list[MessageNode]:
    """One node per message, chained linearly. Each node takes its leading
    scaffold plus its own attributed span (the last node sweeps any trailing
    scaffold), so concatenating the path reproduces the exact rendered
    sequence. ``mask`` is the renderer's ``sampled_mask`` slice — True only on
    assistant emission tokens, the CE targets."""
    spans = rendered.message_token_spans()
    has_content = bool(rendered.is_content)
    nodes: list[MessageNode] = []
    cursor = 0
    for i, (message, span) in enumerate(zip(messages, spans)):
        # max() guards against non-contiguous attribution (outer spans may
        # overlap); the cursor must never move backward or tokens duplicate.
        end = len(rendered.token_ids) if i == len(messages) - 1 else (max(span[1], cursor) if span else cursor)
        nodes.append(
            MessageNode(
                parent=i - 1 if i else None,
                message=message,
                sampled=message.role == "assistant",
                token_ids=rendered.token_ids[cursor:end],
                mask=rendered.sampled_mask[cursor:end],
                is_content=rendered.is_content[cursor:end] if has_content else [],
            )
        )
        cursor = end
    return nodes


def _maybe_json(value: Any) -> Any:
    """Decode a JSON-encoded messages/tools cell; anything that doesn't parse
    to a message-shaped value (a list, or a dict with a ``role``) is content
    text, not serialization — e.g. a plain-text completion of ``"42"``."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list) or (isinstance(parsed, dict) and "role" in parsed):
            return parsed
    return value
