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
    from renderers.base import RenderedTokens, Renderer


def load_static_sft_rows(config: StaticDatasetConfig) -> list[dict]:
    """Load the dataset rows the env replays; ``task_idx`` indexes this list."""
    from datasets import load_dataset

    dataset = load_dataset(config.name, config.subset, split=config.split)
    if config.max_examples is not None:
        dataset = dataset.select(range(min(config.max_examples, len(dataset))))
    rows = [dict(row) for row in dataset]
    if not rows:
        raise ValueError(f"Static SFT dataset {config.name!r} split {config.split!r} loaded no rows.")
    return rows


def row_to_rollout(row: dict, task_idx: int, config: StaticDatasetConfig, renderer: Renderer) -> Rollout:
    """Render one dataset row into a completed, tokenized rollout."""
    messages = [parse_message(m) for m in _row_messages(row, config)]
    rendered = renderer.render([message_to_wire(m) for m in messages], tools=_row_tools(row, config) or None)
    if not rendered.sampled_mask:
        raise ValueError(
            f"{type(renderer).__name__} does not attribute sampled tokens — static sft needs a "
            'hand-coded [renderer] config (e.g. name = "qwen3").'
        )
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
        return normalize_messages(_maybe_json(row[config.messages_column]), default_role="assistant")
    if config.prompt_column not in row or config.completion_column not in row:
        raise ValueError(
            "Static SFT rows must have either "
            f"'{config.messages_column}' or both '{config.prompt_column}' and '{config.completion_column}'."
        )
    prompt = normalize_messages(_maybe_json(row[config.prompt_column]), default_role="user")
    completion = normalize_messages(_maybe_json(row[config.completion_column]), default_role="assistant")
    return prompt + completion


def _row_tools(row: dict, config: StaticDatasetConfig) -> list[dict[str, Any]]:
    raw = row.get(config.tools_column)
    if raw is None:
        return []
    parsed = _maybe_json(raw)
    if not isinstance(parsed, list):
        raise TypeError(f"Static SFT tools must be a list, got {type(parsed).__name__}")
    return parsed


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
        end = len(rendered.token_ids) if i == len(messages) - 1 else (span[1] if span else cursor)
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
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
