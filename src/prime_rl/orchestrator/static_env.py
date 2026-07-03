"""Dataset-backed train env: static SFT rows rendered into honest Traces.

A ``StaticSFTEnv`` is duck-type compatible with :class:`~prime_rl.orchestrator.envs.TrainEnv`
but owns no env server and calls no model: ``start()`` loads the HF dataset and
``run_rollout(task_idx)`` renders that row's messages into a ``Rollout`` whose
``MessageNode``\\ s carry their real delta tokens — assistant nodes are ``sampled``
with the loss-mask marking their trainable span. Everything downstream (the sink's
``trace_to_samples``, token batching, rollout records) consumes it like any generated
rollout; the ``static-sft`` algorithm routes the mask-True tokens into the ``ce`` loss.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import TypeAdapter
from renderers.base import Renderer, create_renderer
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.orchestrator import StaticDatasetConfig, TrainEnvConfig
from prime_rl.configs.sft import LossMaskConfig
from prime_rl.orchestrator.algo import Algorithm
from prime_rl.orchestrator.types import Rollout
from prime_rl.utils.chat_template import (
    deserialize_tool_calls,
    normalize_messages,
    strip_message_content,
)
from prime_rl.utils.logger import get_logger

_MESSAGE_ADAPTER: TypeAdapter = TypeAdapter(vf.Message)

STATIC_ROLLOUT_TYPE = Rollout[vf.WireTask]

# Per-role keys the typed v1 messages accept — anything else in a dataset row's
# message dict (metadata columns, provider extras) is dropped before validation.
_MESSAGE_KEYS = {
    "system": ("role", "content"),
    "user": ("role", "content"),
    "assistant": ("role", "content", "reasoning_content", "tool_calls"),
    "tool": ("role", "content", "tool_call_id", "name"),
}


class NullSampler:
    """Sampler stand-in for envs that generate nothing. ``pool=None`` tells the
    dispatcher there is no client to pin; never-live means rollouts never age
    off-policy."""

    samples_from_live_policy = False
    pool = None

    def __init__(self) -> None:
        self.connected_pools: list = []

    async def setup(self) -> None:
        pass


def resolve_messages(example: dict) -> list[dict]:
    """A dataset row -> OAI-shaped message dicts. ``messages`` takes precedence
    over the ``prompt``/``completion`` split form (same contract as the
    standalone SFT trainer's dataset)."""
    if "messages" in example:
        messages = normalize_messages(example["messages"], default_role="assistant")
    elif "prompt" in example and "completion" in example:
        messages = normalize_messages(example["prompt"], default_role="user") + normalize_messages(
            example["completion"], default_role="assistant"
        )
    else:
        raise ValueError(
            "All examples in the dataset must have either a 'messages' column or both 'prompt' and 'completion' columns"
        )
    messages = deserialize_tool_calls(messages)
    return strip_message_content(messages)


def resolve_tools(example: dict) -> list[dict]:
    """Parse the row's ``tools`` / ``tool_defs`` column (JSON string or list of
    dicts) into OAI function-calling shape for the renderer."""
    raw_tools = example.get("tools", example.get("tool_defs"))
    if not raw_tools:
        return []
    if isinstance(raw_tools, str):
        raw_tools = json.loads(raw_tools)
    return [
        t
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t
        else {
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description"),
                "parameters": t.get("parameters"),
                **({} if t.get("strict") is None else {"strict": t["strict"]}),
            },
        }
        for t in raw_tools
    ]


def _to_vf_message(message: dict) -> vf.Message:
    """An OAI-shaped message dict -> the typed v1 message a ``MessageNode``
    carries. Tool calls flatten from the nested OAI ``function`` shape to the
    v1 ``ToolCall`` (``arguments`` as a raw JSON string); unknown keys drop."""
    role = message.get("role")
    if role not in _MESSAGE_KEYS:
        raise ValueError(f"Invalid message role: {role}")
    slim = {k: message[k] for k in _MESSAGE_KEYS[role] if message.get(k) is not None}
    if slim.get("tool_calls"):
        tool_calls = []
        for i, tc in enumerate(slim["tool_calls"]):
            # ``deserialize_tool_calls`` parsed arguments to a dict for the
            # renderer; the typed ToolCall carries the raw JSON string.
            arguments = tc["function"].get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            tool_calls.append(
                {"id": tc.get("id") or f"call_{i}", "name": tc["function"]["name"], "arguments": arguments}
            )
        slim["tool_calls"] = tool_calls
    return _MESSAGE_ADAPTER.validate_python(slim)


def render_to_nodes(
    renderer: Renderer,
    messages: list[dict],
    tools: list[dict],
    loss_mask_config: LossMaskConfig,
) -> list[vf.MessageNode]:
    """Render a conversation once and slice it into per-message graph nodes.

    ``message_token_spans`` attributes each message's contiguous token block;
    the scaffold between blocks (``message_indices == -1``) attaches to the
    *following* node, matching the live renderer-client convention ("a node's
    leading template scaffold + its own tokens"). Trailing scaffold lands on
    the last node. The loss mask is the role mask ANDed with the renderer's
    ``sampled_mask`` — the same semantics as ``renderers.build_training_sample``.
    """
    rendered = renderer.render(messages, tools=tools or None)
    n = len(rendered.token_ids)
    has_sampled_info = len(rendered.sampled_mask) == n

    def role_trains(message: dict) -> bool:
        role = message.get("role")
        trains = getattr(loss_mask_config, role, None) if isinstance(role, str) else None
        if trains is None:
            raise ValueError(f"Invalid message role: {role}")
        return trains

    loss_mask: list[bool] = []
    for k, msg_idx in enumerate(rendered.message_indices):
        if msg_idx < 0:
            loss_mask.append(False)
        elif has_sampled_info and not rendered.sampled_mask[k]:
            loss_mask.append(False)
        else:
            loss_mask.append(role_trains(messages[msg_idx]))

    spans = rendered.message_token_spans()
    nodes: list[vf.MessageNode] = []
    prev_end = 0
    last_assistant_idx = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    for i, message in enumerate(messages):
        span = spans[i] if i < len(spans) else None
        # A token-less message still gets an (empty) node so the recorded
        # conversation stays complete.
        end = span[1] if span is not None else prev_end
        if i == len(messages) - 1:
            end = n  # trailing scaffold (e.g. eos) attaches to the last node
        node_tokens = rendered.token_ids[prev_end:end]
        node_mask = loss_mask[prev_end:end]
        prev_end = end
        is_assistant = message.get("role") == "assistant"
        nodes.append(
            vf.MessageNode(
                parent=i - 1 if i > 0 else None,
                message=_to_vf_message(message),
                sampled=is_assistant,
                token_ids=node_tokens,
                mask=node_mask,
                logprobs=[0.0] * sum(node_mask),
                finish_reason="stop" if i == last_assistant_idx else None,
            )
        )
    return nodes


class StaticSFTEnv:
    """A train env whose "taskset" is a HF SFT dataset. Duck-type compatible
    with ``TrainEnv``; spawns no server, calls no model, never errors a
    healthy row."""

    def __init__(
        self,
        config: TrainEnvConfig,
        algorithm: Algorithm,
        tokenizer: PreTrainedTokenizer,
        renderer_config,
    ):
        assert config.dataset is not None
        self.config = config
        self.dataset_config: StaticDatasetConfig = config.dataset
        self.algorithm = algorithm
        self.sampler = NullSampler()
        # The sink fans the env temperature out per token; context tokens are
        # masked and ce ignores it, so any constant works.
        self.sampling_args: dict = {"temperature": 1.0}
        self.num_tasks: int = 0
        self.requires_group_scoring: bool = False
        self.renderer: Renderer = create_renderer(tokenizer, renderer_config)
        self._dataset = None
        self._env_server_process = None  # Envs.shutdown compatibility

    @property
    def name(self) -> str:
        return self.config.resolved_name

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        """Load (and interleave) the HF dataset; no server to spawn."""
        from prime_rl.trainer.sft.data import load_sft_dataset

        dataset = await asyncio.to_thread(load_sft_dataset, self.dataset_config)
        if self.dataset_config.max_examples is not None:
            dataset = dataset.take(min(self.dataset_config.max_examples, len(dataset)))
        self._dataset = dataset
        self.num_tasks = len(dataset)
        get_logger().info(
            f"Env {self.name} ready: static dataset {self.dataset_config.name} with {self.num_tasks} examples"
        )

    async def run_rollout(self, client: Any, task_idx: int, model_name: str, cache_salt: str | None) -> Rollout:
        """Render row ``task_idx`` into a completed Rollout. ``client`` /
        ``model_name`` / ``cache_salt`` are part of the env interface but
        meaningless here — nothing is sampled."""
        assert self._dataset is not None, f"Env {self.name} not started — call start() first."
        example = dict(self._dataset[task_idx % self.num_tasks])
        rollout = STATIC_ROLLOUT_TYPE(
            task=vf.WireTask(idx=task_idx, prompt=None),
            is_completed=True,
            stop_condition="agent_completed",
        )
        try:
            messages = resolve_messages(example)
            tools = resolve_tools(example)
            nodes = await asyncio.to_thread(
                render_to_nodes, self.renderer, messages, tools, self.dataset_config.loss_mask
            )
        except Exception as exc:
            rollout.capture_error(exc)
            rollout.stop_condition = "error"
            return rollout

        num_tokens = sum(len(node.token_ids) for node in nodes)
        max_length = self.dataset_config.max_length
        if max_length is not None and num_tokens > max_length:
            rollout.errors.append(
                vf.Error(type="Oversized", message=f"Example renders to {num_tokens} tokens > max_length {max_length}")
            )
            rollout.stop_condition = "error"
            return rollout
        if not any(any(node.mask) for node in nodes):
            rollout.errors.append(
                vf.Error(type="NoTrainableTokens", message="No tokens pass the loss mask for this example")
            )
            rollout.stop_condition = "error"
            return rollout

        rollout.nodes = nodes
        return rollout

    def shutdown(self) -> None:
        pass
