from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.clients.client import resolve_client
from verifiers.v1.clients.train import tool_to_wire
from verifiers.v1.dialects.chat import ChatDialect, message_to_wire, parse_message
from verifiers.v1.types import content_text

from prime_rl.configs.algorithm import EPDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.transports.batch import TrainingSample
from prime_rl.utils.chat_template import deserialize_tool_calls
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from renderers.base import Renderer
    from verifiers.v1.clients.client import Client

    from prime_rl.orchestrator.clients import InferenceClient

INFO_KEY = "epd"
"""``trace.info`` key holding the teacher's decisions, one per recorded action of the trial."""


def action_starts(messages: vf.Messages) -> list[int]:
    """Where each recorded action begins: an assistant message after a non-assistant one."""
    return [
        i
        for i, message in enumerate(messages)
        if message.role == "assistant" and (i == 0 or messages[i - 1].role != "assistant")
    ]


def pack(messages: vf.Messages, decisions: vf.Messages) -> tuple[vf.Messages, list[bool]]:
    """Interleave a trial with the teacher's decisions, ``o_0, a'_0, a_0, o_1, a'_1, ...``,
    up to the observation after the last decision. Returns the messages and which of
    them are decisions. A trailing recorded action with no observation after it is left out."""
    starts = action_starts(messages)
    packed = list(messages[: starts[0]])
    is_decision = [False] * len(packed)
    for decision, start, end in zip(decisions, starts, [*starts[1:], len(messages)]):
        packed.append(decision)
        is_decision.append(True)
        step = messages[start:end]
        if all(message.role == "assistant" for message in step):
            break
        packed.extend(step)
        is_decision.extend([False] * len(step))
    return packed, is_decision


def _message_text(message: vf.Message) -> str:
    if message.role == "assistant":
        parts = [message.content or ""]
        parts += [f"[call {call.name}] {call.arguments}" for call in message.tool_calls or []]
        return "\n".join(part for part in parts if part)
    return content_text(message.content)


def render_trial(trace: vf.Trace, number: int) -> str:
    """One trial as a transcript, headed by its outcome. System messages are left out:
    they repeat across trials."""
    lines = [f"## Trial {number}: reward {trace.reward:.4g}, ended by {trace.stop_condition}"]
    for message in trace.branches[-1].messages:
        if message.role == "system":
            continue
        name = f" {message.name}" if message.role == "tool" and message.name else ""
        lines.append(f"[{message.role}{name}]\n{_message_text(message)}")
    return "\n\n".join(lines)


def fit_experience(transcripts: list[str], max_chars: int) -> str:
    """The transcripts that fit ``max_chars`` together, in order."""
    kept: list[str] = []
    size = 0
    for transcript in transcripts:
        if size + len(transcript) + 2 > max_chars:
            continue
        kept.append(transcript)
        size += len(transcript) + 2
    if not kept:
        raise ValueError(
            f"no trial fits max_experience_chars={max_chars}; the shortest is {min(map(len, transcripts))} chars"
        )
    if len(kept) < len(transcripts):
        get_logger().warning(
            f"Left {len(transcripts) - len(kept)} of {len(transcripts)} trials out of the experience "
            f"to fit max_experience_chars={max_chars}"
        )
    return "\n\n".join(kept)


class EPDAlgorithm(Algorithm):
    """Experience Distillation (arXiv:2607.21051). A group's rollouts are trials of
    one task. ``score_group`` renders them as the task's experience and has the
    frozen teacher, with that experience in a leading system message, re-decide
    every recorded step of every trial; the decisions are stored on the trace.

    ``trace_to_samples`` then renders the packed sequence ``o_0, a'_0, a_0, o_1, ...``
    without the experience through the policy's renderer, the same path as SFT data,
    with the trial's tools, and trains only on the teacher's decisions. A sample is a
    re-rendering rather than the branch's token sequence, so it carries no branch
    identity: it gets no trainer annotations and never goes stale. No advantage is
    assigned."""

    def __init__(self, config: EPDAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.config = config
        self.sampling = vf.SamplingConfig(temperature=config.temperature, max_tokens=config.max_tokens)
        self.renderer: Renderer | None = None
        self.teacher: Client | None = None

    async def setup(self) -> None:
        from renderers.base import create_renderer, load_tokenizer

        renderer_config = self.clients.train_client.renderer
        self.renderer = create_renderer(load_tokenizer(self.clients.model_name), renderer_config)
        teacher_clients = await self.connect(self.config.teacher, renderer_config=renderer_config)
        self.teacher = resolve_client(teacher_clients.train_client)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        trials = [trace for _, trace in iter_trainable_traces(episodes)]
        transcripts = [render_trial(trace, number) for number, trace in enumerate(trials, 1)]
        experience = self.config.template.replace(
            "{experience}", fit_experience(transcripts, self.config.max_experience_chars)
        )
        await asyncio.gather(*(self.replay(trace, experience) for trace in trials))

    async def replay(self, trace: vf.Trace, experience: str) -> None:
        """Sample the teacher's decision at each recorded action of the trial, in order:
        each decision sees the earlier ones, the recorded actions and their observations."""
        teacher = self.teacher
        assert teacher is not None, "teacher not connected — Algorithm.setup() must run first"
        branch = trace.branches[-1]
        messages = branch.messages
        starts = action_starts(messages)
        decisions: vf.Messages = []
        if starts and starts[0] > 0:
            tools = [tool_to_wire(tool) for tool in branch.tools]
            system = vf.SystemMessage(content=experience)
            for _ in starts:
                context, _ = pack(messages, decisions)
                body = {
                    "model": self.config.teacher.name,
                    "messages": [message_to_wire(message) for message in [system, *context]],
                    **({"tools": tools} if tools else {}),
                }
                response = await teacher.get_response(ChatDialect(), body, self.sampling)
                decisions.append(response.message)
        else:
            get_logger().warning(f"Skipping trace {trace.id}: it has no observation before its first action")
        trace.info[INFO_KEY] = {"decisions": [message_to_wire(decision) for decision in decisions]}

    def trace_to_samples(self, trace: vf.Trace, *, env_name: str) -> list[TrainingSample]:
        from renderers.base import build_training_sample

        renderer = self.renderer
        assert renderer is not None, "renderer not built — Algorithm.setup() must run first"
        decisions = [parse_message(decision) for decision in trace.info[INFO_KEY]["decisions"]]
        if not decisions:
            return []
        branch = trace.branches[-1]
        messages, is_decision = pack(branch.messages, decisions)
        try:
            wire = deserialize_tool_calls([message_to_wire(message) for message in messages])
        except json.JSONDecodeError as e:
            # A teacher decision whose tool-call arguments are not JSON cannot be rendered.
            get_logger().warning(f"Skipping trace {trace.id}: {e}")
            return []
        rendered = build_training_sample(
            renderer,
            wire,
            tools=[tool_to_wire(tool) for tool in branch.tools],
            ensure_final_stop=True,
            message_loss_mask=is_decision,
        )
        if rendered.multi_modal_data is not None:
            raise ValueError("epd does not support multimodal trials")
        num_tokens = len(rendered.token_ids)
        return [
            TrainingSample(
                token_ids=np.asarray(rendered.token_ids, dtype=np.int32),
                mask=np.asarray(rendered.loss_mask, dtype=bool),
                logprobs=np.zeros(num_tokens, dtype=np.float32),
                temperatures=np.zeros(0, dtype=np.float32),  # filled by TrainSink.process_group
                env_name=env_name,
            )
        ]
