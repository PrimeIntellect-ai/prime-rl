"""What the orchestrator knows about a finished rollout.

An env worker keeps a rollout whole: its message graph, its tokens and its training samples
(`prime_rl.transports.payload`). The orchestrator gets an `EpisodeHead` instead: the episode's
identity and standing, and per trace the scalars that scoring, admission, staleness and metrics
read. A head is a few kilobytes whatever the rollout's length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import verifiers.v1 as vf
from pydantic import BaseModel, Field, PrivateAttr
from verifiers.v1.trace import Error, RunInfo, Timing
from verifiers.v1.types import Usage

from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY, UNLABELED

MAX_ERROR_CHARS = 4000
"""An error's message and traceback keep their last this many characters."""


class AgentHead(BaseModel):
    name: str = "agent"
    trainable: bool = True


class TaskHead(BaseModel):
    type: str
    key: str | None = None
    hash: str | None = None
    idx: int | None = None


class LengthTokens(BaseModel):
    """Model-generated and new input tokens on the trace's trainable branches, counted once
    per call, split by whether a sub-agent made the call (GRPO's length penalty)."""

    output: float = 0.0
    input: float = 0.0
    subagent_output: float = 0.0
    subagent_input: float = 0.0


class TraceHead(BaseModel):
    """One trace's identity, standing and summary, without its message graph."""

    id: str
    task: TaskHead
    agent: AgentHead = Field(default_factory=AgentHead)
    rewards: dict[str, vf.Reward | None] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)
    """prime-rl's entries of the trace's info: the references to its training samples, and the
    credit the orchestrator assigns."""
    ok: bool = False
    is_completed: bool = False
    is_timeout: bool = False
    stop_condition: str | None = None
    errors: list[Error] = Field(default_factory=list)
    dropped: str | None = None
    timing: Timing = Field(default_factory=Timing)
    usage: Usage | None = None
    reward: float = 0.0
    num_turns: int = 0
    num_branches: int = 0
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    num_total_tokens: int = 0
    is_truncated: bool = False
    stopped_early: bool = False
    """Whether the trace is truncated or the env skipped its task scoring: its reward, if any,
    is the fixed reward the env sets for that stop."""
    has_sampled_tokens: bool = False
    """Whether any node has trainable (model-sampled) tokens."""
    trainable_tokens: int = 0
    """Sampled tokens on the trace's trainable branches, each node counted once."""
    trainable_turns: int = 0
    """Sampled nodes on the trace's trainable branches."""
    trainable_branches: list[int] = Field(default_factory=list)
    """Indices of the trace's trainable branches (`Trace.branches`)."""
    token_layout: dict[int, list[int]] = Field(default_factory=dict)
    """The trainable tokens of each training sample, by branch index: one count per sampled
    node that trains in it, in order (`prime_rl.orchestrator.algo.token_layout`)."""
    branch_advantages: dict[int, Any] = Field(default_factory=dict, exclude=True)
    """The orchestrator's advantages of a trace credited per token, by branch index, one
    float32 per trainable token (`assign_advantages`). Never serialized."""
    length_tokens: LengthTokens = Field(default_factory=LengthTokens)
    versions: dict[int, int] = Field(default_factory=dict)
    """Sampled tokens per policy version that sampled them (``UNLABELED``: the engine reported none)."""

    @property
    def has_error(self) -> bool:
        return not self.ok

    @property
    def is_dropped(self) -> bool:
        return self.dropped is not None

    @property
    def last_error(self) -> Error | None:
        return self.errors[-1] if self.errors else None


@dataclass
class EpisodeRecord:
    """The episode as the trace monitors persist it: its JSON record (``Episode.to_json``) and
    the index row's source (``index_record``)."""

    json: bytes
    index: dict[str, Any]


class EpisodeHead(BaseModel):
    """One episode's identity and standing, and its traces' heads."""

    id: str
    env: vf.EnvInfo = Field(default_factory=vf.EnvInfo)
    task: TaskHead
    group: vf.GroupInfo | None = None
    run: RunInfo | None = None
    ok: bool = False
    errors: list[Error] = Field(default_factory=list)
    traces: list[TraceHead] = Field(default_factory=list)
    _record: EpisodeRecord | None = PrivateAttr(default=None)

    def record_run(self, run: RunInfo) -> None:
        self.run = run

    @property
    def last_error(self) -> Error | None:
        return self.errors[-1] if self.errors else None

    @property
    def num_total_tokens(self) -> int:
        return sum(trace.num_total_tokens for trace in self.traces)

    def take_record(self) -> EpisodeRecord | None:
        """The episode's record, which the head then no longer holds."""
        record, self._record = self._record, None
        return record

    def to_checkpoint(self) -> dict[str, Any]:
        """The head as a rollout checkpoint holds it: with the record it still holds and its
        traces' token advantages, which its serialized form leaves out."""
        return {
            "head": self.model_dump(mode="python"),
            "record": self._record,
            "advantages": [trace.branch_advantages for trace in self.traces],
        }

    @classmethod
    def from_checkpoint(cls, dumped: dict[str, Any]) -> EpisodeHead:
        head = cls.model_validate(dumped["head"])
        head._record = dumped["record"]
        for trace, advantages in zip(head.traces, dumped["advantages"], strict=True):
            trace.branch_advantages = advantages
        return head


def clip_error(error: Error) -> Error:
    def clip(text: str | None) -> str | None:
        return text[-MAX_ERROR_CHARS:] if text is not None else None

    return error.model_copy(update={"message": clip(error.message), "traceback": clip(error.traceback)})


def task_head(task: vf.TraceTask) -> TaskHead:
    return TaskHead(type=task.type, key=task.key, hash=task.hash, idx=getattr(task.data, "idx", None))


def _versions(trace: vf.Trace) -> dict[int, int]:
    """Sampled tokens per policy version, from the engine's ``weight_versions`` labels:
    a node without labels has all its sampled tokens ``UNLABELED``."""
    counts: dict[int, int] = {}
    for node in trace.nodes:
        if node.weight_versions:
            for span in node.weight_versions:
                version = int(span.version) if span.version.isdigit() else UNLABELED
                counts[version] = counts.get(version, 0) + span.end - span.start
        elif sampled := int(np.count_nonzero(node.mask)):
            counts[UNLABELED] = counts.get(UNLABELED, 0) + sampled
    return counts


def trace_head(trace: vf.Trace) -> TraceHead:
    from prime_rl.orchestrator.algo.grpo import length_penalty_tokens
    from prime_rl.orchestrator.algo.routing import trainable_nodes
    from prime_rl.orchestrator.trajectories import trainable_token_layout

    trainable = trainable_nodes(trace)
    on_trainable_branch = {id(node): node for branch in trace.branches if branch.trainable for node in branch.nodes}
    output, input, subagent_output, subagent_input = length_penalty_tokens(trace)
    info = {PAYLOAD_INFO_KEY: trace.info[PAYLOAD_INFO_KEY]} if PAYLOAD_INFO_KEY in trace.info else {}
    return TraceHead(
        id=trace.id,
        task=task_head(trace.task),
        agent=AgentHead(name=trace.agent.name, trainable=trace.agent.trainable),
        rewards=trace.rewards,
        metrics=trace.metrics,
        info=info,
        ok=trace.ok,
        is_completed=trace.is_completed,
        is_timeout=trace.is_timeout,
        stop_condition=trace.stop_condition,
        errors=[clip_error(error) for error in trace.errors],
        dropped=trace.dropped,
        timing=trace.timing,
        usage=trace.usage,
        reward=trace.reward,
        num_turns=trace.num_turns,
        num_branches=trace.num_branches,
        num_input_tokens=trace.num_input_tokens,
        num_output_tokens=trace.num_output_tokens,
        num_total_tokens=trace.num_total_tokens,
        is_truncated=trace.is_truncated,
        stopped_early=trace.is_truncated or "scoring_skipped" in trace.info,
        has_sampled_tokens=any(np.any(node.mask) for node in trace.nodes),
        trainable_tokens=sum(int(np.count_nonzero(node.mask)) for node in trainable),
        trainable_turns=sum(node.sampled for node in on_trainable_branch.values()),
        trainable_branches=[branch.index for branch in trace.branches if branch.trainable],
        token_layout=trainable_token_layout(trace),
        length_tokens=LengthTokens(
            output=output, input=input, subagent_output=subagent_output, subagent_input=subagent_input
        ),
        versions=_versions(trace),
    )


def episode_head(episode: vf.Episode) -> EpisodeHead:
    """The head of a finished episode."""
    return EpisodeHead(
        id=episode.id,
        env=episode.env,
        task=task_head(episode.task),
        group=episode.group,
        run=episode.run,
        ok=episode.ok,
        errors=[clip_error(error) for error in episode.errors],
        traces=[trace_head(trace) for trace in episode.traces],
    )


def recorded_head(episode: vf.Episode, float_decimals: int | None = None) -> EpisodeHead:
    """The head of an episode held whole, holding the episode's record."""
    from prime_rl.monitors.file.traces.index import index_record

    head = episode_head(episode)
    head._record = EpisodeRecord(json=episode.to_json(float_decimals=float_decimals), index=index_record(episode))
    return head
