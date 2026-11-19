"""Completed evals keep the inputs to reporting, independent of transcript size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import verifiers.v1 as vf
    from verifiers.v1.trace import Reward, Timing


@dataclass(slots=True)
class AgentSummary:
    name: str
    trainable: bool


@dataclass(slots=True)
class ErrorSummary:
    type: str


@dataclass(slots=True)
class TraceSummary:
    id: str
    agent: AgentSummary
    reward: float
    rewards: dict[str, Reward | None]
    metrics: dict[str, float | None]
    timing: Timing
    num_input_tokens: int
    num_output_tokens: int
    num_total_tokens: int
    num_turns: int
    num_branches: int
    has_error: bool
    last_error: ErrorSummary | None
    dropped: str | None
    is_truncated: bool
    is_timeout: bool
    is_completed: bool
    stop_condition: str | None

    @property
    def is_dropped(self) -> bool:
        return self.dropped is not None

    @classmethod
    def from_trace(cls, trace: vf.Trace) -> TraceSummary:
        return cls(
            id=trace.id,
            agent=AgentSummary(trace.agent.name, trace.agent.trainable),
            reward=trace.reward,
            rewards=trace.rewards,
            metrics=trace.metrics,
            timing=trace.timing,
            num_input_tokens=trace.num_input_tokens,
            num_output_tokens=trace.num_output_tokens,
            num_total_tokens=trace.num_total_tokens,
            num_turns=trace.num_turns,
            num_branches=trace.num_branches,
            has_error=trace.has_error,
            last_error=ErrorSummary(trace.last_error.type) if trace.last_error is not None else None,
            dropped=trace.dropped,
            is_truncated=trace.is_truncated,
            is_timeout=trace.is_timeout,
            is_completed=trace.is_completed,
            stop_condition=trace.stop_condition,
        )


@dataclass(slots=True)
class EpisodeSummary:
    id: str
    ok: bool
    group: vf.GroupInfo | None
    traces: list[TraceSummary]
    policy_version: int | None

    @classmethod
    def from_episode(cls, episode: vf.Episode, policy_version: int | None) -> EpisodeSummary:
        return cls(
            id=episode.id,
            ok=episode.ok,
            group=episode.group,
            traces=[TraceSummary.from_trace(trace) for trace in episode.traces],
            policy_version=policy_version,
        )
