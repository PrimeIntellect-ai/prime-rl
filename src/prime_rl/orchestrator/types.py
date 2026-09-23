"""Shared orchestrator data carriers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from prime_rl.transports.batch import TrainingSample

if TYPE_CHECKING:
    import verifiers.v1 as vf

    from prime_rl.orchestrator.metrics import EvalEpisodes, TrainEpisodes


@dataclass
class Progress:
    """Persistent counters; ``step`` is the trainer-aligned step (1-indexed)."""

    step: int = 1
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


WorkKind = Literal["train", "eval"]

CancelReason = Literal["stale", "overload", "superseded"]


@dataclass
class GroupCancellation:
    """Terminal marker for a dropped group: one message covering every episode
    the group still owed the sink (in-flight and never-dispatched), so
    count-to-``group_size`` finalization still fires. ``reason`` distinguishes
    pipeline decisions (staleness, overload cut, superseded eval) from episode errors."""

    kind: WorkKind
    env_name: str
    group_id: str
    step: int
    count: int
    reason: CancelReason


@dataclass(frozen=True)
class DispatchFailure:
    """An environment request that failed before producing an episode."""

    kind: WorkKind
    env_name: str
    group_id: str
    step: int
    policy_version: int
    task_type: str
    task_key: str
    task_hash: str
    error: vf.Error


if TYPE_CHECKING:
    DispatchResult: TypeAlias = vf.WireEpisode | DispatchFailure | GroupCancellation
else:
    DispatchResult: TypeAlias = Any


@dataclass(frozen=True)
class TaskRequest:
    """A task selected by a train or eval source with its pinned run step."""

    env_name: str
    task: vf.Task
    step: int
    rollouts: int | None = None
    """Rollouts of the task this request asks for; None is the env's group size."""
    group_id: str | None = None
    """The group these rollouts join (a resume completing a task's landed group); None mints one."""


@dataclass
class LiveTrace:
    stage: str = "pending"
    turns: int = 0


@dataclass
class InflightEpisode:
    """Scheduling state for one in-flight environment run."""

    kind: WorkKind
    env_name: str
    group_id: uuid.UUID
    task: vf.Task
    policy_version: int
    step: int
    client_config: vf.ClientConfig | None = None
    started_at: float = 0.0
    """``time.monotonic()`` at dispatch; feeds episode-duration estimates."""
    dispatch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Names the episode on the live view from dispatch until its first trace streams."""
    live: dict[str, LiveTrace] = field(default_factory=dict)
    """The episode's in-flight traces by id, as far as the env server's stream has
    told: which phase each is in and how many turns it has committed."""


@dataclass
class GroupState:
    """Per-group dispatcher state with its pinned run step."""

    kind: WorkKind
    env_name: str
    task: vf.Task
    """The group's task — its data is shipped on every dispatch."""
    step: int
    episodes_to_schedule: int
    target_episodes: int
    emitted: int = 0
    policy_version_at_start: int = 0
    group_id: uuid.UUID | None = None


@dataclass
class FinalizedGroup:
    """One train group the sink has finished with: its returned episodes, the
    trainer payload compiled from the traces that may train (empty when the
    group trains nothing), and the attempts that never returned an episode."""

    env_name: str
    episodes: list[vf.Episode]
    samples: dict[str, list[TrainingSample]]
    """Compiled payload by trace id; only traces that survived scoring and admission."""
    survivors: list[vf.Trace]
    """Traces the algorithm let through, before compilation dropped any."""
    failures: list[DispatchFailure]
    cancellation: GroupCancellation | None
    admitted: bool
    """Whether the curriculum admitted the group; a stale drop is never admitted."""

    @property
    def stale(self) -> bool:
        return self.cancellation is not None and self.cancellation.reason == "stale"

    @property
    def owed(self) -> int:
        """The group's full episode budget: arrived, failed and cancelled."""
        cancelled = self.cancellation.count if self.cancellation is not None else 0
        return len(self.episodes) + len(self.failures) + cancelled


@dataclass
class TrainBatch:
    """Returned episodes, dispatch failures, shipped cohort, and trainer payload."""

    episodes: TrainEpisodes
    cohort: TrainEpisodes
    samples: list[TrainingSample]
    failures: list[DispatchFailure]
    # Episodes with traces retained for a later batch are not discarded.
    buffered_episode_ids: set[str]
    # Group cancellations can account for attempts that returned no episode.
    cancelled_attempts: int = 0
    # Stale attempts are a subset of cancelled_attempts.
    stale_attempts: int = 0
    # Queued traces the staleness sweep voided since the last cut.
    stale_drops: int = 0


@dataclass
class EvalBatch:
    """One env's eval epoch.

    ``episodes`` is the full returned cohort (errored included), while
    ``failures`` accounts for requests that returned no verifier artifact.
    """

    env_name: str
    step: int
    episodes: EvalEpisodes
    failures: list[DispatchFailure]
    cancelled: int = 0
