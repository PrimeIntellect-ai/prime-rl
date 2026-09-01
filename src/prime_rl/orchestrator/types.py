"""Shared orchestrator data carriers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from prime_rl.transports.batch import TrainingSample

if TYPE_CHECKING:
    import verifiers.v1 as vf

    from prime_rl.orchestrator.metrics import EvalEpisodes, TrainEpisodes


@dataclass
class Policy:
    """Mutable shared view of the policy. Passed by reference so observers
    see new versions immediately."""

    version: int = 0
    model_name: str = ""


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
    source_index: int | None = None


@dataclass
class InflightEpisode:
    """Scheduling state for one in-flight environment run."""

    kind: WorkKind
    env_name: str
    group_id: uuid.UUID
    task: vf.Task
    policy_version: int
    step: int
    source_index: int | None = None
    client_config: vf.ClientConfig | None = None
    started_at: float = 0.0
    """``time.monotonic()`` at dispatch; feeds episode-duration estimates."""


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
    source_index: int | None = None


@dataclass
class TrainBatch:
    """Returned episodes, dispatch failures, shipped cohort, and trainer payload."""

    episodes: TrainEpisodes
    cohort: TrainEpisodes
    samples: list[TrainingSample]
    failures: list[DispatchFailure]
    # Group cancellations can account for attempts that returned no episode.
    cancelled_attempts: int = 0
    # Stale attempts are a subset of cancelled_attempts.
    stale_attempts: int = 0


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


class VersionObserver(Protocol):
    """Notified around each policy update; walked by the watcher.

    ``on_version_pending`` fires *before* the inference engines are paused for
    the weight update; ``on_new_version`` fires *after* the new weights are live
    and ``Policy`` has been mutated."""

    async def on_version_pending(self, step: int) -> None: ...

    async def on_new_version(self, step: int) -> None: ...
