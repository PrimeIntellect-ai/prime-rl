"""Shared orchestrator data carriers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
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

CancelReason = Literal["stale", "overload", "superseded", "requested"]
"""Why the dispatcher dropped a group: a train group past ``max_off_policy_steps``, an
overload cut, an eval epoch superseded by a newer checkpoint, or an eval task its
streaming taskset asked to cancel (``Taskset.cancel``)."""


@dataclass
class GroupCancellation:
    """Terminal marker for a dropped group: one message covering every episode
    the group still owed the sink (in-flight and never-dispatched), so
    count-to-``group_size`` finalization still fires. ``reason`` distinguishes
    pipeline decisions (staleness, overload cut, superseded eval, a taskset's cancel)
    from episode errors."""

    kind: WorkKind
    env_name: str
    group_id: str
    step: int
    count: int
    reason: CancelReason
    task: vf.Task | None = field(default=None, compare=False)
    """The dropped group's task, for the taskset's ``on_cancelled`` hook (set for a
    ``requested`` cancel)."""


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
    task: vf.Task | None = field(default=None, compare=False)
    """The dispatched task, for the taskset's ``on_failed`` hook."""
    exception: BaseException | None = field(default=None, compare=False, repr=False)
    """What the request raised, for the taskset's ``on_failed`` hook."""


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
    seat: str | None = None
    """The env agent running the trace (``Trace.agent.name``), once streamed."""
    durable: bool = False
    """Whether the trace carries a durable ref (``vf.durable.refs_of``): a durable call
    run again may look it up, so its endings flush to the monitors at once."""


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


@dataclass
class EvalBatch:
    """One env's eval epoch, or one window of a streaming env's epoch.

    ``episodes`` is the full returned cohort (errored included), while
    ``failures`` accounts for requests that returned no verifier artifact. A
    streaming epoch has no size, so it is cut into windows (``EvalSink``): each
    window is a batch numbered by ``window``, and only the last one is ``final``.
    A bounded epoch is one final batch with no ``window``.
    """

    env_name: str
    step: int
    episodes: EvalEpisodes
    failures: list[DispatchFailure]
    cancelled: int = 0
    window: int | None = None
    final: bool = True


class VersionObserver(Protocol):
    """Notified around each policy update; walked by the watcher.

    ``on_version_pending`` fires *before* the inference engines are paused for
    the weight update; ``on_new_version`` fires *after* the new weights are live
    and ``Policy`` has been mutated."""

    async def on_version_pending(self, step: int) -> None: ...

    async def on_new_version(self, step: int) -> None: ...
