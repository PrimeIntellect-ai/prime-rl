"""Shared dataclasses for the orchestrator. Data carriers only; no behavior."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import verifiers.v1 as vf

from prime_rl.transport import TrainingSample

if TYPE_CHECKING:
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


RunKind = Literal["train", "eval"]


@dataclass
class InflightEpisode:
    """Scheduling state for one in-flight environment run."""

    kind: RunKind
    env_name: str
    group_id: uuid.UUID
    task: vf.Task
    policy_version: int
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0
    eval_step: int | None = None


@dataclass
class GroupState:
    """Per-group dispatcher state: what's left to schedule + the pinned
    client (for prefix-cache hits)."""

    kind: RunKind
    env_name: str
    task: vf.Task
    """The group's task — its data is shipped on every dispatch."""
    episodes_to_schedule: int
    target_episodes: int
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


@dataclass
class RunContext:
    """Orchestrator-owned identity for one environment run."""

    kind: RunKind
    env_name: str
    group_id: uuid.UUID
    task: vf.Task
    policy_version: int
    off_policy_steps: int = 0
    eval_step: int | None = None


@dataclass
class TrainingTrace:
    """Trainer-derived state composed with one plain verifiers trace."""

    context: RunContext
    episode: vf.Episode
    trace: vf.Trace
    samples: list[TrainingSample]
    advantages: list[float] | None = None

    def assign_advantages(self, values: float | list[float]) -> None:
        """Write the rl advantage stream: a scalar broadcast over the
        trace's trainable (mask-True) tokens (0.0 elsewhere), or a per-token
        list already aligned full-length to the samples' concatenated
        ``token_ids``. A trace never assigned ships no advantage stream."""
        total = sum(len(sample.token_ids) for sample in self.samples)
        if isinstance(values, (int, float)):
            self.advantages = [
                float(values) if trainable else 0.0 for sample in self.samples for trainable in sample.mask
            ]
            return
        if len(values) != total:
            raise ValueError(
                f"per-token advantages must align with the trace's tokens: "
                f"got {len(values)}, expected {total} (env '{self.context.env_name}')."
            )
        self.advantages = [float(v) for v in values]

    def scalar_advantage(self) -> float | None:
        """Scalar view of the per-token advantage stream for monitoring: the
        mean over assigned (non-zero) positions — exact for the uniform GRPO
        case, 0.0 for a zero-advantage group, None when no credit was assigned."""
        if not self.advantages:
            return None
        nonzero = [a for a in self.advantages if a != 0.0]
        return sum(nonzero) / len(nonzero) if nonzero else 0.0

    @property
    def is_trainable(self) -> bool:
        """Whether the trace carries a training signal — a nonzero advantage on some token. A
        uniform-reward GRPO group (all-zero advantages) or an unscored trace has no gradient."""
        return bool(self.advantages) and any(a != 0.0 for a in self.advantages)


@dataclass
class EpisodeRun:
    """One environment episode plus its orchestrator and training state."""

    context: RunContext
    episode: vf.Episode
    training: list[TrainingTrace] = field(default_factory=list)
    is_admitted: bool = True

    @property
    def traces(self) -> list[vf.Trace]:
        return self.episode.traces


@dataclass
class TrainBatch:
    """``episodes`` is the observation window since the last ship — every episode of every group
    finalized in that span (errored + rejected included; episodes of still-incomplete groups wait
    for a later window). Its ``.effective`` / ``.metrics`` views drive logging. ``samples`` is the
    trainer-bound payload from the admitted cohort — an empty list means nothing
    ships, which would stall the trainer."""

    episodes: TrainEpisodes
    samples: list[TrainingSample]


@dataclass
class EvalBatch:
    """One env's eval epoch. ``episodes`` is the full returned cohort (errored included); its
    ``.effective`` / ``.metrics`` views drive logging."""

    env_name: str
    step: int
    episodes: EvalEpisodes


class VersionObserver(Protocol):
    """Notified around each policy update; walked by the watcher.

    ``on_version_pending`` fires *before* the inference engines are paused for
    the weight update; ``on_new_version`` fires *after* the new weights are live
    and ``Policy`` has been mutated."""

    async def on_version_pending(self, step: int) -> None: ...

    async def on_new_version(self, step: int) -> None: ...
