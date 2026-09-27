"""The data that flows through the orchestrator.

The dispatcher emits ``vf.Episode`` and nothing else: the env's, or one it
synthesizes for an attempt that returned none (a failed request, a cancelled
attempt). A sink completes them into a ``Group`` — one task's rollouts — and the
queue cuts groups into a ``Batch`` for the trainer; the evaluator collects them into
an ``Epoch`` per env and step.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

import verifiers.v1 as vf

from prime_rl.transports.batch import TrainingSample

Kind = Literal["train", "eval"]

CANCELLED = "Cancelled"
"""Error type of an episode the pipeline voided (stale, overload, superseded, drain)."""


@dataclass
class Progress:
    """Persistent counters; ``step`` is the trainer-aligned step (1-indexed)."""

    step: int = 1
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


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

    kind: Kind
    env_name: str
    group_id: uuid.UUID
    task: vf.Task
    policy_version: int
    step: int
    started_at: float = 0.0
    """``time.monotonic()`` at dispatch; feeds episode-duration estimates."""
    dispatch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Names the episode on the live view from dispatch until its first trace streams."""
    live: dict[str, LiveTrace] = field(default_factory=dict)
    """The episode's in-flight traces by id, as far as the env server's stream has
    told: which phase each is in and how many turns it has committed."""


# ── episode provenance ─────────────────────────────────────────────────────


def env_of(episode: vf.Episode) -> str:
    if episode.env.name is None:
        raise ValueError("Orchestrated episode is missing its environment name")
    return episode.env.name


def group_of(episode: vf.Episode) -> str:
    if episode.group is None:
        raise ValueError("Orchestrated episode is missing its rollout group")
    return episode.group.id


def work_of(episode: vf.Episode) -> vf.TrainWorkInfo | vf.EvalWorkInfo:
    if not isinstance(episode.run, vf.TrainRunInfo):
        raise ValueError("Orchestrated episode is missing its run provenance")
    return episode.run.work


def cancel(episode: vf.Episode, reason: str) -> None:
    episode.ok = False
    episode.errors.append(vf.Error(type=CANCELLED, message=reason))


def is_cancelled(episode: vf.Episode) -> bool:
    return any(error.type == CANCELLED for error in episode.errors)


def cancel_reason(episode: vf.Episode) -> str | None:
    return next((error.message for error in episode.errors if error.type == CANCELLED), None)


def has_error(episode: vf.Episode) -> bool:
    """Failed on its own, as opposed to voided by the pipeline."""
    return not is_cancelled(episode) and (not episode.ok or any(trace.has_error for trace in episode.traces))


def staleness(episode: vf.Episode, step: int) -> int:
    """Versions between the policy that generated the episode and the one batch
    ``step`` trains on (v{step-1}). Frozen-sourced episodes (no span) are never stale."""
    policy = work_of(episode).policy
    return max(0, (step - 1) - policy.start) if policy is not None else 0


# ── groups, batches, epochs ────────────────────────────────────────────────


@dataclass
class Group:
    """One task's rollouts, complete: exactly the episodes the dispatcher owed for it."""

    env: str
    id: str
    step: int
    """The batch window (train) or eval step the group was dispatched for."""
    episodes: list[vf.Episode]
    admitted: bool = True
    """The curriculum's verdict; a group that never reached it is not admitted."""
    samples: dict[str, list[TrainingSample]] = field(default_factory=dict)
    """Trainer payload by trace id: only the traces that train. Empty when none does."""

    @property
    def traces(self) -> list[vf.Trace]:
        return [trace for episode in self.episodes for trace in episode.traces]

    @property
    def version(self) -> int | None:
        """The policy version the group was dispatched at; None when frozen-sourced."""
        spans = [span for episode in self.episodes if (span := work_of(episode).policy) is not None]
        return min(span.start for span in spans) if spans else None

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def split(self, n: int) -> tuple[Group, Group]:
        """The group's first ``n`` sampled traces and the rest, each with the episodes
        (narrowed to the traces they keep) that own them."""
        head_ids = set(list(self.samples)[:n])

        def narrow(keep: set[str], samples: dict[str, list[TrainingSample]]) -> Group:
            episodes = []
            for episode in self.episodes:
                traces = [trace for trace in episode.traces if trace.id in keep]
                if traces:
                    episodes.append(episode.model_copy(update={"traces": traces}))
            return Group(self.env, self.id, self.step, episodes, self.admitted, samples)

        head = narrow(head_ids, {trace_id: self.samples[trace_id] for trace_id in head_ids})
        tail_ids = {trace.id for trace in self.traces} - head_ids
        tail = narrow(
            tail_ids, {trace_id: samples for trace_id, samples in self.samples.items() if trace_id in tail_ids}
        )
        return head, tail


@dataclass
class Batch:
    """Everything that finished since the last cut, whether it ships or not."""

    step: int
    groups: list[Group]

    @property
    def samples(self) -> list[TrainingSample]:
        return [sample for group in self.groups for samples in group.samples.values() for sample in samples]


@dataclass
class Epoch:
    """One env's eval epoch."""

    env: str
    step: int
    groups: list[Group]
