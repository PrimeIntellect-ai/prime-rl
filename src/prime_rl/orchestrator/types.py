"""Shared dataclasses for the orchestrator. Data carriers only; no behavior."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Protocol

import verifiers.v1 as vf
from pydantic import ConfigDict, Field
from verifiers.v1.task import TaskT

from prime_rl.transport import TrainingSample

if TYPE_CHECKING:
    from prime_rl.orchestrator.metrics import EvalRollouts, TrainRollouts


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


RolloutKind = Literal["train", "eval"]


@dataclass
class InflightRollout:
    """Per-task scheduling state in the dispatcher; one entry per in-flight
    ``run_rollout`` / ``run_group`` task."""

    kind: RolloutKind
    env_name: str
    group_id: uuid.UUID
    policy_version: int
    rollout_count: int
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0
    eval_step: int | None = None


@dataclass
class GroupState:
    """Per-group dispatcher state: what's left to schedule + the pinned
    client (for prefix-cache hits)."""

    kind: RolloutKind
    env_name: str
    task_idx: int
    rollouts_to_schedule: int
    target_rollouts: int
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


class Rollout(vf.Trace[TaskT], Generic[TaskT]):
    """A completed rollout: the env's typed ``vf.Trace`` *is* the rollout — prime-rl's
    orchestration metadata lives on it directly (set by the dispatcher once the rollout
    returns), so there's no wrapper. Train vs eval is the ``kind`` discriminator. All metadata
    fields are ``exclude=True``, so dumping a Rollout yields a plain trace — the on-disk
    ``results.jsonl`` is unchanged.

    It is also the single currency the scoring hooks receive: a hook reads the trace
    directly (``rollout.reward``, ``rollout.nodes``, ``rollout.num_turns``) and writes
    credit through :meth:`assign_advantages` (scalar broadcast or per-token), which
    spreads over the samples' trainable (mask-True) tokens."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # ``samples`` holds msgspec structs

    kind: RolloutKind = Field(default="train", exclude=True)
    env_name: str = Field(default="", exclude=True)
    group_id: uuid.UUID = Field(default_factory=uuid.uuid4, exclude=True)
    policy_version: int = Field(default=0, exclude=True)
    off_policy_steps: int = Field(default=0, exclude=True)
    samples: list[TrainingSample] = Field(default_factory=list, exclude=True)
    # Per-token rl advantage stream, full-length-N (= len(token_ids)) per
    # sample, concatenated across the rollout's samples in order; 0.0 on
    # non-trainable positions. None = no credit assigned (advantage-based
    # filters skip it; the wire ships no advantage stream).
    advantages: list[float] | None = Field(default=None, exclude=True)
    is_filtered: bool = Field(default=False, exclude=True)
    filter_results: dict[str, bool] = Field(default_factory=dict, exclude=True)
    eval_step: int | None = Field(default=None, exclude=True)

    def assign_advantages(self, values: float | list[float]) -> None:
        """Write the rl advantage stream: a scalar broadcast over the
        rollout's trainable (mask-True) tokens (0.0 elsewhere), or a per-token
        list already aligned full-length to the samples' concatenated
        ``token_ids``. A rollout never assigned ships no advantage stream."""
        total = sum(len(sample.token_ids) for sample in self.samples)
        if isinstance(values, (int, float)):
            self.advantages = [
                float(values) if trainable else 0.0 for sample in self.samples for trainable in sample.mask
            ]
            return
        if len(values) != total:
            raise ValueError(
                f"per-token advantages must align with the rollout's tokens: "
                f"got {len(values)}, expected {total} (env '{self.env_name}')."
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
        """Whether the rollout carries a training signal — a nonzero advantage on some token. A
        uniform-reward GRPO group (all-zero advantages) or an unscored rollout has no gradient."""
        return bool(self.advantages) and any(a != 0.0 for a in self.advantages)


@dataclass
class TrainBatch:
    """``rollouts`` is the full arrival window since the last ship (errored + filtered included; its
    ``.effective`` / ``.metrics`` views drive logging). ``samples`` is the trainer-bound payload (the
    shipped cohort's post-filter survivors) — an empty list means nothing ships, which would stall the
    trainer. Trainable counts derive from ``rollouts`` (``r.is_trainable``) and token totals from
    ``samples``, so neither is carried as a field."""

    rollouts: TrainRollouts
    samples: list[TrainingSample]


@dataclass
class EvalBatch:
    """One env's eval epoch. ``rollouts`` is the full returned cohort (errored included); its
    ``.effective`` / ``.metrics`` views drive logging."""

    env_name: str
    step: int
    rollouts: EvalRollouts


class VersionObserver(Protocol):
    """Notified around each policy update; walked by the watcher.

    ``on_version_pending`` fires *before* the inference engines are paused for
    the weight update; ``on_new_version`` fires *after* the new weights are live
    and ``Policy`` has been mutated."""

    async def on_version_pending(self, step: int) -> None: ...

    async def on_new_version(self, step: int) -> None: ...
