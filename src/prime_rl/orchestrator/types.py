"""Shared dataclasses for the orchestrator. Data carriers only; no behavior."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, Protocol, cast

import verifiers.v1 as vf
from pydantic import ConfigDict, Field
from verifiers.v1.task import DataT

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
class GroupState:
    """Per-group dispatcher state: what's left to schedule + the pinned
    client (for prefix-cache hits)."""

    kind: RolloutKind
    env_name: str
    task_idx: int
    rollouts_to_schedule: int
    target_rollouts: int
    task: vf.Task | None = None
    """The group's task (v1 envs — its data is shipped on every dispatch). ``None`` for
    legacy envs, which are addressed by ``task_idx`` alone."""
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


class Rollout(vf.Trace[DataT], Generic[DataT]):
    """A completed rollout: the env's typed ``vf.Trace`` *is* the rollout, carrying only the links
    prime-rl needs to place a loose trace back among its peers — its episode, its comparison group,
    its env. Everything about the dispatch itself lives on the ``Episode``, which is the thing that
    was dispatched. All added fields are ``exclude=True``, so dumping a Rollout yields a plain
    trace on the wire; ``vf.Trace.record_run`` mirrors them into ``info`` on arrival so the on-disk
    records stay fully placeable.

    It is also the currency the scoring hooks receive: a hook reads the trace directly
    (``rollout.reward``, ``rollout.nodes``, ``rollout.num_turns``)."""

    env_name: str = Field(default="", exclude=True)
    group_id: uuid.UUID = Field(default_factory=uuid.uuid4, exclude=True)
    # Links the traces of one episode; stamped into ``info`` on arrival so
    # saved records keep their grouping.
    episode_id: str = Field(default="", exclude=True)


class TrainRollout(Rollout[DataT], Generic[DataT]):
    """A rollout on the training path, which alone carries training state: the trainer-bound
    samples built from its branches, the credit assigned over them, and the filter verdicts. Eval
    rollouts have none of this, so they are plain ``Rollout``\\ s and can't be asked for it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # ``samples`` holds msgspec structs

    samples: list[TrainingSample] = Field(default_factory=list, exclude=True)
    # Per-token rl advantage stream, full-length-N (= len(token_ids)) per
    # sample, concatenated across the rollout's samples in order; 0.0 on
    # non-trainable positions. None = no credit assigned (advantage-based
    # filters skip it; the wire ships no advantage stream).
    advantages: list[float] | None = Field(default=None, exclude=True)
    is_filtered: bool = Field(default=False, exclude=True)
    filter_results: dict[str, bool] = Field(default_factory=dict, exclude=True)

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


class Episode(vf.WireEpisode):
    """The env's own ``vf.Episode`` extended with the facts of the dispatch it came from — the only
    thing prime-rl genuinely adds, so the episode itself travels rather than a wrapper around it.
    Those fields are ``exclude=True``, so dumping an Episode yields a plain wire episode.

    An episode that produced no traces is not a special case and needs no stand-in rollout: vf
    already records why on ``errors`` (its ``run_episode`` puts the exception there and returns the
    episode with ``ok`` false), and prime-rl's own outcomes — an off-policy cancel, a task that
    raised before reaching the env — are minted the same way. So ``is_empty`` is simply "no
    traces", and ``last_error`` says why in one vocabulary for every cause.

    Train and eval are the two subclasses rather than a discriminator field, so each carries only
    what its path means: an eval episode has a step it belongs to, a train episode has the policy
    it was generated from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # traces are ``Rollout``s

    KIND: ClassVar[RolloutKind]
    """Which path this episode is on, for the run record and the dispatcher's counters."""

    env_name: str = Field(default="", exclude=True)
    group_id: uuid.UUID = Field(default_factory=uuid.uuid4, exclude=True)
    policy_version: int = Field(default=0, exclude=True)
    """The policy that generated it — the thing being trained on one path, measured on the other."""

    @property
    def rollouts(self) -> list[Rollout]:
        """The episode's traces, typed as the rollouts prime-rl works with."""
        return cast(list[Rollout], self.traces)

    @property
    def is_empty(self) -> bool:
        """Whether nothing came back at all — ``last_error`` then carries the reason. Not the
        same as failing: an episode can error and still have traces (vf keeps the completed subset
        and marks its clean siblings failed), and that failure is accounted for through those
        traces. vf's ``ok`` is the success sentinel; this is only "there is nothing here"."""
        return not self.traces


class TrainEpisode(Episode):
    """An episode collected for training, which alone can go stale relative to the live policy."""

    KIND: ClassVar[RolloutKind] = "train"

    off_policy_steps: int = Field(default=0, exclude=True)
    """How stale it was by the time it shipped — meaningless for eval, which never trains on it."""

    @property
    def rollouts(self) -> list[TrainRollout]:
        return cast(list[TrainRollout], self.traces)


class EvalEpisode(Episode):
    """An episode collected for one eval epoch. ``step`` is the training step whose eval triggered
    it — always known, unlike on the train path, so it is not optional here. It is the source for
    the ``run.step`` each of its traces gets stamped with on arrival."""

    KIND: ClassVar[RolloutKind] = "eval"

    step: int = Field(default=0, exclude=True)


@dataclass
class InflightEpisode:
    """One episode in flight, and the facts of the dispatch that will be stamped onto it when it
    lands. The pair is the whole lifecycle: an ``InflightEpisode`` going out, an ``Episode`` coming
    back — so nothing downstream has to know how a rollout was scheduled."""

    kind: RolloutKind
    env_name: str
    group_id: uuid.UUID
    policy_version: int
    episodes_owed: int
    """How many episodes this dispatch owes the sink — one, except on the legacy group path."""
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0
    eval_step: int | None = None

    def stamp(self, episode: vf.WireEpisode, *, policy_version: int, eval_step: int | None) -> Episode:
        """Mint the landed episode: the env's own, carrying the dispatch it came from. The group's
        values win over this dispatch's when it is still alive, so they are passed in rather than
        read off ``self``."""
        common = {
            **dict(episode),
            "env_name": self.env_name,
            "group_id": self.group_id,
            "policy_version": policy_version,
        }
        if self.kind == "eval":
            assert eval_step is not None, "eval episode missing its step"
            return EvalEpisode.model_construct(**common, step=eval_step)
        return TrainEpisode.model_construct(**common, off_policy_steps=self.off_policy_steps)


@dataclass
class TrainBatch:
    """``rollouts`` is the observation window since the last ship — every rollout of every group
    finalized in that span (errored + filtered included; rollouts of still-incomplete groups wait
    for a later window). Its ``.effective`` / ``.metrics`` views drive logging. ``samples`` is the
    trainer-bound payload (the shipped cohort's post-filter survivors) — an empty list means nothing
    ships, which would stall the trainer. Trainable counts derive from ``rollouts.effective``
    (``r.is_trainable``) and token totals from ``samples``, so neither is carried as a field."""

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
