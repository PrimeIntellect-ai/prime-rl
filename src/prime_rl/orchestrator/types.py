"""Shared dataclasses for the orchestrator. Data carriers only; no behavior."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, Self, cast

import verifiers.v1 as vf
from pydantic import ConfigDict, Field
from verifiers.v1.task import DataT
from verifiers.v1.trace import EXCLUDE_FIELDS

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
    a consumer that works in loose traces — the sample monitors — needs to place one back among its
    peers. Anything episode-scoped is read off the ``Episode``, which is the atomic unit everything
    else passes around. All added fields are ``exclude=True``, so dumping a Rollout yields a plain
    trace on the wire; ``vf.Trace.record_run`` mirrors them into ``info`` on arrival so the on-disk
    records stay fully placeable.

    It is also the currency the scoring hooks receive: a hook reads the trace directly
    (``rollout.reward``, ``rollout.nodes``, ``rollout.num_turns``)."""

    env_name: str = Field(default="", exclude=True)
    # Links the traces of one episode; stamped into ``info`` on arrival so
    # saved records keep their grouping.
    episode_id: str = Field(default="", exclude=True)


class TrainRollout(Rollout[DataT], Generic[DataT]):
    """A rollout on the training path, which alone carries training state: the trainer-bound
    samples built from its branches, the credit assigned over them, and the filter verdicts. Eval
    rollouts have none of this, so they are plain ``Rollout``\\ s and can't be asked for it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # ``samples`` holds msgspec structs

    samples: list[TrainingSample] = Field(default_factory=list, exclude=True)
    detections: dict[str, bool] = Field(default_factory=dict, exclude=True)
    """What each degeneracy detector measured on this trace — a measurement, not a verdict."""
    is_filtered: bool = Field(default=False, exclude=True)
    """The sink's verdict: this rollout is not trained on. Kept for the metrics window, which
    reports what came back as well as what shipped."""

    def assign_advantages(self, value: float) -> None:
        """Write ``value`` as the credit for every trainable token, node by node. Credit lives on
        the nodes so branches sharing one cannot disagree about it, and so the trainer reads it
        aligned to the tokens (``Branch.advantages``) rather than re-sliced by offset."""
        for node in self.nodes:
            trainable = sum(node.mask)
            node.advantages = [value] * trainable if trainable else None

    @property
    def advantages(self) -> list[float] | None:
        """Every assigned credit on this trace, or ``None`` if it was never scored — which a
        trace assigned all zeros is not."""
        if all(node.advantages is None for node in self.nodes):
            return None
        return [a for node in self.nodes for a in (node.advantages or [])]

    def scalar_advantage(self) -> float | None:
        """Scalar view of the credit for monitoring: the mean over assigned (non-zero) positions —
        exact for the uniform GRPO case, 0.0 for a zero-advantage group, None when unscored."""
        advantages = self.advantages
        if not advantages:
            return None
        nonzero = [a for a in advantages if a != 0.0]
        return sum(nonzero) / len(nonzero) if nonzero else 0.0

    @property
    def is_trainable(self) -> bool:
        """Whether the rollout carries a training signal — a nonzero advantage on some token. A
        uniform-reward GRPO group (all-zero advantages) or an unscored rollout has no gradient."""
        advantages = self.advantages
        return bool(advantages) and any(a != 0.0 for a in advantages)


class Episode(vf.WireEpisode):
    """The env's own ``vf.Episode`` extended with the facts of the dispatch it came from — the only
    thing prime-rl genuinely adds, so the episode itself travels rather than a wrapper around it.
    Those fields are ``exclude=True``, so dumping an Episode yields a plain wire episode.

    An episode that produced no traces is not a special case and needs no stand-in rollout: vf
    already records why on ``errors`` (its ``run_episode`` puts the exception there and returns the
    episode with ``ok`` false), and prime-rl's own outcomes — an off-policy cancel, a task that
    raised before reaching the env — are minted the same way. So ``is_empty`` is simply "no
    traces", and ``last_error`` says why in one vocabulary for every cause.

    Which path it is on is ``run.type``, vf's own discriminator, stamped by the dispatcher when the
    episode lands — so there is one episode class and no prime-rl-side kind."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # traces are ``Rollout``s

    env_name: str = Field(default="", exclude=True)
    """The env as prime-rl names it (the config key), which is not vf's ``env.id``."""
    group_id: uuid.UUID = Field(default_factory=uuid.uuid4, exclude=True)
    policy_version: int = Field(default=0, exclude=True)
    """The policy that generated it — the thing being trained on one path, measured on the other."""
    off_policy_steps: int = Field(default=0, exclude=True)
    """How stale it was by the time it shipped. Always 0 on the eval path, which never trains."""

    @property
    def rollouts(self) -> list[TrainRollout]:
        """The episode's traces, typed as the rollouts prime-rl works with. Every trace is built as
        a ``TrainRollout`` (``Env.run`` is shared), so the training fields are always reachable —
        on the eval path they simply stay empty."""
        return cast(list[TrainRollout], self.traces)

    def narrow(self, keep: Callable[[Rollout], bool]) -> Self | None:
        """This episode with only the traces that pass ``keep``, or ``None`` if none do. A subset
        stays a list of episodes rather than a flat trace list, so the episode-level aggregates
        keep describing what survived. The kept traces are the same objects, not copies."""
        traces = [t for t in self.traces if keep(t)]
        return self.model_copy(update={"traces": traces}) if traces else None

    def to_record(self) -> dict[str, Any]:
        """JSON record without raw tensors — the episode form of ``Trace.to_record``, and the unit
        ``traces.jsonl`` stores: one episode per line, matching what verifiers writes and what its
        ``read_episodes`` expects."""
        return self.model_dump(mode="json", exclude={"traces": {"__all__": EXCLUDE_FIELDS}})

    @property
    def is_empty(self) -> bool:
        """Whether nothing came back at all — ``last_error`` then carries the reason. Not the
        same as failing: an episode can error and still have traces (vf keeps the completed subset
        and marks its clean siblings failed), and that failure is accounted for through those
        traces. vf's ``ok`` is the success sentinel; this is only "there is nothing here"."""
        return not self.traces


def group_rollouts(episodes: Iterable[Episode]) -> list[TrainRollout]:
    """Every trace of a group, flat — the view an algorithm comparing across the whole cohort
    wants, where the episode an attempt came from does not matter."""
    return [r for e in episodes for r in e.rollouts]


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

    def stamp(self, episode: vf.WireEpisode, *, run_id: str, policy_version: int, eval_step: int | None) -> Episode:
        """Mint the landed episode: the env's own, carrying the dispatch it came from. The group's
        values win over this dispatch's when it is still alive, so they are passed in rather than
        read off ``self``.

        The run record is written here because this is where a dispatch's facts become an episode's,
        and its ``type`` is what tells the rest of the orchestrator which path the episode is on. A
        train episode's step is not known yet — it belongs to whichever batch window is collecting
        when it lands, so the main loop fills it in."""
        landed = Episode.model_construct(
            **dict(episode),
            env_name=self.env_name,
            group_id=self.group_id,
            policy_version=policy_version,
            off_policy_steps=self.off_policy_steps,
        )
        if self.kind == "eval":
            assert eval_step is not None, "eval episode missing its step"
            run: vf.RunInfo = vf.EvalRunInfo(id=run_id, step=eval_step)
        else:
            run = vf.TrainRunInfo(id=run_id)
        landed.record_run(
            run,
            env_name=self.env_name,
            group_id=str(self.group_id),
            policy_version=policy_version,
        )
        return landed


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
