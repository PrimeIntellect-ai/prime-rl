"""Shared dataclasses for the orchestrator. Data carriers only; no behavior."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Protocol, cast

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


Rollout = vf.Trace
"""The env's own trace, unextended — what an eval rollout is, and the read side of a training one.
Where a trace sits (its episode, its group, its env) is the episode's to say."""


class TrainRollout(vf.Trace[DataT], Generic[DataT]):
    """A rollout on the training path — the one place prime-rl extends a verifiers type, because
    trainer-bound state has nowhere else to live: the samples built from its branches, what the
    degeneracy measurements found on it, and whether the drop policy kept it. All of it is
    ``exclude=True``, so dumping one yields a plain trace on the wire.

    ``env_name`` rides along because a sample is routed by it (the trainer's per-env loss config)
    and the sink's pending batch is a trace list. Eval rollouts carry the fields unset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # ``samples`` holds msgspec structs

    env_name: str = Field(default="", exclude=True)
    samples: list[TrainingSample] = Field(default_factory=list, exclude=True)
    degeneracy: dict[str, bool] = Field(default_factory=dict, exclude=True)
    """What each degeneracy measurement found on this trace — a measurement, not a verdict."""
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


Episode = vf.WireEpisode
"""The env's own episode, unextended: everything prime-rl needs to say about a dispatch has a
place on it already — the env it ran (``env.name``), the group it was planned in (``group``), and
the run it belongs to (``run``, which on the training path carries the policy version and how
stale it got)."""


def narrow(episode: Episode, keep: Callable[[Rollout], bool]) -> Episode | None:
    """The episode with only the traces that pass ``keep``, or ``None`` if none do. A subset stays
    a list of episodes rather than a flat trace list, so the episode-level aggregates keep
    describing what survived. The kept traces are the same objects, not copies."""
    traces = [t for t in episode.traces if keep(t)]
    return episode.model_copy(update={"traces": traces}) if traces else None


def rollouts_of(episode: Episode) -> list[TrainRollout]:
    """An episode's traces, typed as the rollouts prime-rl works with. Every trace is built as a
    ``TrainRollout`` (``Env.run`` is shared), so the training fields are always reachable — on the
    eval path they simply stay empty."""
    return cast(list[TrainRollout], episode.traces)


def to_record(episode: Episode) -> dict[str, Any]:
    """JSON record without the per-node training tensors — the episode form of
    ``Trace.to_record``, and the unit ``traces.jsonl`` stores: one episode per line. The tensors
    are the trainer's, not the record's, and raw numpy bytes don't round-trip through json."""
    return episode.model_dump(mode="json", exclude={"traces": {"__all__": EXCLUDE_FIELDS}})


def group_id_of(episode: Episode) -> str:
    """The group an episode was planned in. The dispatcher plans every episode into one, so this
    is always set by the time anything downstream asks."""
    assert episode.group is not None, "the dispatcher plans every episode into a group"
    return episode.group.id


def env_name_of(episode: Episode) -> str:
    """The env as prime-rl names it — its config key, which is not vf's ``env.id``."""
    return episode.env.name or ""


def run_of(episode: Episode) -> vf.TrainRunInfo:
    """An episode's run record. Every episode the orchestrator produces belongs to the training
    run — the ones it trains on and the ones it evaluates along the way — so this is always a
    ``TrainRunInfo``, and ``kind`` says which of the two it is."""
    assert isinstance(episode.run, vf.TrainRunInfo), "the dispatcher records the run on arrival"
    return episode.run


def group_rollouts(episodes: Iterable[Episode]) -> list[TrainRollout]:
    """Every trace of a group, flat — the view an algorithm comparing across the whole group
    wants, where the episode an attempt came from does not matter."""
    return [r for e in episodes for r in rollouts_of(e)]


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
    eval_step: int | None = None

    def stamp(self, episode: Episode, *, run_id: str, policy: vf.PolicySpan | None, eval_step: int | None) -> Episode:
        """Write the dispatch's facts onto the landed episode, in the places the episode already
        has for them. The group's values win over this dispatch's when it is still alive, so they
        are passed in rather than read off ``self``.

        The run's metadata says what the episode is to the run, which is all the rest of the
        orchestrator needs to route it. An eval knows its step here; an episode to train on does
        not — it belongs to whichever batch window is collecting when it lands, so the main loop
        fills that in."""
        episode.env.name = self.env_name
        episode.group = vf.GroupInfo(id=str(self.group_id))
        if self.kind == "eval":
            assert eval_step is not None, "eval episode missing its step"
            metadata: vf.EpisodeMetadata = vf.EvalMetadata(step=eval_step)
        else:
            metadata = vf.TrainMetadata()
        episode.run = vf.TrainRunInfo(id=run_id, metadata=metadata, policy=policy)
        return episode


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
