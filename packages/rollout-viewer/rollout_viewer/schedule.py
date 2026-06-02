"""Schedule-aware turn alignment — the protocol structure, generalized.

A *schedule* is the expected sequence of turn-slots an interaction protocol
produces. It is regime-generic:

  - single_turn            : one slot.
  - single_agent_multiturn : a loop of the lone actor (e.g. model→tool→model…),
                             length = the run's modal turn count.
  - multi_agent            : the seat/phase order (e.g. debate [a,b,a,b,judge]).

Observed ``Episode`` steps are aligned to the schedule so the timeline can show
actual-vs-expected — ``matched`` / ``truncated`` (expected slot, no step) /
``extra`` (step beyond schedule) / ``deviated`` (actor mismatch) — for ANY regime.
Debate is one instance, not a special case.

The "expected" schedule is inferred at the **run** level: the canonical (modal)
turn pattern across a run's episodes. An individual episode then aligns to it, so
a short rollout reads as *truncated relative to its run*, not just "5 turns". When
a run config that defines the protocol is available it should override the
inference (``source="config"``); v1 infers (``source="inferred"``).
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from rollout_viewer.schema import Episode, EpisodeKind, Step

AlignStatus = Literal["matched", "truncated", "extra", "deviated"]


def step_actor(step: Step) -> str:
    """The actor of a step: the seat for multi-agent, else the lone agent."""
    return step.member_id if step.member_id is not None else "agent"


class TurnSlot(BaseModel):
    """One expected turn in a schedule."""

    index: int
    actor: str  # member_id, or "agent" for single-actor regimes
    phase: str | None = None  # propose/critique/judge, tool-loop label, ...


class Schedule(BaseModel):
    """The expected turn structure of a protocol (regime-generic)."""

    regime: EpisodeKind
    slots: list[TurnSlot] = Field(default_factory=list)
    source: Literal["config", "inferred"] = "inferred"

    @property
    def length(self) -> int:
        return len(self.slots)


class AlignedTurn(BaseModel):
    """One position on the timeline: an expected slot ⨯ an observed step."""

    index: int
    status: AlignStatus
    slot: TurnSlot | None  # expected (None when the step is beyond the schedule)
    step: Step | None  # observed (None when the slot was expected but missing)


def infer_run_schedule(episodes: list[Episode]) -> Schedule:
    """Infer a run's canonical schedule as the modal turn pattern.

    The pattern is the (actor, phase) sequence; the modal *full sequence* across
    episodes is the canonical schedule. Ties break toward the longest sequence
    (the fullest observed protocol run). Raises on an empty run — a caller asking
    for a schedule with no episodes is a bug, not an empty schedule.
    """
    if not episodes:
        raise ValueError("cannot infer a schedule from zero episodes")

    regime = _dominant_regime(episodes)
    seqs: Counter[tuple[tuple[str, str | None], ...]] = Counter()
    for ep in episodes:
        if ep.error is not None:
            continue  # errored rollouts have partial trajectories — don't bias the mode
        seqs[tuple((step_actor(s), s.phase) for s in ep.steps)] += 1
    if not seqs:
        # every episode errored: fall back to the longest observed sequence
        longest = max(episodes, key=lambda e: len(e.steps))
        canonical = tuple((step_actor(s), s.phase) for s in longest.steps)
    else:
        # most common; tie -> longest
        top = max(seqs.items(), key=lambda kv: (kv[1], len(kv[0])))
        canonical = top[0]

    slots = [TurnSlot(index=i, actor=actor, phase=phase) for i, (actor, phase) in enumerate(canonical)]
    return Schedule(regime=regime, slots=slots, source="inferred")


def align(episode: Episode, schedule: Schedule) -> list[AlignedTurn]:
    """Align an episode's observed steps to the expected schedule.

    Positional zip of expected slots against observed steps:
      - both present, actor matches  -> matched
      - both present, actor differs  -> deviated
      - slot present, no step        -> truncated (rollout ended early)
      - step present, no slot        -> extra (rollout ran past the schedule)
    """
    out: list[AlignedTurn] = []
    n = max(len(schedule.slots), len(episode.steps))
    for i in range(n):
        slot = schedule.slots[i] if i < len(schedule.slots) else None
        step = episode.steps[i] if i < len(episode.steps) else None
        if slot is not None and step is not None:
            status: AlignStatus = "matched" if step_actor(step) == slot.actor else "deviated"
        elif slot is not None:
            status = "truncated"
        else:
            status = "extra"
        out.append(AlignedTurn(index=i, status=status, slot=slot, step=step))
    return out


def _dominant_regime(episodes: list[Episode]) -> EpisodeKind:
    counts = Counter(ep.kind for ep in episodes)
    return counts.most_common(1)[0][0]
