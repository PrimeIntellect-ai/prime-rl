"""What the orchestrator knows about an episode, recorded as it learns it.

An episode is serialized once, when it arrives, so a trace's later facts cannot be
implied by where its record sits. Both events below are stamped explicitly: what the
orchestrator knows at arrival goes onto the trace before it is logged, and what a batch
adds when it takes the episode becomes an append-only update the reader folds back on.
"""

from __future__ import annotations

import time
from typing import Any

import verifiers.v1 as vf

from prime_rl.monitors.base import Kind
from prime_rl.monitors.file.traces.update import make_update


def stamp_arrival(episodes: list[vf.Episode], kind: Kind, step: int) -> None:
    """Stamp what an episode's traces are known to have done as they land: the kind of
    work, the step that dispatched them, and the step they came back at."""
    now = time.time()
    for episode in episodes:
        work = getattr(episode.run, "work", None)
        for trace in episode.traces:
            trace.info["kind"] = kind
            if work is not None:
                trace.info["dispatch"] = {"step": work.step, "time": trace.timing.start}
            trace.info["arrival"] = {"step": step, "time": now}


def stamp_batch(episodes: list[vf.Episode], step: int) -> list[dict[str, Any]]:
    """The updates a batched cohort adds over its arrival records: membership, the step
    it ties to, the scalar advantage, and the per-token advantage streams."""
    now = time.time()
    updates = []
    for episode in episodes:
        for trace in episode.traces:
            info: dict[str, Any] = {"effective": True, "ship": {"step": step, "time": now}}
            if (advantage := trace.info.get("advantage")) is not None:
                info["advantage"] = advantage
            branches = {
                branch.index: {"advantages": advantages}
                for branch in trace.branches
                if (advantages := branch.advantages) is not None
            }
            updates.append(make_update(trace.id, info=info, branches=branches))
    return updates
