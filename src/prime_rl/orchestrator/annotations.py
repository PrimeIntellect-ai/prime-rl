"""What the orchestrator knows about an episode, recorded as it learns it.

An episode is serialized once, by the env worker that ran it, so a trace's later facts cannot
be implied by where its record sits. The worker stamps the kind of work and the step that
dispatched it; what the orchestrator learns later becomes an append-only update the reader
folds back on: when the episode came back, and what a batch adds when it takes the episode.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from prime_rl import monitors
from prime_rl.monitors.base import Kind
from prime_rl.monitors.file.traces.update import make_update
from prime_rl.orchestrator.credit import trace_advantage

if TYPE_CHECKING:
    from prime_rl.env.heads import EpisodeHead, EpisodeRecord


def stamp_arrival(episode: EpisodeHead, record: EpisodeRecord | None, kind: Kind, step: int) -> list[dict[str, Any]]:
    """Stamp when an episode's traces came back, and at which step: into its record's index
    row, and as updates over its record."""
    arrival = {"arrival": {"step": step, "time": time.time()}}
    if record is not None:
        for trace in record.index.get("traces") or []:
            trace["info"] = {**(trace.get("info") or {}), **arrival}
    return [make_update(trace.id, info={"kind": kind, **arrival}) for trace in episode.traces]


async def log_arrival(episode: EpisodeHead, kind: Kind, step: int) -> None:
    """Log an arrived episode's record, stamped with its arrival, then the arrival's updates
    over it. The record goes first: under a trace sample, the file monitor annotates only the
    traces it wrote. An episode no monitor keeps a record of has nothing to annotate."""
    record = episode.take_record()
    if record is None:
        return
    updates = stamp_arrival(episode, record, kind, step)
    await monitors.log_records([(episode, record)], step, kind)
    await monitors.log_annotations(updates)


def stamp_batch(episodes: list[EpisodeHead], step: int) -> list[dict[str, Any]]:
    """The updates a batched cohort adds over its arrival records: membership, the step
    it ties to, and the trace's advantage with the branches it credits."""
    now = time.time()
    updates = []
    for episode in episodes:
        for trace in episode.traces:
            info: dict[str, Any] = {"effective": True, "ship": {"step": step, "time": now}}
            if (advantage := trace_advantage(trace)) is not None:
                info["advantage"] = advantage
                info["advantage_branches"] = trace.trainable_branches
            updates.append(make_update(trace.id, info=info))
    return updates
