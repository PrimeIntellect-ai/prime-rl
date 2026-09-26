"""The dispatcher's side of the live traces.

Env servers stream every in-flight rollout as deltas (``verifiers.v1.serve.delta``).
``LiveStream`` relays each delta to the monitors in batches, stamped with who
dispatched it, and the dispatcher keeps only what its own progress line needs: the
phase and turn count of each live trace.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from prime_rl import monitors as default_monitors
from prime_rl.monitors.file.traces.live import STAGES, stage
from prime_rl.orchestrator.types import InflightEpisode, LiveTrace
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger

FLUSH_INTERVAL_S = 0.5
EVENT_CAP = 20_000
"""Buffered events before the oldest deltas are dropped."""


def dispatch_info(meta: InflightEpisode) -> dict[str, Any]:
    """Who an in-flight episode is, stamped on the first line of each of its live traces
    and on its pending placeholder."""
    data = meta.task.data
    name = getattr(data, "name", None)
    return {
        "id": meta.dispatch_id,
        "kind": meta.kind,
        "env": meta.env_name,
        "group": str(meta.group_id),
        "task": str(name) if name else f"idx={data.idx}",
        "policy_version": meta.policy_version,
        "step": meta.step,
        "started": time.time() - (time.monotonic() - meta.started_at) if meta.started_at else None,
    }


def stage_counts(inflight: list[InflightEpisode]) -> str:
    """``boot 1 · running 3`` — the in-flight traces by phase, in lifecycle order; an
    episode with no trace yet counts as pending."""
    counts = dict.fromkeys(STAGES, 0)
    for meta in inflight:
        if not meta.live:
            counts["pending"] += 1
        for trace in meta.live.values():
            counts[trace.stage] = counts.get(trace.stage, 0) + 1
    return " · ".join(f"{name} {n}" for name, n in counts.items() if n)


class LiveStream:
    """Buffers live events and hands them to the monitors at most twice a second."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.dropped = False
        self.monitors: Any = default_monitors
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self.publish(), name="live_stream")

    async def stop(self) -> None:
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None
        await self.flush()

    def dispatched(self, meta: InflightEpisode) -> None:
        """The episode was dispatched; until a trace streams, this is all there is to show."""
        self.events.append({"pending": meta.dispatch_id, "dispatch": dispatch_info(meta)})

    def delta(self, meta: InflightEpisode, delta: dict[str, Any]) -> None:
        """Fold one delta into the episode's phase/turn bookkeeping and relay it."""
        first = not meta.live
        trace_id = delta["trace"]
        if delta.get("discard"):
            meta.live.pop(trace_id, None)
        else:
            trace = meta.live.setdefault(trace_id, LiveTrace())
            trace.turns += len(delta.get("calls") or [])
            changed = delta.get("set") or {}
            if delta.get("errors"):
                trace.stage = "error"
            elif "timing" in changed:
                trace.stage = stage({"timing": changed["timing"]})
        self.events.append({"delta": delta, "dispatch": dispatch_info(meta)})
        # After the delta, and only once a trace streams: a reader never sees the
        # episode in neither place (a discard as the first delta streams nothing).
        if first and meta.live:
            self.events.append({"dispatched": meta.dispatch_id})
        if len(self.events) > EVENT_CAP:
            # Past the cap the oldest deltas go (the live view of those traces misses a
            # turn), never the bookkeeping events that create or remove files.
            kept = [event for event in self.events if "delta" not in event]
            deltas = [event for event in self.events if "delta" in event]
            self.events = kept + deltas[len(deltas) - EVENT_CAP // 2 :]
            if not self.dropped:
                self.dropped = True
                get_logger().warning(f"Live trace buffer over {EVENT_CAP} events - dropping the oldest deltas")

    def retired(self, meta: InflightEpisode) -> None:
        """An episode left the in-flight set (finished, cancelled, dropped): its live
        traces are over, and so is its pending placeholder if no trace ever streamed."""
        if not meta.live:
            self.events.append({"dispatched": meta.dispatch_id})
        self.events.extend({"done": trace_id} for trace_id in meta.live)

    async def flush(self) -> None:
        if not self.events:
            return
        events, self.events = self.events, []
        try:
            await self.monitors.log_live(events)
        except asyncio.CancelledError:
            # stop() cancels the publisher mid-write and flushes once more: nothing is lost
            self.events = events + self.events
            raise

    async def publish(self) -> None:
        while True:
            await asyncio.sleep(FLUSH_INTERVAL_S)
            await self.flush()
