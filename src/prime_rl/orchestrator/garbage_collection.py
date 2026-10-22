"""Garbage collection for the orchestrator.

CPython's full (generation-2) collection visits every live container and every reference in it,
holding the GIL. The orchestrator keeps each in-flight and buffered episode as Python lists with
one entry per token (token ids, masks, logprobs), so with ~12k agentic episodes in flight a full
collection held the GIL for up to ~200 s. Python 3.12 runs one whenever the long-lived heap has
grown by 25% since the last, so the freezes got longer and further apart as the heap grew (42 s
every ~100 s after 40 minutes, ~190 s every ~400 s after 4 hours). Each one stalled dispatch,
completions and the trainer's weight-broadcast handshake.

``GarbageCollector`` freezes the startup heap and turns off automatic full collections. Young
collections still run: they only visit objects allocated since the last one. A full collection
runs every ``full_interval`` seconds instead, to free reference cycles that reached the oldest
generation, and every collection that holds the GIL for ``warn_seconds`` or longer is logged.
"""

from __future__ import annotations

import asyncio
import gc
import time

from prime_rl.configs.orchestrator import GarbageCollectionConfig
from prime_rl.utils.logger import format_time, get_logger

# CPython runs a full collection once this many young (generation-1) collections have run, and
# then only if the long-lived heap has grown by 25%. No run reaches this count.
NEVER = 2**31 - 1


class GarbageCollector:
    def __init__(self, config: GarbageCollectionConfig):
        self.config = config
        self.started_at: float | None = None
        self.max_pause = 0.0
        self.total_pause = 0.0
        self.collections = 0
        self.slow: list[tuple[int, float]] = []

    def start(self) -> None:
        """Freeze the heap built so far (it lives for the whole run) and stop automatic full
        collections. Call once, after setup."""
        gc.collect()
        gc.freeze()
        if not self.config.automatic_full:
            threshold0, threshold1, _ = gc.get_threshold()
            gc.set_threshold(threshold0, threshold1, NEVER)
        gc.callbacks.append(self.on_collection)
        get_logger().info(
            f"Froze {gc.get_freeze_count():,} startup objects; automatic full collections "
            f"{'on' if self.config.automatic_full else 'off'}, scheduled every "
            f"{format_time(self.config.full_interval) if self.config.full_interval else 'never'}"
        )

    def on_collection(self, phase: str, info: dict) -> None:
        # Runs inside the collection, in whichever thread triggered it: no logging here.
        if phase == "start":
            self.started_at = time.perf_counter()
            return
        if self.started_at is None:
            return
        pause = time.perf_counter() - self.started_at
        self.started_at = None
        self.collections += 1
        self.total_pause += pause
        self.max_pause = max(self.max_pause, pause)
        if pause >= self.config.warn_seconds:
            self.slow.append((info["generation"], pause))

    def gauges(self) -> dict[str, float]:
        """GC pauses since the last call, for the pipeline view. Logs the slow ones."""
        for generation, pause in self.slow:
            get_logger().warning(f"Generation-{generation} garbage collection held the GIL for {format_time(pause)}")
        gauges = {
            "gc/collections": float(self.collections),
            "gc/pause_max": self.max_pause,
            "gc/pause_total": self.total_pause,
        }
        self.slow, self.collections, self.max_pause, self.total_pause = [], 0, 0.0, 0.0
        return gauges

    async def run(self) -> None:
        """Run a full collection every ``full_interval`` seconds; run as a background task."""
        if self.config.full_interval is None:
            return
        while True:
            await asyncio.sleep(self.config.full_interval)
            start = time.perf_counter()
            freed = gc.collect()
            get_logger().info(
                f"Scheduled full garbage collection freed {freed:,} objects in {format_time(time.perf_counter() - start)}"
            )
