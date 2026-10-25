"""Garbage collection for the orchestrator.

CPython's full (generation-2) collection visits every live container and every reference in it,
holding the GIL. With thousands of agentic episodes in flight and buffered, that froze the event
loop for minutes at a time. Python 3.12 runs one whenever the long-lived heap has grown by 25%
since the last, so the freezes got longer and further apart as the heap grew (42 s every ~100 s
after 40 minutes, ~190 s every ~400 s after 4 hours). Each one stalled dispatch, completions and
the trainer's weight-broadcast handshake.

``GarbageCollector`` freezes the startup heap and turns off automatic full collections. Young
collections still run: they only visit objects allocated since the last one. A full collection
runs every ``interval`` steps instead, right after the previous batch is released, to free
reference cycles that reached the oldest generation.
"""

from __future__ import annotations

import gc
import time

from prime_rl.utils.logger import format_time, get_logger

# CPython runs a full collection once this many young (generation-1) collections have run, and
# then only if the long-lived heap has grown by 25%. No run reaches this count.
NEVER = 2**31 - 1

# Log every collection that holds the GIL at least this long.
WARN_SECONDS = 1.0


class GarbageCollector:
    def __init__(self, interval: int):
        assert interval > 0, "gc interval must be a positive integer"
        self.interval = interval
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
        threshold0, threshold1, _ = gc.get_threshold()
        gc.set_threshold(threshold0, threshold1, NEVER)
        gc.callbacks.append(self.on_collection)
        get_logger().info(
            f"Froze {gc.get_freeze_count():,} startup objects; full garbage collection every {self.interval} steps"
        )

    def run(self, step: int) -> None:
        if step > 0 and step % self.interval == 0:
            start = time.perf_counter()
            freed = gc.collect()
            get_logger().info(
                f"Scheduled full garbage collection freed {freed:,} objects in {format_time(time.perf_counter() - start)}"
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
        if pause >= WARN_SECONDS:
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
