"""PeriodicLogger: owned by each async component, fires on a shared interval.

Each async component (dispatcher, watcher, orchestrator main loop) constructs
its own ``PeriodicLogger`` with a ``collect()`` callable that returns both
its human-readable console body string and its flat wandb metrics dict —
in one call, so any drain-on-read counters fire exactly once per tick.

The logger wakes every ``interval`` seconds and emits to:

- **Console** at info-level: a single line per component formatted as
  ``Name       | k=v | k=v | …`` with a padded name column so columns
  align across components.
- **Wandb** on the ``_timestamp`` axis: each metric key registered at
  construction via ``wandb.define_metric(step_metric="_timestamp")`` so it
  goes on the wall-clock time axis, not the step axis.

Lifecycle: ``start()`` when the owning component starts (spawns the task);
``stop()`` when it stops (cancels the task). The logger is live for the
whole lifetime of its owner — same as the watcher's polling loop or the
dispatcher's dispatch loop.

When ``wandb_enabled=False`` (e.g. ``--no-wandb``), the wandb side is
skipped and the console line still fires. No defensive try/except around
``wandb.log`` — if wandb is enabled, it's expected to work.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import wandb

from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger

# Padded width for the leading name column across all periodic-logger
# lines so the ``|`` separators visually align between components.
# Current component names: "Event Loop" (10), "Dispatcher" (10),
# "Watcher" (7). Bump if a longer name is added.
NAME_WIDTH = 10


class PeriodicLogger:
    """``await logger.start()`` from inside the owning component."""

    def __init__(
        self,
        *,
        name: str,
        collect: Callable[[], tuple[str, dict[str, float]]],
        metric_keys: list[str],
        interval: float,
        wandb_enabled: bool,
    ) -> None:
        self.name = name
        self.collect = collect
        self.interval = interval
        self.wandb_enabled = wandb_enabled
        self.task: asyncio.Task | None = None
        self.stopped = asyncio.Event()

        # Register the wall-clock time axis for our specific metric keys up
        # front. Only the keys we'll actually log get a ``define_metric``
        # call — no glob patterns, no lazy registration.
        if self.wandb_enabled:
            for key in metric_keys:
                wandb.define_metric(key, step_metric="_timestamp")

    async def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name=f"{self.name}_periodic_logger")

    async def run(self) -> None:
        try:
            while not self.stopped.is_set():
                try:
                    await asyncio.wait_for(self.stopped.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass
                self.emit()
        except asyncio.CancelledError:
            return

    def emit(self) -> None:
        body, payload = self.collect()
        get_logger().info(f"{self.name:<{NAME_WIDTH}} | {body}")
        if self.wandb_enabled and payload:
            payload["_timestamp"] = time.time()
            wandb.log(payload)

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None
