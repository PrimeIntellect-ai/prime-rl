"""PeriodicLogger: orchestrator's pipeline view, fires every ``interval``
seconds. ``collect()`` returns ``(console_body, payload)`` in one call so
drain-on-read counters fire exactly once per tick. The payload goes to every
registered monitor as a time-keyed row (``step=None``)."""

from __future__ import annotations

import asyncio
from typing import Callable

from prime_rl import monitors
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger


class PeriodicLogger:
    def __init__(
        self,
        *,
        name: str,
        collect: Callable[[], tuple[str, dict[str, float]]],
        interval: float,
    ) -> None:
        self.name = name
        self.collect = collect
        self.interval = interval
        self.task: asyncio.Task | None = None
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name=f"{self.name}_periodic_logger")

    async def run(self) -> None:
        try:
            while not self.stopped.is_set():
                try:
                    await asyncio.wait_for(self.stopped.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass
                await self.emit()
        except asyncio.CancelledError:
            return

    async def emit(self) -> None:
        body, payload = self.collect()
        get_logger().info(body)
        if payload:
            # Time-keyed row: the pipeline view is sampled on wall time, not
            # the training step. Each monitor stamps its own time axis.
            await monitors.log(payload, step=None)

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None
