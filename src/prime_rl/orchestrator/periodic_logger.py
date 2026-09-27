"""PeriodicLogger: one pipeline log line every ``interval`` seconds. Components
register a ``status`` (a console fragment) and/or ``gauges`` (a metrics dict);
each tick joins the fragments into the line and sends the merged gauges to the
monitors as a time-keyed row (``step=None``). Drain-on-read counters fire exactly
once per tick because every provider is read once."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from prime_rl import monitors as default_monitors
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger


class PeriodicLogger:
    def __init__(self, *, name: str, interval: float) -> None:
        self.name = name
        self.interval = interval
        self._status: list[Callable[[], str | None]] = []
        self._gauges: list[Callable[[], dict[str, float]]] = []
        self.monitors = default_monitors
        self.task: asyncio.Task | None = None
        self.stopped = asyncio.Event()

    def register(
        self,
        *,
        status: Callable[[], str | None] | None = None,
        gauges: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        if status is not None:
            self._status.append(status)
        if gauges is not None:
            self._gauges.append(gauges)

    def bind(self, *, monitors=None) -> None:
        if monitors is not None:
            self.monitors = monitors

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

    def collect(self) -> tuple[str, dict[str, float]]:
        body = "; ".join(fragment for provider in self._status if (fragment := provider()))
        payload: dict[str, float] = {}
        for provider in self._gauges:
            payload |= provider()
        return body, payload

    async def emit(self) -> None:
        body, payload = self.collect()
        if body:
            get_logger().info(body)
        if payload:
            await self.monitors.log(payload, step=None)

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None
