from __future__ import annotations

import asyncio
from collections import deque
from time import perf_counter

import numpy as np
from pydantic import BaseModel


async def safe_cancel(task: asyncio.Task) -> None:
    """Safely cancels and awaits an asyncio.Task."""
    task.cancel()
    try:
        await task
    except BaseException:
        pass


async def safe_cancel_all(tasks: list[asyncio.Task]) -> None:
    """Safely cancels and awaits all asyncio.Tasks."""
    await asyncio.gather(*[safe_cancel(task) for task in tasks])


class EventLoopLagMonitor:
    """Monitors how busy the main event loop is by timing short sleeps.

    Vendored from verifiers.utils.async_utils (the orchestrator now runs on
    v1 and no longer depends on v1 verifiers)."""

    def __init__(self, measure_interval: float = 0.1, max_measurements: int = 1000):
        assert measure_interval > 0 and max_measurements > 0
        self.measure_interval = measure_interval
        self.max_measurements = max_measurements
        self.lags: deque[float] = deque(maxlen=max_measurements)

    async def measure_lag(self) -> float:
        next_time = perf_counter() + self.measure_interval
        await asyncio.sleep(self.measure_interval)
        return perf_counter() - next_time

    async def run(self) -> None:
        """Loop measuring event-loop lag; run as a background task."""
        while True:
            self.lags.append(await self.measure_lag())


class EventLoopLagStats(BaseModel):
    """Snapshot of event-loop lag statistics."""

    min: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    max: float = 0.0
    n: int = 0

    @classmethod
    def from_monitor(cls, monitor: EventLoopLagMonitor) -> EventLoopLagStats:
        n = len(monitor.lags)
        if n == 0:
            return cls(n=0)
        arr = np.array(monitor.lags)
        return cls(
            min=float(arr.min()),
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            p90=float(np.percentile(arr, 90)),
            p99=float(np.percentile(arr, 99)),
            max=float(arr.max()),
            n=n,
        )
