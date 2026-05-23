"""Event-loop lag monitor used by perf/r3 microbenches.

Polls the asyncio loop every `interval` seconds via `asyncio.sleep` and records
the difference between expected wake time and actual wake time. Anything > the
sample interval is real, unblockable lag (the loop was busy doing sync work
or the GIL was held).
"""

from __future__ import annotations

import asyncio
import statistics
from collections import deque
from time import perf_counter


class LagMonitor:
    def __init__(self, interval: float = 0.01, capacity: int = 200_000):
        self.interval = interval
        self.lags: deque[float] = deque(maxlen=capacity)
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def _run(self) -> None:
        while not self._stop.is_set():
            t0 = perf_counter()
            await asyncio.sleep(self.interval)
            lag = (perf_counter() - t0) - self.interval
            if lag < 0:
                lag = 0.0
            self.lags.append(lag)

    def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    def reset(self) -> None:
        self.lags.clear()

    def snapshot(self) -> dict[str, float]:
        if not self.lags:
            return {"n": 0}
        arr = sorted(self.lags)
        n = len(arr)

        def pct(p: float) -> float:
            i = max(0, min(n - 1, int(round((p / 100) * (n - 1)))))
            return arr[i]

        return {
            "n": n,
            "min": arr[0],
            "median": arr[n // 2],
            "mean": statistics.fmean(arr),
            "p90": pct(90),
            "p99": pct(99),
            "max": arr[-1],
        }


def fmt_ms(s: float) -> str:
    if s < 1.0:
        return f"{s*1000:.1f}ms"
    return f"{s:.2f}s"


def fmt(d: dict) -> str:
    if not d or d.get("n", 0) == 0:
        return "no samples"
    return (
        f"n={d['n']:5d} min={fmt_ms(d['min'])} mean={fmt_ms(d['mean'])} "
        f"median={fmt_ms(d['median'])} p90={fmt_ms(d['p90'])} "
        f"p99={fmt_ms(d['p99'])} max={fmt_ms(d['max'])}"
    )
