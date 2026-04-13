from __future__ import annotations

import asyncio
import math

from aiolimiter import AsyncLimiter

from prime_rl.utils.logger import get_logger


class RateLimiter:
    """Rate limiter that supports variable-cost acquire (N rollouts at once).

    Wraps aiolimiter.AsyncLimiter, which rejects acquire(amount) when amount
    exceeds max_rate. We loop instead so group-scoring envs work correctly.
    If max_rate is None, acquiring is a no-op (unlimited).
    """

    def __init__(self, max_rate: float | None = None, time_period: float = 60):
        self._limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period) if max_rate is not None else None

    @property
    def unlimited(self) -> bool:
        return self._limiter is None

    async def acquire(self, count: int = 1) -> None:
        if self._limiter is None:
            return
        for _ in range(count):
            await self._limiter.acquire()


class ConcurrencyLimiter:
    """Slot-based concurrency limiter for rollouts.

    Uses a simple counter with an asyncio.Event for wakeup. All calls must
    happen on the same event loop (single-threaded asyncio), so no lock is needed.
    If max_concurrency is None, all operations are no-ops (unlimited).
    """

    def __init__(self, max_concurrency: int | None = None):
        self._max = max_concurrency
        self._used = 0
        self._available = asyncio.Event()
        self._available.set()

    @property
    def unlimited(self) -> bool:
        return self._max is None

    @property
    def max_concurrency(self) -> int | None:
        return self._max

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> float:
        if self._max is None:
            return math.inf
        return self._max - self._used

    async def acquire(self, count: int = 1) -> None:
        """Block until *count* concurrency slots are available, then reserve them."""
        if self._max is None:
            return
        while self.remaining < count:
            self._available.clear()
            await self._available.wait()
        self._used += count

    def release(self, count: int = 1) -> None:
        """Return *count* slots to the pool and wake any blocked acquirers."""
        if self._max is None:
            return
        self._used -= count
        assert self._used >= 0, f"ConcurrencyLimiter released too many slots (used={self._used})"
        self._available.set()


class RolloutLimiter:
    """Combined rate + concurrency limiter for rollout scheduling.

    Acquire gates on both limits (rate first, then concurrency).
    Release only applies to concurrency (rate tokens are not returned).
    """

    def __init__(self, max_concurrent_rollouts: int | None = None, max_rollouts_per_minute: float | None = None):
        self.concurrency = ConcurrencyLimiter(max_concurrent_rollouts)
        self.rate = RateLimiter(max_rollouts_per_minute, time_period=60)

        parts = []
        if not self.concurrency.unlimited:
            parts.append(f"max_concurrent_rollouts={max_concurrent_rollouts}")
        if not self.rate.unlimited:
            parts.append(f"max_rollouts_per_minute={max_rollouts_per_minute}")
        if parts:
            get_logger().info(f"RolloutLimiter initialized ({', '.join(parts)})")
        else:
            get_logger().info("RolloutLimiter initialized (unlimited)")

    async def acquire(self, count: int = 1) -> None:
        """Acquire rate tokens then concurrency slots (blocks until both are available)."""
        await self.rate.acquire(count)
        await self.concurrency.acquire(count)

    def release(self, count: int = 1) -> None:
        """Release concurrency slots."""
        self.concurrency.release(count)
