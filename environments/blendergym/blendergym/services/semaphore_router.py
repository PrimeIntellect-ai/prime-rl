"""SemaphoreRouter: least-loaded GPU selection with per-GPU semaphore.

Used by both Render Service (max_concurrent=1) and Score Service
(max_concurrent=2). The lease() context manager guarantees that GPU
slots are always released, even on cancellation or exceptions.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from .base import BaseGPURouter


class SemaphoreRouter(BaseGPURouter):
    """Least-loaded GPU routing with per-GPU semaphore back-pressure."""

    def __init__(self, gpu_pool: list[int], max_concurrent: int = 1):
        super().__init__(gpu_pool)
        self._semaphores = {g: asyncio.Semaphore(max_concurrent) for g in gpu_pool}
        self._running: dict[int, int] = {g: 0 for g in gpu_pool}
        self._pending: dict[int, int] = {g: 0 for g in gpu_pool}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def lease(self) -> AsyncGenerator[int, None]:
        async with self._lock:
            gpu = min(
                self.gpu_pool,
                key=lambda g: self._running[g] + self._pending[g],
            )
            self._pending[gpu] += 1

        try:
            await self._semaphores[gpu].acquire()
            self._pending[gpu] -= 1
            self._running[gpu] += 1
        except (asyncio.CancelledError, Exception):
            self._pending[gpu] -= 1
            raise

        try:
            yield gpu
        finally:
            self._running[gpu] -= 1
            self._semaphores[gpu].release()
