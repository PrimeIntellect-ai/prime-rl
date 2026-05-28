"""PeriodicLogger: orchestrator's pipeline view, fires every ``interval``
seconds. ``collect()`` returns ``(console_body, wandb_payload)`` in one
call so drain-on-read counters fire exactly once per tick. Wandb writes
land on the ``_timestamp`` axis."""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import wandb

from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger


class PeriodicLogger:
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
        get_logger().info(body)
        if self.wandb_enabled and payload:
            payload["_timestamp"] = time.time()
            wandb.log(payload)

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None
