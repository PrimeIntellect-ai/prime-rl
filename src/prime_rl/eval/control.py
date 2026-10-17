"""SIGHUP reloads a run's desired eval concurrency from eval-control.json."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger

ResizeMode = Literal["drain", "hard"]


class ConcurrencyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: int | None = Field(..., strict=True, ge=1)
    mode: ResizeMode = "drain"


class EvalControl:
    def __init__(self, run_dir: Path, apply: Callable[[int | None, ResizeMode], Awaitable[None]]) -> None:
        self.path = run_dir / "eval-control.json"
        self.status_path = run_dir / "eval-control.status.json"
        self.apply = apply
        self.requested = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.previous_handler = None
        self.status: dict = {}

    def start(self) -> None:
        self.previous_handler = signal.getsignal(signal.SIGHUP)
        asyncio.get_running_loop().add_signal_handler(signal.SIGHUP, self.requested.set)
        self.task = asyncio.create_task(self.serve(), name="eval-control")
        get_logger().info(f"Live concurrency: write {self.path}, then send SIGHUP to PID {os.getpid()}")

    async def serve(self) -> None:
        while True:
            await self.requested.wait()
            self.requested.clear()
            try:
                request = ConcurrencyRequest.model_validate_json(self.path.read_text())
            except (OSError, ValueError) as exc:
                get_logger().warning(f"Ignoring invalid eval control request in {self.path}: {exc}")
                continue
            await self.apply(request.target, request.mode)

    def publish(self, *, target: int | None, cap: int, inflight: int) -> None:
        self.status = {
            "pid": os.getpid(),
            "running": self.task is not None,
            "target": target,
            "cap": cap,
            "inflight": inflight,
        }
        self.write_status()

    def write_status(self) -> None:
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.status) + "\n")
        temporary.replace(self.status_path)

    async def stop(self) -> None:
        if self.task is None:
            return
        asyncio.get_running_loop().remove_signal_handler(signal.SIGHUP)
        signal.signal(signal.SIGHUP, self.previous_handler)
        await safe_cancel(self.task)
        self.task = None
        self.status["running"] = False
        self.write_status()
