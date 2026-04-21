import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from prime_rl.utils.pathing import get_step_path
from prime_rl.utils.utils import get_latest_ckpt_step

OnNewVersion = Callable[[int, Path], Awaitable[None]]


class WeightWatcher:
    """Polls a broadcast directory for new trainer checkpoints.

    When a new step's STABLE marker appears, calls `on_new_version(step, step_path)`.
    """

    def __init__(
        self,
        broadcast_dir: Path,
        on_new_version: OnNewVersion,
        poll_interval: float = 1.0,
    ):
        self.broadcast_dir = broadcast_dir
        self.on_new_version = on_new_version
        self.poll_interval = poll_interval
        self.current_step = 0

    async def run(self) -> None:
        while True:
            if self.broadcast_dir.exists():
                latest = get_latest_ckpt_step(self.broadcast_dir)
                if latest is not None and latest > self.current_step:
                    step_path = get_step_path(self.broadcast_dir, latest)
                    self.current_step = latest
                    await self.on_new_version(latest, step_path)
            await asyncio.sleep(self.poll_interval)
