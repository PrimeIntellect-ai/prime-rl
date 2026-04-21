import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_step_path
from prime_rl.utils.utils import get_latest_ckpt_step

OnNewVersion = Callable[[int, Path], Awaitable[None]]


class WeightWatcher:
    """Polls a broadcast directory for new trainer checkpoints.

    Each tick jumps straight to the LATEST step on disk, skipping intermediates
    the trainer's cleanup may have pruned. If the update races with cleanup
    (dir disappears mid-load), we log and try again next tick with the newest
    step that's still there.
    """

    def __init__(
        self,
        broadcast_dir: Path,
        on_new_version: OnNewVersion,
        poll_interval: float = 0.5,
    ):
        self.broadcast_dir = broadcast_dir
        self.on_new_version = on_new_version
        self.poll_interval = poll_interval
        self.current_step = 0
        self.logger = get_logger()

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval)
            if not self.broadcast_dir.exists():
                continue
            latest = get_latest_ckpt_step(self.broadcast_dir)
            if latest is None or latest <= self.current_step:
                continue
            step_path = get_step_path(self.broadcast_dir, latest)
            if not step_path.exists():
                continue  # raced with trainer cleanup
            try:
                await self.on_new_version(latest, step_path)
                self.current_step = latest
            except Exception as exc:
                # dir can disappear mid-RPC; don't kill the watcher, just skip
                # this step and pick up the next fresher one on the next tick.
                self.logger.warning(f"Weight update for step {latest} failed: {exc}. Skipping.")
                self.current_step = latest
