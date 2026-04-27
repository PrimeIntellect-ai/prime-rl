import asyncio
import time
from pathlib import Path
from typing import Protocol

from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_step_path
from prime_rl.utils.utils import get_latest_ckpt_step


class VersionObserver(Protocol):
    async def on_new_version(self, step: int) -> None: ...


class WeightWatcher:
    """Polls a broadcast directory for new trainer checkpoints.

    Each tick jumps straight to the LATEST step on disk, skipping intermediates
    the trainer's cleanup may have pruned. Observers are notified in order; if
    any one raises (e.g. dir disappears mid-update), we log and pick up the
    next fresher step on the next tick.
    """

    def __init__(
        self,
        broadcast_dir: Path,
        observers: list[VersionObserver],
        poll_interval: float = 0.5,
    ):
        self.broadcast_dir = broadcast_dir
        self.observers = observers
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
            if not get_step_path(self.broadcast_dir, latest).exists():
                continue  # raced with trainer cleanup
            try:
                t0 = time.perf_counter()
                for obs in self.observers:
                    await obs.on_new_version(latest)
                self.logger.success(f"Weights updated to step {latest} in {time.perf_counter() - t0:.2f}s")
                self.current_step = latest
            except Exception as exc:
                self.logger.warning(f"Weight update for step {latest} failed: {exc}. Skipping.")
                self.current_step = latest
