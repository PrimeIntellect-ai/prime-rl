import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path
from prime_rl.utils.utils import get_latest_ckpt_step


class VersionObserver(Protocol):
    async def on_new_version(self, step: int) -> None: ...


@dataclass
class WatcherInputs:
    """Pre-built inputs for the WeightWatcher. `setup_watcher` produces this
    from config; tests construct it directly with a tmp_path + stub observers."""

    broadcast_dir: Path
    observers: list[VersionObserver]
    poll_interval: float = 0.5


class WeightWatcher:
    """Polls a broadcast directory for new trainer checkpoints.

    Each tick jumps straight to the LATEST step on disk, skipping intermediates
    the trainer's cleanup may have pruned. Observers are notified in order; if
    any one raises (e.g. dir disappears mid-update), we log and pick up the
    next fresher step on the next tick.
    """

    def __init__(self, inputs: WatcherInputs):
        self.broadcast_dir = inputs.broadcast_dir
        self.observers = inputs.observers
        self.poll_interval = inputs.poll_interval
        self.current_step = 0
        self.logger = get_logger()

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval)
            await self.tick()

    async def tick(self) -> None:
        """One poll iteration. Extracted so tests can drive it without a loop."""
        if not self.broadcast_dir.exists():
            return
        latest = get_latest_ckpt_step(self.broadcast_dir)
        if latest is None or latest <= self.current_step:
            return
        if not get_step_path(self.broadcast_dir, latest).exists():
            return  # raced with trainer cleanup
        try:
            t0 = time.perf_counter()
            for obs in self.observers:
                await obs.on_new_version(latest)
            self.logger.success(f"Weights updated to step {latest} in {time.perf_counter() - t0:.2f}s")
            self.current_step = latest
        except Exception as exc:
            self.logger.warning(f"Weight update for step {latest} failed: {exc}. Skipping.")
            self.current_step = latest


def setup_watcher(cfg: OrchestratorConfig, *, observers: list[VersionObserver]) -> WeightWatcher:
    """Translate config → WeightWatcher. Tests should construct
    `WeightWatcher(WatcherInputs(...))` directly."""
    return WeightWatcher(
        WatcherInputs(
            broadcast_dir=get_broadcast_dir(cfg.output_dir),
            observers=observers,
        )
    )
