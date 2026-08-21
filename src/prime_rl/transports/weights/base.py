import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, final

import torch.nn as nn

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_broadcast_dir, get_step_path

# Broadcast-dir sentinels, shared by every transport. STARTED marks a broadcast
# attempt in flight — the consumer of a live transport reacts to it by joining
# the transfer. STABLE marks a finished broadcast — for filesystem, the weights
# are fully on disk. RECEIVER_READY is the consumer's reply on the NCCL path:
# the engines are paused and inside the receive RPC, so the trainer may enter
# the collective.
STARTED_MARKER = "STARTED"
STABLE_MARKER = "STABLE"
RECEIVER_READY_MARKER = "NCCL_READY"


class WeightBroadcast(ABC):
    """Trainer-side weight publisher. ``broadcast`` wraps the transport's
    ``_broadcast`` with the shared sentinel protocol: the master resets the
    step dir and raises STARTED, the transport moves the weights, the master
    raises STABLE and prunes old step dirs."""

    # A live transport transfers directly into a running consumer: the trainer
    # blocks until the receiver joins, and a version the consumer never
    # receives strands the trainer inside the transfer.
    requires_live_consumer: ClassVar[bool] = False

    def __init__(self, output_dir: Path, keep_interval: int | None = None):
        self.logger = get_logger()
        self.world = get_world()
        self.output_dir = output_dir
        self.keep_interval = keep_interval

    @final
    def broadcast(self, model: nn.Module, step: int) -> None:
        """Broadcast policy v{step} to the inference pool."""
        start_time = time.perf_counter()
        step_dir = self.step_dir(step)
        if self.world.is_master:
            # Reset per attempt so a re-broadcast (e.g. on resume) never trips
            # the consumer or the trainer on stale markers of a previous run.
            shutil.rmtree(step_dir, ignore_errors=True)
            step_dir.mkdir(parents=True)
            (step_dir / STARTED_MARKER).touch()
        self._broadcast(model, step, step_dir)
        if self.world.is_master:
            (step_dir / STABLE_MARKER).touch()
            self._clean(step)
            self.logger.debug(f"Broadcasted weights for step {step} in {time.perf_counter() - start_time:.2f}s")

    @final
    def broadcast_startup(self, model: nn.Module, step: int) -> None:
        """Startup broadcast of v{step}, run before the first training step so
        a broken transport fails fast. Prunes broadcast dirs beyond ``step``
        first — stale leftovers of a longer crashed run would otherwise steer
        the consumer past the resume point. Filesystem skips when v{step} is
        already stable on disk (rewriting a dir a consumer may be reading is
        never safe); live transports always re-send, nothing persists."""
        if self.world.is_master:
            broadcast_dir = get_broadcast_dir(self.output_dir)
            for old_step in get_all_ckpt_steps(broadcast_dir):
                if old_step > step:
                    shutil.rmtree(get_step_path(broadcast_dir, old_step), ignore_errors=True)
        if not self.requires_live_consumer and self.is_stable(step):
            self.logger.debug(f"Skipping startup broadcast - step {step} is already stable on disk")
            return
        self.broadcast(model, step)

    def is_stable(self, step: int) -> bool:
        """Whether a complete broadcast for ``step`` is on disk."""
        return (self.step_dir(step) / STABLE_MARKER).exists()

    def step_dir(self, step: int) -> Path:
        return get_step_path(get_broadcast_dir(self.output_dir), step)

    @abstractmethod
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        """Move v{step}'s weights to the consumer. Rank synchronization is the
        transport's own job — NCCL must hold non-master ranks back until the
        receiver is ready (see ``NCCLWeightBroadcast._broadcast``)."""

    def _clean(self, step: int) -> None:
        """Remove old broadcast dirs, keeping ``step``, ``step - 1`` (a lagging
        consumer may still be reading it) and ``keep_interval`` multiples."""
        broadcast_dir = get_broadcast_dir(self.output_dir)
        for old_step in get_all_ckpt_steps(broadcast_dir):
            if old_step >= step - 1:
                continue
            if self.keep_interval and old_step % self.keep_interval == 0:
                continue
            shutil.rmtree(get_step_path(broadcast_dir, old_step), ignore_errors=True)
