import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

import torch.nn as nn

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_broadcast_dir, get_step_path

# Broadcast-dir sentinels. Every transport goes through the same four stages:
# the trainer resets the step dir and raises ``.sender_ready``, the consumer
# joins the transfer and replies ``.receiver_ready`` (for NCCL with the
# engines paused inside the receive RPC), the trainer raises ``.started`` and
# moves the weights, then raises ``.finished`` — for filesystem, the weights
# are fully on disk.
SENDER_READY_MARKER = ".sender_ready"
RECEIVER_READY_MARKER = ".receiver_ready"
STARTED_MARKER = ".started"
FINISHED_MARKER = ".finished"


def prune_broadcasts_beyond(output_dir: Path, step: int) -> None:
    """Remove broadcast dirs beyond ``step``. Resume hygiene, run by the
    trainer master before its startup broadcast: stale leftovers of a longer
    crashed run would otherwise steer the consumer past the resume point."""
    broadcast_dir = get_broadcast_dir(output_dir)
    for old_step in get_all_ckpt_steps(broadcast_dir):
        if old_step > step:
            shutil.rmtree(get_step_path(broadcast_dir, old_step), ignore_errors=True)


class WeightBroadcast(ABC):
    """Trainer-side weight publisher. ``broadcast`` wraps the transport's
    ``_broadcast`` with the shared sentinel handshake: every version is
    offered (``.sender_ready``), acknowledged by the consumer
    (``.receiver_ready``), transferred (``.started``), and committed
    (``.finished``). The trainer therefore runs in lockstep with its consumer
    on every transport — a broadcast nobody receives blocks and then fails."""

    def __init__(self, output_dir: Path, timeout: int):
        self.logger = get_logger()
        self.world = get_world()
        self.output_dir = output_dir
        self.timeout = timeout

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
            (step_dir / SENDER_READY_MARKER).touch()
            self._wait_for_receiver_ready(step_dir)
            (step_dir / STARTED_MARKER).touch()
        self._broadcast(model, step, step_dir)
        if self.world.is_master:
            (step_dir / FINISHED_MARKER).touch()
            self._clean(step)
            self.logger.debug(f"Broadcasted weights for step {step} in {time.perf_counter() - start_time:.2f}s")

    def step_dir(self, step: int) -> Path:
        return get_step_path(get_broadcast_dir(self.output_dir), step)

    def _wait_for_receiver_ready(self, step_dir: Path) -> None:
        """Wait for the consumer to acknowledge the offered version. Bounded:
        a consumer that dies mid-handshake must fail the run instead of
        stranding the trainer forever."""
        ready_file = step_dir / RECEIVER_READY_MARKER
        self.logger.debug(f"Waiting for the receiver at {ready_file}")
        start = time.monotonic()
        last_log = start
        while not ready_file.exists():
            now = time.monotonic()
            if now - start > self.timeout:
                raise TimeoutError(f"No receiver joined the broadcast within {self.timeout}s ({ready_file})")
            if now - last_log > 60:
                # A busy consumer (e.g. an evals process mid-epoch) can lag legitimately;
                # raise weight_broadcast.timeout if this trips on long eval epochs.
                self.logger.warning(f"Still waiting for the broadcast receiver after {now - start:.0f}s")
                last_log = now
            time.sleep(0.1)
        self.logger.debug("Receiver ready, starting the transfer")

    @abstractmethod
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        """Move v{step}'s weights to the consumer. Rank synchronization is the
        transport's own job — non-master ranks must be held back until the
        master finished the handshake (see ``NCCLWeightBroadcast._broadcast``)."""

    def _clean(self, step: int) -> None:
        """Remove old broadcast dirs, keeping ``step`` and ``step - 1`` (a
        lagging consumer may still be reading it). Broadcasts are purely
        transitive — run state persistence is the checkpoint's job."""
        broadcast_dir = get_broadcast_dir(self.output_dir)
        for old_step in get_all_ckpt_steps(broadcast_dir):
            if old_step < step - 1:
                shutil.rmtree(get_step_path(broadcast_dir, old_step), ignore_errors=True)
