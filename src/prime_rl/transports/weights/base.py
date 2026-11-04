import asyncio
import itertools
import json
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import final

import torch.nn as nn

from prime_rl.configs.shared import WeightBroadcastConfig
from prime_rl.orchestrator.clients import AdminPlane, Engine
from prime_rl.trainer.world import get_world, share_from_master, wait_for_master
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
# Raised by the consumer when it replaces a failed publisher: in-flight transfers of that step stop.
# Also raised when an inference server is lost during an elastic transfer, which is then sent again.
ABORTED_MARKER = ".aborted"
# Raised by the consumer once every inference server took the version (elastic transfers only).
RECEIVED_MARKER = ".received"


def write_marker(path: Path, content: str) -> None:
    """Write a marker the other side reads as soon as it exists: it appears complete."""
    partial = path.with_name(f"{path.name}.tmp")
    partial.write_text(content)
    partial.replace(path)


class BroadcastAborted(Exception):
    """A receiver stopped waiting on a publisher that was replaced. Inference was not touched."""


def prune_broadcasts_beyond(output_dir: Path, step: int) -> None:
    """Remove broadcast dirs beyond ``step``. Resume hygiene, run by the
    trainer master before its startup broadcast: stale leftovers of a longer
    crashed run would otherwise steer the consumer past the resume point."""
    broadcast_dir = get_broadcast_dir(output_dir)
    for old_step in get_all_ckpt_steps(broadcast_dir):
        if old_step > step:
            shutil.rmtree(get_step_path(broadcast_dir, old_step), ignore_errors=True)


class WeightSender(ABC):
    """Trainer-side weight publisher. ``broadcast`` wraps the transport's
    ``_broadcast`` with the shared sentinel handshake: every version is
    offered (``.sender_ready``), acknowledged by the consumer
    (``.receiver_ready``), transferred (``.started``), and committed
    (``.finished``). The trainer therefore runs in lockstep with its consumer
    on every transport — a broadcast nobody receives blocks and then fails.

    A transport that ``confirms_delivery`` also waits for the consumer's verdict: ``.received``
    commits the version, ``.aborted`` (an inference server was lost) offers it again. Each offer
    carries an attempt ID that the consumer's acknowledgement echoes."""

    confirms_delivery = False

    def __init__(self, output_dir: Path, timeout: int):
        self.logger = get_logger()
        self.world = get_world()
        self.output_dir = output_dir
        self.timeout = timeout
        self.ack: dict = {}
        """The master's last acknowledgement: what the consumer sends with it (e.g. a group size)."""
        self.offers = 0
        """Broadcasts started by this process. Every rank counts them in lockstep, so the store keys
        that pace the ranks are new for each one, also when a version is published again."""

    @final
    def broadcast(self, model: nn.Module, step: int) -> None:
        """Broadcast policy v{step} to the inference pool."""
        start_time = time.perf_counter()
        step_dir = self.step_dir(step)
        self.offers += 1
        for attempt in itertools.count():
            if self.world.is_master:
                # Reset per attempt so a re-broadcast (e.g. on resume) never trips
                # the consumer or the trainer on stale markers of a previous run.
                shutil.rmtree(step_dir, ignore_errors=True)
                step_dir.mkdir(parents=True)
                attempt_id = uuid.uuid4().hex
                write_marker(step_dir / SENDER_READY_MARKER, attempt_id)
                self.ack = self._wait_for_receiver_ready(step_dir, attempt_id)
                (step_dir / STARTED_MARKER).touch()
            wait_for_master(f"weight_broadcast/{self.offers}/{attempt}", self.timeout)
            sent = self._broadcast(model, step, step_dir)
            if not self.confirms_delivery:
                break
            outcome = None
            if self.world.is_master:
                outcome = "received" if self._wait_for_delivery(step_dir, sent is not False) else "aborted"
            if (
                share_from_master(f"weight_broadcast/{self.offers}/{attempt}/outcome", outcome, self.timeout)
                == "received"
            ):
                break
            self.logger.warning(f"Inference lost a server while receiving policy v{step}; sending it again")
        if self.world.is_master:
            (step_dir / FINISHED_MARKER).touch()
            self._clean(step)
            self.logger.debug(f"Broadcasted weights for step {step} in {time.perf_counter() - start_time:.2f}s")

    def step_dir(self, step: int) -> Path:
        return get_step_path(get_broadcast_dir(self.output_dir), step)

    def _wait_for_receiver_ready(self, step_dir: Path, attempt_id: str) -> dict:
        """Wait for the consumer to acknowledge the offered version and return what it sent with
        it. Bounded: a consumer that dies mid-handshake must fail the run instead of stranding the
        trainer forever."""
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
        ack = json.loads(ready_file.read_text() or "{}")
        if ack.get("attempt", attempt_id) != attempt_id:
            raise RuntimeError(f"The receiver acknowledged attempt {ack['attempt']}, not {attempt_id} ({ready_file})")
        self.logger.debug("Receiver ready, starting the transfer")
        return ack

    def _wait_for_delivery(self, step_dir: Path, sent: bool) -> bool:
        """Whether the consumer confirmed that every inference server took the version, rather than
        abort it. A failed send waits for the abort: the consumer stops the servers' transfer."""
        start = time.monotonic()
        while True:
            if (step_dir / ABORTED_MARKER).exists():
                return False
            if sent and (step_dir / RECEIVED_MARKER).exists():
                return True
            if time.monotonic() - start > self.timeout:
                raise TimeoutError(f"The receiver neither confirmed nor aborted {step_dir} within {self.timeout}s")
            time.sleep(0.1)

    @abstractmethod
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> bool | None:
        """Move v{step}'s weights to the consumer. All ranks enter only after
        the master finished the handshake. Returns False if the master's send was stopped."""

    def _clean(self, step: int) -> None:
        """Remove old broadcast dirs, keeping ``step`` and ``step - 1`` (a
        lagging consumer may still be reading it). Broadcasts are purely
        transitive — run state persistence is the checkpoint's job."""
        broadcast_dir = get_broadcast_dir(self.output_dir)
        for old_step in get_all_ckpt_steps(broadcast_dir):
            if old_step < step - 1:
                shutil.rmtree(get_step_path(broadcast_dir, old_step), ignore_errors=True)


class WeightReceiver(ABC):
    """Consumer-side counterpart of ``WeightSender`` — moves the inference
    engines onto trainer-offered policy versions. It runs in the consumer
    process (the RL orchestrator, or the SFT evals process): the weight
    watcher, the orchestrator's startup rendezvous, and the evals process all
    drive the same object instead of hand-rolling transport branches. The
    engines' in-process receive hooks are its data plane, reached through the
    admin plane."""

    def __init__(
        self,
        broadcast_dir: Path,
        config: WeightBroadcastConfig,
        admin_plane: AdminPlane,
        model_name: str,
    ) -> None:
        self.logger = get_logger()
        self.broadcast_dir = broadcast_dir
        self.config = config
        self.admin_plane = admin_plane
        self.model_name = model_name
        self.aborted = asyncio.Event()
        self.awaited_step: int | None = None
        """The offer this receiver last waited for, found by name rather than by listing the directory."""

    async def initialize(self) -> None:
        """One-time transport bootstrap (rendezvous groups, sessions)."""

    reconnect_retryable = True
    """A failed ``reconnect`` can run again. False for a one-shot rendezvous with the publisher."""

    async def reconnect(self) -> None:
        """Join a replacement publisher's transfer group. Only transports whose inference
        workers hold a session with the publisher reconnect."""

    def abort(self) -> None:
        """Stop waiting on the current publisher (it was replaced): waits for its markers raise
        ``BroadcastAborted`` until ``clear_abort``, and in-flight recoverable NCCL transfers stop
        at their next poll. A load from complete files is not interrupted."""
        self.aborted.set()
        for step in self._offered_steps():
            if self.step_dir(step).exists():
                (self.step_dir(step) / ABORTED_MARKER).touch()

    def clear_abort(self) -> None:
        self.aborted.clear()

    def clear_offers(self, after: int | None = None, through: int | None = None) -> None:
        """Drop every offered version: a replaced publisher's offers can never complete. Steps in
        (``after``, ``through``] are dropped by name too: the listing can still hide the step directory
        the publisher created last, which would surface as an offer nobody completes."""
        steps = self._offered_steps()
        if after is not None and through is not None:
            steps.update(range(after + 1, through + 1))
        for step in steps:
            shutil.rmtree(self.step_dir(step), ignore_errors=True)

    def _offered_steps(self) -> set[int]:
        """Steps in the broadcast directory, including the one this receiver last waited for: a listing
        of a shared filesystem can lag behind a step directory found by name."""
        steps = set(get_all_ckpt_steps(self.broadcast_dir))
        if self.awaited_step is not None:
            steps.add(self.awaited_step)
        return steps

    async def _wait_for(self, path: Path, cancelled: Callable[[], bool] | None = None) -> None:
        while not path.exists():
            if self.aborted.is_set():
                raise BroadcastAborted(f"Stopped waiting for {path}")
            if cancelled is not None and cancelled():
                raise asyncio.CancelledError
            await asyncio.sleep(0.1)

    def step_dir(self, step: int) -> Path:
        return get_step_path(self.broadcast_dir, step)

    def is_published(self, step: int) -> bool:
        """Whether the trainer has offered v{step}."""
        return (self.step_dir(step) / SENDER_READY_MARKER).exists()

    def next_version(self, current: int) -> int:
        """Newest version offered beyond ``current``; ``current`` if none."""
        published = [step for step in get_all_ckpt_steps(self.broadcast_dir) if self.is_published(step)]
        return max(published, default=current)

    async def wait_published(self, step: int, cancelled: Callable[[], bool] | None = None) -> None:
        """Block until the trainer offers v{step}. Runs before the orchestrator
        advances ``policy.version`` — the version must never move ahead of a
        confirmed offer."""
        self.awaited_step = step
        await self._wait_for(self.step_dir(step) / SENDER_READY_MARKER, cancelled)

    def _ack(self, step: int, **payload) -> None:
        """Acknowledge the offered version — unblocks the waiting trainer. ``payload`` (e.g. the
        transfer group to join) reaches the trainer as JSON; the file appears complete."""
        ready = self.step_dir(step) / RECEIVER_READY_MARKER
        if not payload:
            ready.touch()
            return
        write_marker(ready, json.dumps(payload))

    @abstractmethod
    async def receive(self, step: int) -> None:
        """Acknowledge the offered v{step} and move the engines onto it."""

    def offered_attempt(self, step: int) -> str:
        """The attempt ID of the trainer's offer of v{step} (empty if none)."""
        try:
            return (self.step_dir(step) / SENDER_READY_MARKER).read_text()
        except FileNotFoundError:
            # Not offered yet, or the trainer is resetting the step directory for a new attempt.
            return ""

    async def wait_for_new_offer(self, step: int, attempt: str, newer: Callable[[], bool] | None = None) -> bool:
        """Wait until the trainer offers v{step} again (a new attempt): True. False once ``newer``
        tells that it publishes a newer version instead."""
        while self.offered_attempt(step) in ("", attempt):
            if self.aborted.is_set():
                raise BroadcastAborted(f"Stopped waiting for policy v{step} of a replaced trainer")
            if newer is not None and newer():
                return False
            await asyncio.sleep(0.1)
        return True

    def can_catch_up(self, step: int) -> bool:
        """Whether ``catch_up`` can give servers v{step} without the trainer."""
        return False

    async def catch_up(self, step: int, engines: list[Engine]) -> None:
        """Give rejoining servers the v{step} the others serve."""
        raise NotImplementedError

    async def sync_startup(self, step: int, timeout: float) -> None:
        """Rendezvous with the trainer's startup broadcast of v{step}."""
        await asyncio.wait_for(self.wait_published(step), timeout=timeout)
        await self.receive(step)
