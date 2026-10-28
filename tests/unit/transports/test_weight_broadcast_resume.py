import asyncio
import threading
import time

import pytest
import torch.nn as nn

import prime_rl.transports.weights.base as weights_base
from prime_rl.transports.weights.base import (
    FINISHED_MARKER,
    RECEIVER_READY_MARKER,
    SENDER_READY_MARKER,
    STARTED_MARKER,
    WeightReceiver,
    WeightSender,
    prune_broadcasts_beyond,
    prune_broadcasts_from,
)
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path

ALL_MARKERS = (SENDER_READY_MARKER, RECEIVER_READY_MARKER, STARTED_MARKER, FINISHED_MARKER)
# "all": the crashed attempt finished broadcasting v{R} (crash in step R+1/R+2, after the step R
# checkpoint). "offer": the crashed attempt died waiting for the receiver of its startup offer.
LEFTOVER_MARKERS = {"all": ALL_MARKERS, "offer": (SENDER_READY_MARKER,)}


def _write_leftovers(output_dir, steps, markers):
    for step in steps:
        step_dir = get_step_path(get_broadcast_dir(output_dir), step)
        step_dir.mkdir(parents=True)
        for marker in markers:
            (step_dir / marker).touch()


@pytest.mark.parametrize("markers", LEFTOVER_MARKERS.values(), ids=LEFTOVER_MARKERS.keys())
@pytest.mark.parametrize("resume_step", [0, 40])
def test_prune_broadcasts_from_drops_startup_version_and_beyond(tmp_path, markers, resume_step):
    _write_leftovers(tmp_path, [resume_step - 1, resume_step, resume_step + 1], markers)
    prune_broadcasts_from(tmp_path, resume_step)
    broadcast_dir = get_broadcast_dir(tmp_path)
    assert not get_step_path(broadcast_dir, resume_step).exists()
    assert not get_step_path(broadcast_dir, resume_step + 1).exists()
    if resume_step > 0:
        assert get_step_path(broadcast_dir, resume_step - 1).exists()


def test_prune_broadcasts_from_without_broadcast_dir(tmp_path):
    prune_broadcasts_from(tmp_path, 0)


class _FakeSender(WeightSender):
    def __init__(self, output_dir, timeout, transferred: threading.Event):
        super().__init__(output_dir, timeout)
        self.transferred = transferred

    def _broadcast(self, model, step, step_dir):
        self.transferred.set()


class _FakeAdminPlane:
    """Stands in for the engines' NCCL receive RPC: acks once paused, then waits for the
    trainer's transfer (which, for a stale offer, never comes)."""

    def __init__(self, transferred: threading.Event, timeout: float):
        self.transferred = transferred
        self.timeout = timeout

    async def update_weights(self, weight_dir, *, transport, step, on_paused):
        on_paused()
        deadline = asyncio.get_running_loop().time() + self.timeout
        while not self.transferred.is_set():
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("update_weights RPC timed out")
            await asyncio.sleep(0.01)


class _FakeReceiver(WeightReceiver):
    def __init__(self, broadcast_dir, admin_plane, trainer_joined: threading.Event):
        super().__init__(broadcast_dir, config=None, admin_plane=admin_plane, model_name="model")
        self.trainer_joined = trainer_joined

    async def initialize(self) -> None:
        # The NCCL rendezvous (/init_broadcaster) returns only once the trainer master has
        # created the communicator in setup_weight_sender.
        assert await asyncio.to_thread(self.trainer_joined.wait, 5)

    async def receive(self, step: int) -> None:
        await self.admin_plane.update_weights(
            self.step_dir(step), transport="nccl", step=step, on_paused=lambda: self._ack(step)
        )


@pytest.mark.parametrize("markers", LEFTOVER_MARKERS.values(), ids=LEFTOVER_MARKERS.keys())
@pytest.mark.parametrize("resume_step", [0, 40])
def test_startup_broadcast_survives_stale_offer_of_crashed_attempt(tmp_path, monkeypatch, markers, resume_step):
    """A restarted attempt must not ack the previous attempt's offer of its startup version.
    Models the trainer's startup order: prune, join the rendezvous (setup_weight_sender), load
    the checkpoint (slow), then broadcast; the consumer syncs as soon as its rendezvous returns."""
    monkeypatch.setattr(weights_base, "wait_for_master", lambda key, timeout: None)
    _write_leftovers(tmp_path, [resume_step - 1, resume_step, resume_step + 1], markers)
    trainer_joined, transferred = threading.Event(), threading.Event()
    trainer_result = {}

    def trainer():
        prune_broadcasts_from(tmp_path, resume_step)
        trainer_joined.set()  # setup_weight_sender
        time.sleep(0.3)  # checkpoint load, while the consumer is already syncing
        prune_broadcasts_beyond(tmp_path, resume_step)
        try:
            _FakeSender(tmp_path, timeout=3, transferred=transferred).broadcast(nn.Linear(1, 1), resume_step)
            trainer_result["ok"] = True
        except TimeoutError as e:
            trainer_result["error"] = e

    async def orchestrator():
        receiver = _FakeReceiver(get_broadcast_dir(tmp_path), _FakeAdminPlane(transferred, timeout=3), trainer_joined)
        await receiver.initialize()
        await receiver.sync_startup(resume_step, timeout=5)

    thread = threading.Thread(target=trainer)
    thread.start()
    asyncio.run(orchestrator())
    thread.join()
    assert trainer_result == {"ok": True}
    step_dir = get_step_path(get_broadcast_dir(tmp_path), resume_step)
    assert all((step_dir / marker).exists() for marker in ALL_MARKERS)
