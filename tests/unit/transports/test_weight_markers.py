"""The weight-broadcast marker handshake, exercised without any hardware.

These are CPU tests on purpose. The handshake is pure filesystem state: the sender raises
markers in a fixed order and blocks on the consumer's acknowledgements. Nothing about it
needs a GPU, yet the transport packages import linux-only dependencies at module level and
so had no unit coverage at all.

They do **not** cover `NIXLWeightSender._broadcast` / `NIXLWeightReceiver.receive` — that
needs NIXL, UCx and a live inference deployment. What they pin down is the shared machinery
those two rely on, including the fifth `.receiver_applied` stage that lets a consumer which
applies the weights itself (rather than a filesystem reader, which loads after `.finished`)
tell the trainer it is done without a fire-and-forget notification (#3742).
"""

import threading
import time
from pathlib import Path

import pytest
from torch import nn

from prime_rl.transports.weights.base import (
    FINISHED_MARKER,
    RECEIVER_APPLIED_MARKER,
    RECEIVER_READY_MARKER,
    SENDER_READY_MARKER,
    STARTED_MARKER,
    WeightSender,
    wait_for_marker,
)

STEP = 3


class PlainSender(WeightSender):
    """A transport that moves the weights inside `_broadcast` and needs no extra ack."""

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        (step_dir / "weights").write_bytes(b"payload")


class ApplyingSender(WeightSender):
    """The NIXL shape: the consumer applies the weights, so the trainer waits for its ack."""

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        wait_for_marker(
            step_dir / RECEIVER_APPLIED_MARKER,
            self.timeout,
            what="the consumer to apply the policy update",
        )


def run_consumer(step_dir: Path, *, applies: bool, observed: list[str], proceed: threading.Event) -> threading.Thread:
    """Stand-in for the orchestrator side of the handshake.

    Records what it observes so the test can prove ordering without trusting timestamps:
    it writes `.receiver_ready` before it starts waiting for `.started`, so seeing
    `.started` implies `.receiver_ready` already existed.
    """

    def wait_for(marker: Path) -> None:
        deadline = time.monotonic() + 10
        while not marker.exists():
            assert time.monotonic() < deadline, f"consumer stalled waiting for {marker.name}"
            time.sleep(0.01)

    def body() -> None:
        wait_for(step_dir / SENDER_READY_MARKER)
        observed.append("sender_ready")
        (step_dir / RECEIVER_READY_MARKER).touch()
        wait_for(step_dir / STARTED_MARKER)
        observed.append("started")
        if applies:
            (step_dir / RECEIVER_APPLIED_MARKER).touch()
            observed.append("receiver_applied")
        proceed.set()

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    return thread


def test_wait_for_marker_returns_once_the_marker_appears(tmp_path):
    marker = tmp_path / ".late"
    threading.Timer(0.15, marker.touch).start()

    started = time.monotonic()
    wait_for_marker(marker, timeout=10, what="a late marker")

    assert marker.exists()
    assert time.monotonic() - started >= 0.1, "should have actually waited for the marker"


def test_wait_for_marker_times_out_and_names_what_it_waited_for(tmp_path):
    marker = tmp_path / ".never"

    with pytest.raises(TimeoutError) as excinfo:
        wait_for_marker(marker, timeout=0, what="the consumer to acknowledge")

    message = str(excinfo.value)
    assert str(marker) in message
    assert "the consumer to acknowledge" in message


def test_sender_walks_the_handshake_in_order(tmp_path):
    sender = PlainSender(tmp_path, timeout=10)
    step_dir = sender.step_dir(STEP)
    observed: list[str] = []
    proceed = threading.Event()
    consumer = run_consumer(step_dir, applies=False, observed=observed, proceed=proceed)

    sender.broadcast(nn.Linear(2, 2), STEP)
    consumer.join(timeout=10)
    assert proceed.is_set(), "the consumer never got through the handshake"

    assert (step_dir / SENDER_READY_MARKER).exists()
    assert (step_dir / RECEIVER_READY_MARKER).exists()
    assert (step_dir / STARTED_MARKER).exists()
    assert (step_dir / FINISHED_MARKER).exists()
    # Implies `.receiver_ready` was raised before `.started`.
    assert observed == ["sender_ready", "started"]


def test_sender_fails_instead_of_hanging_when_the_receiver_never_joins(tmp_path):
    sender = PlainSender(tmp_path, timeout=0)
    step_dir = sender.step_dir(STEP)

    with pytest.raises(TimeoutError):
        sender.broadcast(nn.Linear(2, 2), STEP)

    assert (step_dir / SENDER_READY_MARKER).exists()
    assert not (step_dir / STARTED_MARKER).exists(), "must not start the transfer without a receiver"


def test_receiver_applied_gates_the_finished_marker(tmp_path):
    """A consumer that applies the weights itself acks before the trainer commits."""
    sender = ApplyingSender(tmp_path, timeout=10)
    step_dir = sender.step_dir(STEP)
    observed: list[str] = []
    proceed = threading.Event()
    consumer = run_consumer(step_dir, applies=True, observed=observed, proceed=proceed)

    sender.broadcast(nn.Linear(2, 2), STEP)
    consumer.join(timeout=10)
    assert proceed.is_set(), "the consumer never got through the handshake"

    assert (step_dir / RECEIVER_APPLIED_MARKER).exists()
    assert (step_dir / FINISHED_MARKER).exists()
    assert observed == ["sender_ready", "started", "receiver_applied"]


def test_applying_sender_fails_when_the_consumer_never_applies(tmp_path):
    """The trainer must not commit a version the consumer never confirmed.

    The timeout has to be long enough to clear the receiver-ready gate, otherwise the
    failure lands on the wrong stage and the assertion proves nothing.
    """
    sender = ApplyingSender(tmp_path, timeout=1)
    step_dir = sender.step_dir(STEP)
    observed: list[str] = []
    proceed = threading.Event()
    consumer = run_consumer(step_dir, applies=False, observed=observed, proceed=proceed)

    with pytest.raises(TimeoutError):
        sender.broadcast(nn.Linear(2, 2), STEP)

    assert proceed.wait(timeout=10), "the consumer never got past the receiver-ready gate"
    consumer.join(timeout=10)
    # It got all the way to the transfer, and still refused to commit.
    assert (step_dir / STARTED_MARKER).exists()
    assert not (step_dir / RECEIVER_APPLIED_MARKER).exists()
    assert not (step_dir / FINISHED_MARKER).exists()
