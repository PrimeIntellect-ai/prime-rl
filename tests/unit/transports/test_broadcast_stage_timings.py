"""WeightSender timing tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from prime_rl.transports.weights.base import WeightSender

TICK = 0.05


class StubSender(WeightSender):
    def __init__(self, output_dir: Path, wait: float = 0.0, transfer: float = 0.0):
        super().__init__(output_dir, timeout=5)
        self.wait = wait
        self.transfer = transfer
        self.broadcasts: list[int] = []

    def _wait_for_receiver_ready(self, step_dir):
        _sleep(self.wait)

    def _broadcast(self, model, step, step_dir):
        self.broadcasts.append(step)
        _sleep(self.transfer)


def _sleep(seconds: float) -> None:
    if seconds:
        import time

        time.sleep(seconds)


@pytest.fixture
def elapsed():
    import time

    class Timer:
        total = 0.0

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.total = time.perf_counter() - self._start

    return Timer


def test_stages_account_for_the_broadcast(tmp_path, elapsed):
    sender = StubSender(tmp_path, wait=TICK, transfer=TICK)

    with elapsed() as timer:
        timings = sender.broadcast(model=None, step=1)

    assert sum(timings.values()) == pytest.approx(timer.total, abs=0.02)


def test_receiver_wait_is_reported_separately(tmp_path):
    sender = StubSender(tmp_path, wait=10 * TICK)

    timings = sender.broadcast(model=None, step=1)

    assert timings["await_receiver"] >= 10 * TICK
    assert timings["offer"] < 10 * TICK


def test_transfer_is_reported_separately(tmp_path):
    sender = StubSender(tmp_path, transfer=10 * TICK)

    timings = sender.broadcast(model=None, step=1)

    assert timings["transfer"] >= 10 * TICK
    assert timings["offer"] < 10 * TICK
    assert timings["await_receiver"] < 10 * TICK


def test_non_master_reports_only_transfer(tmp_path):
    sender = StubSender(tmp_path, transfer=TICK)
    sender.world = SimpleNamespace(is_master=False, world_size=2)

    timings = sender.broadcast(model=None, step=1)

    assert set(timings) == {"transfer"}


def test_timings_are_per_broadcast(tmp_path):
    sender = StubSender(tmp_path, transfer=10 * TICK)
    slow = sender.broadcast(model=None, step=1)["transfer"]

    sender.transfer = 0.0
    fast = sender.broadcast(model=None, step=2)["transfer"]

    assert fast < slow
    assert sender.broadcasts == [1, 2]
