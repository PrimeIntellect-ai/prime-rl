import threading
import time
from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker


def test_lora_arm_returns_while_receive_thread_is_blocked(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def receive_lora_update(self, step, header_expectation):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(NCCLWeightUpdateWorker, "receive_lora_update", receive_lora_update)

    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.arm_lora_receive(7, {"adapters": [{"lora_name": "debater_a"}]})

    assert entered.wait(timeout=1)
    assert worker._lora_receive_state == {"step": 7, "status": "receiving", "error": None}
    assert worker._lora_receive_thread.is_alive()

    release.set()
    assert worker.wait_lora_receive(7) == {"status": "ok", "step": 7}


def test_lora_arm_rejects_previous_cycle_still_alive(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def receive_lora_update(self, step, header_expectation):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(NCCLWeightUpdateWorker, "receive_lora_update", receive_lora_update)

    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.arm_lora_receive(7, {"adapters": [{"lora_name": "debater_a"}]})
    assert entered.wait(timeout=1)

    with pytest.raises(RuntimeError, match="still in flight"):
        worker.arm_lora_receive(8, {"adapters": [{"lora_name": "debater_a"}]})

    release.set()
    worker.wait_lora_receive(7)


def test_lora_wait_propagates_receive_error(monkeypatch):
    def receive_lora_update(self, step, header_expectation):
        raise ValueError("bad header")

    monkeypatch.setattr(NCCLWeightUpdateWorker, "receive_lora_update", receive_lora_update)

    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.arm_lora_receive(7, {"adapters": [{"lora_name": "debater_a"}]})

    deadline = time.monotonic() + 1
    while worker._lora_receive_state["status"] != "error" and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="bad header"):
        worker.wait_lora_receive(7)


def test_lora_wait_rejects_wrong_step(monkeypatch):
    def receive_lora_update(self, step, header_expectation):
        return None

    monkeypatch.setattr(NCCLWeightUpdateWorker, "receive_lora_update", receive_lora_update)

    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.device = SimpleNamespace(index=0)
    worker.arm_lora_receive(7, {"adapters": [{"lora_name": "debater_a"}]})

    with pytest.raises(RuntimeError, match="not 8"):
        worker.wait_lora_receive(8)
    worker.wait_lora_receive(7)
