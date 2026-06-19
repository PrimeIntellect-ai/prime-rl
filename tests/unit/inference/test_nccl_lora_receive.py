import threading
import time
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker


def _bare_worker():
    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.device = torch.device("cpu")
    return worker


class FakeLoRAModel:
    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id,
        tensors,
        peft_helper,
        device,
        dtype,
        model_vocab_size,
        weights_mapper,
        skip_prefixes,
    ):
        model = SimpleNamespace(
            id=lora_model_id,
            tensors=tensors,
            peft_rank=peft_helper.r,
            device=device,
            dtype=dtype,
            model_vocab_size=model_vocab_size,
            weights_mapper=weights_mapper,
            skip_prefixes=skip_prefixes,
        )
        return model


class FakeAdapterManager:
    def __init__(self):
        self.model = SimpleNamespace(hf_to_vllm_mapper=None, lora_skip_prefixes=None)
        self.capacity = 8
        self.adapters = {}
        self.removed = []
        self.activated = []

    def __len__(self):
        return len(self.adapters)

    def remove_adapter(self, lora_id):
        self.removed.append(lora_id)
        self.adapters.pop(lora_id, None)

    def add_adapter(self, lora_model):
        self.adapters[lora_model.id] = lora_model
        return True

    def activate_adapter(self, lora_id):
        self.activated.append(lora_id)
        return True

    def remove_oldest_adapter(self):
        self.adapters.pop(next(iter(self.adapters)))


def _worker_with_fake_lora_manager():
    worker = _bare_worker()
    worker.nccl_broadcast_receiver = SimpleNamespace(communicator=object())
    adapter_manager = FakeAdapterManager()
    lora_manager = SimpleNamespace(
        _adapter_manager=adapter_manager,
        _lora_model_cls=FakeLoRAModel,
        lora_config=SimpleNamespace(max_lora_rank=8, lora_dtype=torch.float32),
        vocab_size=32000,
    )
    worker.model_runner = SimpleNamespace(lora_manager=lora_manager)
    return worker, adapter_manager


def _lora_header(step=7, lora_name="debater_a", lora_int_id=1):
    return {
        "step": step,
        "num_chunks": 1,
        "adapters": [
            {
                "lora_name": lora_name,
                "lora_int_id": lora_int_id,
                "peft_config": {
                    "r": 4,
                    "lora_alpha": 8,
                    "target_modules": ["q_proj"],
                    "bias": "none",
                },
            }
        ],
    }


def test_lora_arm_returns_while_receive_thread_is_blocked(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def receive_lora_update(self, step, header_expectation):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(NCCLWeightUpdateWorker, "receive_lora_update", receive_lora_update)

    worker = _bare_worker()
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

    worker = _bare_worker()
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

    worker = _bare_worker()
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

    worker = _bare_worker()
    worker.arm_lora_receive(7, {"adapters": [{"lora_name": "debater_a"}]})

    with pytest.raises(RuntimeError, match="not 8"):
        worker.wait_lora_receive(8)
    worker.wait_lora_receive(7)


def test_lora_receive_commits_in_memory_adapter(monkeypatch):
    worker, adapter_manager = _worker_with_fake_lora_manager()
    tensors = {"model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(4, 8)}

    monkeypatch.setattr(NCCLWeightUpdateWorker, "_receive_lora_object", lambda self, communicator: _lora_header())
    monkeypatch.setattr(
        NCCLWeightUpdateWorker, "_receive_lora_chunk_to_host", lambda self, communicator, chunk_idx: dict(tensors)
    )

    worker.receive_lora_update(
        7,
        {"step": 7, "adapters": [{"lora_name": "debater_a", "lora_int_id": 1}]},
    )

    assert adapter_manager.removed == [1]
    assert adapter_manager.activated == [1]
    assert adapter_manager.adapters[1].peft_rank == 4
    assert adapter_manager.adapters[1].tensors == tensors


def test_lora_receive_rejects_header_mismatch_before_tensor_receive(monkeypatch):
    worker, _ = _worker_with_fake_lora_manager()

    monkeypatch.setattr(
        NCCLWeightUpdateWorker, "_receive_lora_object", lambda self, communicator: _lora_header(lora_name="wrong")
    )

    def receive_chunk(self, communicator, chunk_idx):
        raise AssertionError("should not receive tensors after a bad header")

    monkeypatch.setattr(NCCLWeightUpdateWorker, "_receive_lora_chunk_to_host", receive_chunk)

    with pytest.raises(RuntimeError, match="did not match expected"):
        worker.receive_lora_update(
            7,
            {"step": 7, "adapters": [{"lora_name": "debater_a", "lora_int_id": 1}]},
        )
