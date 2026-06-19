from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.worker import nccl as worker_nccl
from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker
from prime_rl.trainer.rl.broadcast import nccl as broadcast_nccl
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


def test_nccl_worker_init_broadcaster_is_idempotent(monkeypatch):
    created = []

    class FakeReceiver:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(worker_nccl, "NCCLWeightBroadcastReceiver", FakeReceiver)

    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.device = SimpleNamespace(index=2)

    worker.init_broadcaster("localhost", 29501, 4, 8, 10, True)
    worker.init_broadcaster("localhost", 29501, 4, 8, 10, True)

    assert len(created) == 1
    assert created[0]["rank"] == 7
    assert created[0]["world_size"] == 9
    assert worker.quantize_in_weight_transfer is True


def test_nccl_worker_init_broadcaster_rejects_mismatch(monkeypatch):
    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(worker_nccl, "NCCLWeightBroadcastReceiver", FakeReceiver)

    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.device = SimpleNamespace(index=0)
    worker.init_broadcaster("localhost", 29501, 0, 8, 10, False)

    with pytest.raises(RuntimeError, match="already initialized"):
        worker.init_broadcaster("localhost", 29502, 0, 8, 10, False)


def test_nccl_broadcast_maybe_clean_removes_old_marker_dir(tmp_path):
    broadcast = object.__new__(NCCLWeightBroadcast)
    run_dir = tmp_path / "run_default"
    old_step_dir = get_step_path(get_broadcast_dir(run_dir), 1)
    old_step_dir.mkdir(parents=True)

    class FakeRunManager:
        used_idxs = [0]
        progress = {0: SimpleNamespace(step=3)}

        def get_run_dir(self, idx: int):
            assert idx == 0
            return run_dir

    broadcast.multi_run_manager = FakeRunManager()

    broadcast.maybe_clean(interval_to_keep=None)

    assert not old_step_dir.exists()


def test_nccl_lora_broadcast_clears_ready_flag_after_barrier_and_wait(tmp_path, monkeypatch):
    """The LoRA NCCL path must mirror full-FT ordering: notify -> barrier -> wait -> clear -> send.

    Both transports withhold batches for a run while ``ready_to_update`` is set, so a LoRA
    broadcast that never clears the flag starves the packer after one update (deadlock).
    """
    events: list[str] = []

    broadcast = object.__new__(NCCLWeightBroadcast)
    broadcast.logger = SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None)
    broadcast.world = SimpleNamespace(is_master=True, world_size=1)
    broadcast.lora_config = SimpleNamespace(dropout=0.0)
    broadcast.ready_timeout = 5

    run_dir = tmp_path / "run_default"

    class FakeRunManager:
        used_idxs = [0]
        ready_to_update = {0: True}
        progress = {0: SimpleNamespace(step=3)}

        def get_run_dir(self, idx: int):
            assert idx == 0
            return run_dir

        def get_state_dict_for_run(self, idx: int):
            return {"w.lora_A": torch.zeros(2, 2, dtype=torch.float32)}

    manager = FakeRunManager()
    broadcast.multi_run_manager = manager

    real_barrier = NCCLWeightBroadcast._sync_trainer_ranks

    def fake_barrier(self):
        events.append("barrier")
        real_barrier(self)  # world_size == 1 -> no-op

    monkeypatch.setattr(NCCLWeightBroadcast, "_sync_trainer_ranks", fake_barrier)

    def fake_wait(path, **kwargs):
        events.append("wait")
        assert path.name == "NCCL_READY"
        assert (path.parent / "STABLE").exists()  # master notified before any rank waits
        assert manager.ready_to_update[0] is True  # flag is cleared only after the wait

    monkeypatch.setattr(broadcast_nccl, "sync_wait_for_path", fake_wait)
    monkeypatch.setattr(
        NCCLWeightBroadcast, "_build_lora_adapter_header", lambda self, model, idx: {"lora_name": "test"}
    )

    def fake_send(step, adapter_header, state_dict):
        events.append("send")
        assert step == 3
        assert manager.ready_to_update[0] is False  # cleared before the send, like full-FT
        assert all(value.dtype == torch.bfloat16 for value in state_dict.values())

    broadcast.nccl_broadcast_sender = SimpleNamespace(dtype=torch.bfloat16, broadcast_lora_update=fake_send)

    broadcast._broadcast_lora_adapters(model=object())

    assert events == ["barrier", "wait", "send"]
    assert manager.ready_to_update[0] is False
