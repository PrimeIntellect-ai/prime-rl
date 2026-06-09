from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm.worker import nccl as worker_nccl
from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker
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
