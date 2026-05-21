from types import SimpleNamespace

from prime_rl.trainer.rl.broadcast import nccl
from prime_rl.trainer.rl.broadcast.nccl import NCCL_READY_MARKER, NCCLWeightBroadcast


class FakeMultiRunManager:
    def __init__(self, run_dir, step: int = 11):
        self.used_idxs = [0]
        self.ready_to_update = [True]
        self.progress = {0: SimpleNamespace(step=step)}
        self._run_dir = run_dir

    def get_run_dir(self, idx: int):
        assert idx == 0
        return self._run_dir


def test_notify_orchestrator_removes_stale_ready_marker_before_stable(tmp_path):
    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcast.multi_run_manager = FakeMultiRunManager(tmp_path)
    broadcast.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    save_dir = tmp_path / "broadcasts" / "step_11"
    save_dir.mkdir(parents=True)
    (save_dir / NCCL_READY_MARKER).touch()
    stable_file = save_dir / "STABLE"
    stable_file.touch()

    broadcast._notify_orchestrator([(0, save_dir)])

    assert not (save_dir / NCCL_READY_MARKER).exists()
    assert stable_file.exists()


def test_master_clears_stale_state_then_barriers_before_waiting_for_inference(tmp_path):
    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcast.world = SimpleNamespace(is_master=True, world_size=2)
    broadcast.multi_run_manager = FakeMultiRunManager(tmp_path)
    broadcast.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    calls = []

    broadcast._notify_orchestrator = lambda pending_runs: calls.append("notify")
    broadcast._wait_for_nccl_ready = lambda pending_runs: calls.append("wait_nccl")
    broadcast._sync_trainer_ranks = lambda: calls.append("barrier")
    broadcast.nccl_broadcast_sender = SimpleNamespace(broadcast_weights=lambda model, step: calls.append("broadcast"))

    broadcast.broadcast_weights(model=None, step=11)

    assert calls == ["notify", "barrier", "wait_nccl", "broadcast"]
    assert broadcast.multi_run_manager.ready_to_update[0] is False


def test_non_master_waits_for_nccl_ready_after_trainer_barrier(tmp_path):
    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcast.world = SimpleNamespace(is_master=False, world_size=2)
    broadcast.multi_run_manager = FakeMultiRunManager(tmp_path)
    broadcast.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    calls = []

    broadcast._sync_trainer_ranks = lambda: calls.append("barrier")
    broadcast._wait_for_nccl_ready = lambda pending_runs: calls.append("wait_nccl")
    broadcast.nccl_broadcast_sender = SimpleNamespace(broadcast_weights=lambda model, step: calls.append("broadcast"))

    broadcast.broadcast_weights(model=None, step=11)

    assert calls == ["barrier", "wait_nccl", "broadcast"]
    assert broadcast.multi_run_manager.ready_to_update[0] is False


def test_sync_trainer_ranks_uses_distributed_barrier(monkeypatch):
    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcast.world = SimpleNamespace(world_size=2)
    calls = []

    monkeypatch.setattr(nccl.dist, "is_available", lambda: True)
    monkeypatch.setattr(nccl.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(nccl.dist, "barrier", lambda: calls.append("barrier"))

    broadcast._sync_trainer_ranks()

    assert calls == ["barrier"]
