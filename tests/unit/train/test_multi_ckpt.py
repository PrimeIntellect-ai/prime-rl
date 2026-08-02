from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import prime_rl.trainer.multi_ckpt as multi_ckpt
from prime_rl.orchestrator.utils import get_weight_dir
from prime_rl.trainer.runs import Progress
from prime_rl.utils.pathing import get_all_ckpt_steps


class FakeCheckpointManager:
    def __init__(self, output_dir: Path, _config) -> None:
        self.ckpt_dir = output_dir / "checkpoints"
        self.ckpt_steps = get_all_ckpt_steps(self.ckpt_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}" / "trainer"

    def mark_stable(self, step: int) -> None:
        step_dir = self.ckpt_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "STABLE").touch()


class FakeMultiRunManager:
    def __init__(self, output_dir: Path) -> None:
        self.max_runs = 1
        self.used_idxs = [0]
        self.idx_2_id = {0: "run"}
        self.config = {0: SimpleNamespace(ckpt=SimpleNamespace(interval=1, keep_last=None, keep_interval=None))}
        self.progress = {0: Progress(step=2)}
        self.lora_num_tokens = torch.zeros(1, dtype=torch.int32)
        self.output_dir = output_dir

    def register_deletion_hook(self, _hook) -> None:
        pass

    def register_creation_hook(self, _hook) -> None:
        pass

    def get_named_parameters_for_run(self, _idx: int) -> list[tuple[str, torch.nn.Parameter]]:
        return []

    def get_run_dir(self, _idx: int) -> Path:
        return self.output_dir / "run"

    def get_orchestrator_config(self, _run_id: str) -> object:
        return object()


def make_checkpoint_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[multi_ckpt.MultiCheckpointManager, Path]:
    run_manager = FakeMultiRunManager(tmp_path)
    monkeypatch.setattr(multi_ckpt, "get_multi_run_manager", lambda: run_manager)
    monkeypatch.setattr(multi_ckpt, "get_world", lambda: SimpleNamespace(rank=0, is_master=True))
    monkeypatch.setattr(multi_ckpt, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(multi_ckpt.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(multi_ckpt.dist, "barrier", lambda: None)

    checkpoint_manager = multi_ckpt.MultiCheckpointManager(tmp_path)
    checkpoint_manager.managers[0] = checkpoint_manager._maybe_create_manager(0)
    return checkpoint_manager, run_manager.get_run_dir(0)


def test_multi_checkpoint_skips_stable_marker_when_broadcast_weights_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_manager, run_dir = make_checkpoint_manager(tmp_path, monkeypatch)
    optimizer = SimpleNamespace(optimizers=[None])
    scheduler = SimpleNamespace(schedulers=[None])

    checkpoint_manager.save(optimizer, scheduler)

    step_dir = run_dir / "checkpoints" / "step_1"
    assert (step_dir / "trainer" / "rank_0.pt").exists()
    assert not (step_dir / "STABLE").exists()
    assert checkpoint_manager.managers[0].ckpt_steps == []
    with pytest.raises(FileNotFoundError):
        get_weight_dir(run_dir, 1)


def test_multi_checkpoint_ignores_a_stable_checkpoint_without_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    incomplete_step_dir = run_dir / "checkpoints" / "step_1"
    incomplete_step_dir.mkdir(parents=True)
    (incomplete_step_dir / "STABLE").touch()

    checkpoint_manager, _ = make_checkpoint_manager(tmp_path, monkeypatch)

    assert checkpoint_manager.managers[0].ckpt_steps == []


def test_multi_checkpoint_marks_stable_after_copying_broadcast_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_manager, run_dir = make_checkpoint_manager(tmp_path, monkeypatch)
    broadcast_dir = run_dir / "broadcasts" / "step_1"
    broadcast_dir.mkdir(parents=True)
    (broadcast_dir / "model.safetensors").write_bytes(b"weights")
    (broadcast_dir / "STABLE").touch()
    optimizer = SimpleNamespace(optimizers=[None])
    scheduler = SimpleNamespace(schedulers=[None])

    checkpoint_manager.save(optimizer, scheduler)

    step_dir = run_dir / "checkpoints" / "step_1"
    assert (step_dir / "STABLE").exists()
    assert (step_dir / "weight" / "model.safetensors").read_bytes() == b"weights"
    assert checkpoint_manager.managers[0].ckpt_steps == [1]
    assert get_weight_dir(run_dir, 1) == step_dir / "weight"


def test_multi_checkpoint_skips_an_unstable_broadcast_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    broadcast_dir = run_dir / "broadcasts" / "step_1"
    broadcast_dir.mkdir(parents=True)
    (broadcast_dir / "model.safetensors").write_bytes(b"partial weights")
    checkpoint_manager, run_dir = make_checkpoint_manager(tmp_path, monkeypatch)
    optimizer = SimpleNamespace(optimizers=[None])
    scheduler = SimpleNamespace(schedulers=[None])

    checkpoint_manager.save(optimizer, scheduler)

    step_dir = run_dir / "checkpoints" / "step_1"
    assert (step_dir / "trainer" / "rank_0.pt").exists()
    assert not (step_dir / "STABLE").exists()
    assert not (step_dir / "weight").exists()
    assert checkpoint_manager.managers[0].ckpt_steps == []


def test_multi_checkpoint_does_not_publish_when_another_rank_state_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    broadcast_dir = run_dir / "broadcasts" / "step_1"
    broadcast_dir.mkdir(parents=True)
    (broadcast_dir / "model.safetensors").write_bytes(b"weights")
    (broadcast_dir / "STABLE").touch()
    checkpoint_manager, run_dir = make_checkpoint_manager(tmp_path, monkeypatch)

    def report_failed_rank(status: torch.Tensor, *_args, **_kwargs) -> None:
        status.zero_()

    monkeypatch.setattr(multi_ckpt.dist, "all_reduce", report_failed_rank)

    optimizer = SimpleNamespace(optimizers=[None])
    scheduler = SimpleNamespace(schedulers=[None])
    checkpoint_manager.save(optimizer, scheduler)

    step_dir = run_dir / "checkpoints" / "step_1"
    assert not (step_dir / "STABLE").exists()
    assert not (step_dir / "weight").exists()
    assert checkpoint_manager.managers[0].ckpt_steps == []


def test_multi_checkpoint_retries_an_incomplete_weight_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    incomplete_weight_dir = run_dir / "checkpoints" / "step_1" / "weight"
    incomplete_weight_dir.mkdir(parents=True)
    (incomplete_weight_dir / "stale.safetensors").write_bytes(b"stale")
    broadcast_dir = run_dir / "broadcasts" / "step_1"
    broadcast_dir.mkdir(parents=True)
    (broadcast_dir / "model.safetensors").write_bytes(b"weights")
    (broadcast_dir / "STABLE").touch()

    checkpoint_manager, run_dir = make_checkpoint_manager(tmp_path, monkeypatch)
    optimizer = SimpleNamespace(optimizers=[None])
    scheduler = SimpleNamespace(schedulers=[None])
    checkpoint_manager.save(optimizer, scheduler)

    step_dir = run_dir / "checkpoints" / "step_1"
    assert (step_dir / "STABLE").exists()
    assert (step_dir / "weight" / "model.safetensors").read_bytes() == b"weights"
    assert not (step_dir / "weight" / "stale.safetensors").exists()
    assert checkpoint_manager.managers[0].ckpt_steps == [1]
