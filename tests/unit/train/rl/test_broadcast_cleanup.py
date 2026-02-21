from pathlib import Path
from types import SimpleNamespace

from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.runs import Progress
from prime_rl.utils.pathing import get_broadcast_dir
from prime_rl.utils.utils import get_step_path


def test_filesystem_broadcast_cleanup_keeps_previous_step(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)

    recorded_calls: list[tuple[Path, int, int, int | None]] = []

    def record_cleanup(path: Path, step: int, keep_last: int, interval_to_keep: int | None) -> None:
        recorded_calls.append((path, step, keep_last, interval_to_keep))

    monkeypatch.setattr("prime_rl.trainer.rl.broadcast.filesystem.maybe_clean", record_cleanup)

    manager = SimpleNamespace(
        used_idxs=[0],
        progress={0: Progress(step=10)},
        get_run_dir=lambda idx: run_dir,
    )
    broadcast = SimpleNamespace(multi_run_manager=manager)

    FileSystemWeightBroadcast.maybe_clean(broadcast, keep_last=1, interval_to_keep=None)

    assert len(recorded_calls) == 1
    cleanup_path, cleanup_step, cleanup_keep_last, cleanup_interval = recorded_calls[0]
    assert cleanup_path == get_broadcast_dir(run_dir)
    assert cleanup_step == 10
    assert cleanup_keep_last == 2
    assert cleanup_interval is None


def test_filesystem_broadcast_rate_limits_writes(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)

    manager = SimpleNamespace(
        ready_to_update_idxs=[0],
        ready_to_update={0: True},
        progress={0: Progress(step=10)},
        get_run_dir=lambda idx: run_dir,
        get_state_dict_for_run=lambda idx: {},
        config={0: SimpleNamespace(model=SimpleNamespace(lora=SimpleNamespace(rank=8, alpha=16)))},
        idx_2_id={0: "run_a"},
        get_orchestrator_config=lambda run_id: object(),
    )

    logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    broadcast = SimpleNamespace(
        logger=logger,
        lora_config=SimpleNamespace(dropout=0.0),
        save_format="safetensors",
        save_sharded=False,
        min_broadcast_interval=1.0,
        _last_broadcast_time=0.0,
        world=SimpleNamespace(is_master=True),
        multi_run_manager=manager,
    )
    broadcast._notify_orchestrator = lambda save_dir: (save_dir / "STABLE").touch()

    monkeypatch.setattr("prime_rl.trainer.rl.broadcast.filesystem.save_state_dict", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.rl.broadcast.filesystem.save_lora_config", lambda *args, **kwargs: None)

    FileSystemWeightBroadcast.broadcast_weights(broadcast, model=SimpleNamespace(), step=10)
    step_10_dir = get_step_path(get_broadcast_dir(run_dir), 10)
    assert manager.ready_to_update[0] is False
    assert (step_10_dir / "STABLE").exists()

    manager.ready_to_update[0] = True
    manager.progress[0] = Progress(step=11)
    FileSystemWeightBroadcast.broadcast_weights(broadcast, model=SimpleNamespace(), step=11)
    step_11_dir = get_step_path(get_broadcast_dir(run_dir), 11)
    assert manager.ready_to_update[0] is True
    assert not step_11_dir.exists()
