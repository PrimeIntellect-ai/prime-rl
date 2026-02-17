from pathlib import Path
from types import SimpleNamespace

from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.runs import Progress
from prime_rl.utils.pathing import get_broadcast_dir


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
