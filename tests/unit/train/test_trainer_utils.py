from pathlib import Path
from types import SimpleNamespace

from prime_rl.trainer import utils as trainer_utils


class StaleCheckpointPath:
    def __init__(self, path: Path):
        self.path = path

    def mkdir(self, *args, **kwargs):
        raise FileExistsError

    def exists(self):
        return False

    def __str__(self):
        return str(self.path)


def test_get_ckpt_disk_metrics_tolerates_stale_file_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer_utils, "get_world", lambda: SimpleNamespace(is_master=True))
    monkeypatch.setattr(
        trainer_utils,
        "get_ckpt_dir",
        lambda output_dir: StaleCheckpointPath(output_dir / "checkpoints"),
    )

    metrics = trainer_utils.get_ckpt_disk_metrics(tmp_path)

    assert metrics["system/ckpt_disk_total_gib"] > 0
    assert metrics["system/ckpt_disk_free_gib"] >= 0
    assert 0 <= metrics["system/ckpt_disk_free_ratio"] <= 1


def test_get_ckpt_disk_metrics_skips_mkdir_on_non_master(monkeypatch, tmp_path):
    class NonMasterCheckpointPath(StaleCheckpointPath):
        def mkdir(self, *args, **kwargs):
            raise AssertionError("non-master ranks must not mkdir the shared checkpoint dir")

    monkeypatch.setattr(trainer_utils, "get_world", lambda: SimpleNamespace(is_master=False))
    monkeypatch.setattr(
        trainer_utils,
        "get_ckpt_dir",
        lambda output_dir: NonMasterCheckpointPath(output_dir / "checkpoints"),
    )

    metrics = trainer_utils.get_ckpt_disk_metrics(tmp_path)

    assert metrics["system/ckpt_disk_total_gib"] > 0
