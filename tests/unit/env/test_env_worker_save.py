"""An env worker saves its running rollouts as restore files, and tells whose run each
restore point continues."""

import asyncio
from types import SimpleNamespace

from verifiers.v1.env import RunSlot

from prime_rl.env import worker
from prime_rl.env.types import SaveRequest
from prime_rl.env.worker import EnvWorker, read_restore
from prime_rl.utils import pathing


def point(trace_id: str) -> dict:
    """A saved trace: the restore fields are on the entry of its checkpoint."""
    checkpoint = {"id": f"c-{trace_id}", "snapshot": {"id": f"s-{trace_id}"}, "restore": {}}
    return {"id": trace_id, "checkpoints": [checkpoint]}


def slot(episode_id: str, *trace_ids: str) -> RunSlot:
    return RunSlot(task=None, id=episode_id, traces=[SimpleNamespace(id=trace_id) for trace_id in trace_ids])


def test_save_maps_restore_points_to_runs_and_deletes_the_snapshots_of_the_rest(tmp_path, monkeypatch):
    discarded: list[str] = []

    class Backend:
        @staticmethod
        async def discard(snapshots) -> list[str]:
            discarded.extend(snapshot.id for snapshot in snapshots)
            return [snapshot.id for snapshot in snapshots]

    async def save_all(skipped: dict[str, int], *, hold: bool) -> list[dict]:
        # t9's run ended while the checkpoints were taken; t2 is one trace of a multi-agent episode.
        assert hold
        skipped["the 'subprocess' runtime takes no memory snapshots"] = 2
        return [point("t1"), point("t2"), point("t9"), point("../t4")]

    monkeypatch.setattr(worker, "save_all", save_all)
    monkeypatch.setattr(worker, "PrimeBackend", Backend)
    monkeypatch.setattr(pathing, "_save_root", str(tmp_path.resolve()))
    env_worker = EnvWorker.__new__(EnvWorker)
    env_worker.taskset_id = "tasks"
    env_worker._warned_skips = set()
    env_worker._slots = {
        "r1": slot("e1", "t1"),
        "r2": slot("e2", "t2", "t3"),
        "r4": slot("e4", "../t4"),
        "r5": slot("e5"),
    }

    directory = tmp_path / "rollouts"
    response = asyncio.run(env_worker._save(SaveRequest(dir=str(directory))))

    assert set(response.restores) == {"r1"}
    ref = response.restores["r1"]
    assert ref.trace_id == "t1" and ref.snapshot == {"id": "s-t1"}
    assert ref.path == str(directory / "t1.restore")
    assert [file.name for file in directory.iterdir()] == ["t1.restore"]
    episode_id, restored = read_restore(ref.path)
    assert episode_id == "e1" and restored == point("t1")
    assert discarded == ["s-t2", "s-t9", "s-../t4"]
    # One reason per running rollout without a restore point.
    assert sorted(response.skipped) == [
        *["its episode has more than one trace, or it ended during the save"] * 3,
        *["the 'subprocess' runtime takes no memory snapshots"] * 2,
    ]


def test_an_unreadable_restore_file_starts_the_rollout_over(tmp_path, monkeypatch):
    monkeypatch.setattr(pathing, "_save_root", str(tmp_path.resolve()))
    assert read_restore(str(tmp_path / "missing.restore")) is None
    (tmp_path / "broken.restore").write_bytes(b"not a pickle")
    assert read_restore(str(tmp_path / "broken.restore")) is None


def test_a_restore_file_outside_the_run_directory_is_not_read(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(pathing, "_save_root", str((tmp_path / "run").resolve()))
    worker.write_restores(tmp_path, [("e1", point("t1"))])
    assert read_restore(str(tmp_path / "t1.restore")) is None
    assert read_restore(str(tmp_path / "run" / ".." / "t1.restore")) is None
