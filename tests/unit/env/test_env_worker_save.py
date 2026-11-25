"""An env worker saves its running rollouts as restore files, and tells whose run each
restore point continues."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from verifiers.v1.env import RunSlot

from prime_rl.env import worker
from prime_rl.env.types import SaveRequest
from prime_rl.env.worker import EnvWorker, read_restore
from prime_rl.utils import pathing


def point(trace_id: str) -> dict:
    """A saved trace: the session data is on the entry of its checkpoint."""
    checkpoint = {"id": f"c-{trace_id}", "snapshot": {"id": f"s-{trace_id}"}, "session": {}}
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

    async def save_all(skipped: dict[str, int], *, hold: bool, watchdog: bool) -> list[dict]:
        # t9's run ended while the checkpoints were taken; t2 is one trace of a multi-agent episode.
        assert hold and not watchdog
        skipped["the 'subprocess' runtime takes no memory snapshots"] = 2
        durable = point("done")
        durable["checkpoints"][0]["time"] = {"end": 1.0}
        return [point("t1"), point("t2"), point("t9"), point("failed"), point("../t4"), durable]

    failure_seen = asyncio.Event()

    class Client:
        async def wait_for_checkpoint(self, snapshot_id, *, timeout_seconds):
            if snapshot_id == "s-failed":
                failure_seen.set()
                raise RuntimeError("upload failed")
            assert snapshot_id == "s-t1"
            await asyncio.wait_for(failure_seen.wait(), timeout=1)
            assert not list(directory.glob("*.restore"))

    @asynccontextmanager
    async def leased_client():
        yield Client()

    monkeypatch.setattr(worker, "leased_client", leased_client)
    monkeypatch.setattr(worker, "save_all", save_all)
    armed: list[bool] = []
    monkeypatch.setattr(worker, "arm_hold_watchdogs", lambda: armed.append(True))
    monkeypatch.setattr(worker, "PrimeBackend", Backend)
    monkeypatch.setattr(pathing, "_save_root", str(tmp_path.resolve()))
    env_worker = EnvWorker.__new__(EnvWorker)
    env_worker.taskset_id = "tasks"
    env_worker._warned_skips = set()
    env_worker._slots = {
        "r1": slot("e1", "t1"),
        "rf": slot("ef", "failed"),
        "rd": slot("ed", "done"),
        "r2": slot("e2", "t2", "t3"),
        "r4": slot("e4", "../t4"),
        "r5": slot("e5"),
    }

    directory = tmp_path / "rollouts"
    response = asyncio.run(env_worker._save(SaveRequest(dir=str(directory))))

    assert set(response.restores) == {"r1", "rd"}
    # The holds end by themselves only once the restore files are written.
    assert armed == [True]
    ref = response.restores["r1"]
    assert ref.trace_id == "t1" and ref.snapshot == {"id": "s-t1"}
    assert ref.path == str(directory / "t1.restore")
    assert sorted(file.name for file in directory.iterdir()) == ["done.restore", "t1.restore"]
    episode_id, restored = read_restore(ref.path)
    assert episode_id == "e1" and restored == point("t1")
    assert discarded == ["s-t2", "s-t9", "s-../t4", "s-failed"]
    # One reason per running rollout without a restore point.
    assert sorted(response.skipped) == [
        *["its episode has more than one trace, or it ended during the save"] * 3,
        "its snapshot upload was not confirmed durable",
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
