"""An env worker saves each running episode as a restore file, and tells whose run each
restore point continues."""

import asyncio
import pickle
from types import SimpleNamespace

from verifiers.v1.env import RunSlot

from prime_rl.env import worker
from prime_rl.env.types import RestoreRef, SaveRequest
from prime_rl.env.worker import EnvWorker, read_restore
from prime_rl.utils import pathing


def point(trace_id: str) -> dict:
    """A saved trace: the session data is on the entry of its checkpoint."""
    checkpoint = {"id": f"c-{trace_id}", "snapshot": {"id": f"s-{trace_id}"}, "session": {}}
    return {"id": trace_id, "checkpoints": [checkpoint]}


def episode_point(episode_id: str, *trace_ids: str) -> dict:
    """An episode's restore point that continues the running calls `trace_ids`."""
    calls = [{"seat": "agent", "n": n, "state": "running", "trace": point(t)} for n, t in enumerate(trace_ids)]
    return {
        "episode_point": 1,
        "episode_id": episode_id,
        "calls": calls,
        "snapshots": [{"id": f"s-{t}"} for t in trace_ids],
    }


def test_save_writes_one_restore_point_per_episode_and_deletes_the_snapshots_of_the_rest(tmp_path, monkeypatch):
    discarded: list[str] = []

    class Backend:
        @staticmethod
        async def discard(snapshots) -> list[str]:
            discarded.extend(snapshot.id for snapshot in snapshots)
            # Prime refuses the newest snapshot of a box that still runs: t3's episode goes on.
            return [snapshot.id for snapshot in snapshots if snapshot.id != "s-t3"]

    async def save_all(*, hold: bool, watchdog: bool, timeout: float | None) -> list[dict]:
        # t9's run ended while the checkpoints were taken.
        assert hold and not watchdog
        return [point("t1"), point("t2"), point("t3"), point("t9")]

    points = {
        # Two running calls of one episode.
        "e1": (episode_point("e1", "t1", "t2"), None),
        # t3's episode also runs an interaction, so it has no restore point.
        "e3": (None, "user call 0: it is an interaction, which no checkpoint can continue"),
        "e5": (None, "agent call 0: the 'subprocess' runtime takes no memory snapshots"),
        # Nothing ran yet: the episode starts over without a reason to report.
        "e6": (episode_point("e6"), None),
    }
    monkeypatch.setattr(worker, "save_all", save_all)
    monkeypatch.setattr(worker, "episode_restore_point", lambda slot, saved: points[slot.id])
    armed: list[bool] = []
    monkeypatch.setattr(worker, "arm_hold_watchdogs", lambda: armed.append(True))
    monkeypatch.setattr(worker, "PrimeBackend", Backend)
    monkeypatch.setattr(pathing, "_save_root", str(tmp_path.resolve()))
    env_worker = EnvWorker.__new__(EnvWorker)
    env_worker.taskset_id = "tasks"
    env_worker._warned_skips = set()
    env_worker._slots = {f"r{e[1:]}": RunSlot(task=None, id=e, traces=[SimpleNamespace(id="t")]) for e in points}

    directory = tmp_path / "rollouts"
    response = asyncio.run(env_worker._save(SaveRequest(dir=str(directory))))

    assert set(response.restores) == {"r1"}
    ref = response.restores["r1"]
    assert ref.episode_id == "e1" and ref.snapshots == [{"id": "s-t1"}, {"id": "s-t2"}]
    assert ref.path == str(directory / "e1.restore")
    assert [file.name for file in directory.iterdir()] == ["e1.restore"]
    assert read_restore(ref.path) == ("e1", points["e1"][0])
    # The holds end by themselves only once the restore files are written.
    assert armed == [True]
    assert discarded == ["s-t3", "s-t9"]
    # A snapshot the worker could not delete goes to the owner, which retries.
    assert response.orphans == [{"id": "s-t3"}]
    # One reason per running episode without a restore point, without the call it names.
    assert sorted(response.skipped) == [
        "it is an interaction, which no checkpoint can continue",
        "the 'subprocess' runtime takes no memory snapshots",
    ]


def test_a_restore_point_of_an_older_checkpoint_still_reads(tmp_path, monkeypatch):
    """A file and a reference written when a restore point continued a single trace."""
    monkeypatch.setattr(pathing, "_save_root", str(tmp_path.resolve()))
    path = tmp_path / "t1.restore"
    path.write_bytes(pickle.dumps({"episode_id": "e1", "trace": point("t1")}))
    assert read_restore(str(path)) == ("e1", point("t1"))

    ref = RestoreRef.__new__(RestoreRef)
    ref.__setstate__(
        {
            "__dict__": {"trace_id": "t1", "path": str(path), "snapshot": {"id": "s-t1"}},
            "__pydantic_extra__": None,
            "__pydantic_fields_set__": {"trace_id", "path", "snapshot"},
            "__pydantic_private__": None,
        }
    )
    assert (ref.episode_id, ref.path, ref.snapshots) == ("t1", str(path), [{"id": "s-t1"}])


def test_an_unreadable_restore_file_starts_the_rollout_over(tmp_path, monkeypatch):
    monkeypatch.setattr(pathing, "_save_root", str(tmp_path.resolve()))
    assert read_restore(str(tmp_path / "missing.restore")) is None
    (tmp_path / "broken.restore").write_bytes(b"not a pickle")
    assert read_restore(str(tmp_path / "broken.restore")) is None


def test_a_restore_file_outside_the_run_directory_is_not_read(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(pathing, "_save_root", str((tmp_path / "run").resolve()))
    worker.write_restores(tmp_path, [("e1", episode_point("e1", "t1"))])
    assert read_restore(str(tmp_path / "e1.restore")) is None
    assert read_restore(str(tmp_path / "run" / ".." / "e1.restore")) is None
