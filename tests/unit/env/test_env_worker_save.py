"""An env worker saves its running rollouts as restore files, and tells whose run each
restore point continues."""

import asyncio
from types import SimpleNamespace

from verifiers.v1.env import RunSlot
from verifiers.v1.restore import RestorePoint

from prime_rl.env import worker
from prime_rl.env.types import SaveRequest
from prime_rl.env.worker import EnvWorker, read_restore


def point(trace_id: str) -> RestorePoint:
    checkpoint = {"id": f"c-{trace_id}", "snapshot": {"id": f"s-{trace_id}", "type": "memory"}}
    return RestorePoint(
        trace={"id": trace_id, "checkpoints": [checkpoint]},
        checkpoint=checkpoint["id"],
        state=None,
        info={},
        pins=[],
        replays={},
        live={},
    )


def slot(episode_id: str, *trace_ids: str) -> RunSlot:
    return RunSlot(task=None, id=episode_id, traces=[SimpleNamespace(id=trace_id) for trace_id in trace_ids])


def test_save_maps_restore_points_to_runs_and_deletes_the_snapshots_of_the_rest(tmp_path, monkeypatch):
    discarded: list[str] = []

    class Backend:
        @staticmethod
        async def discard(snapshots) -> list[str]:
            discarded.extend(snapshot.id for snapshot in snapshots)
            return [snapshot.id for snapshot in snapshots]

    async def save_all() -> list[RestorePoint]:
        # t9's run ended while the checkpoints were taken; t2 is one trace of a multi-agent episode.
        return [point("t1"), point("t2"), point("t9"), point("../t4")]

    monkeypatch.setattr(worker, "save_all", save_all)
    monkeypatch.setattr(worker, "PrimeBackend", Backend)
    env_worker = EnvWorker.__new__(EnvWorker)
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
    assert ref.trace_id == "t1" and ref.snapshot == {"id": "s-t1", "type": "memory"}
    assert ref.path == str(directory / "t1.restore")
    assert [file.name for file in directory.iterdir()] == ["t1.restore"]
    episode_id, restored = read_restore(ref.path)
    assert episode_id == "e1" and restored.trace == point("t1").trace and restored.snapshot == ref.snapshot
    assert discarded == ["s-t2", "s-t9", "s-../t4"]


def test_an_unreadable_restore_file_starts_the_rollout_over(tmp_path):
    assert read_restore(str(tmp_path / "missing.restore")) is None
    (tmp_path / "broken.restore").write_bytes(b"not a pickle")
    assert read_restore(str(tmp_path / "broken.restore")) is None
