import uuid
from types import SimpleNamespace

import orjson
import pytest
import verifiers.v1 as vf
from verifiers.v1.utils.eval import plan_rollouts

from prime_rl.eval import resume
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.utils.pathing import get_file_monitor_dir


def _task(key: str) -> SimpleNamespace:
    return SimpleNamespace(key=key, hash=key)


def _env(name: str, task_keys: list[str], *, group_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, examples=[_task(key) for key in task_keys], config=SimpleNamespace(group_size=group_size)
    )


def _record(env: str, key: str, *, ok: bool = True, group: str | None = None) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "env": {"id": env, "name": env},
        "task": {"type": "Task", "data": {"idx": 0}, "key": key, "hash": key},
        "group": {"id": group or f"group-{key}"},
        "ok": ok,
        "traces": [],
    }


@pytest.mark.parametrize(
    "hashes,rollouts,landed,expected_kept,expected_missing",
    [
        (["a", "b"], 2, [], [[], []], [2, 2]),
        (["a"], 1, [("e1", "a")], [["e1"]], [0]),
        (["changed"], 1, [("e1", "a")], [[]], [1]),
        (["a"], 3, [("e1", "a"), ("e2", "a")], [["e1", "e2"]], [1]),
        (["a"], 1, [("e1", "a"), ("e2", "a")], [["e1"]], [0]),
        (["a", "a"], 2, [("e1", "a")], [["e1"], []], [1, 2]),
        (["a", "changed"], 1, [("e1", "a"), ("e2", "a")], [["e1"], []], [0, 1]),
        (["a"], 2, [("e1", "a"), ("e1", "a")], [["e1"]], [1]),
        (["a"], 1, [("e1", "removed")], [[]], [1]),
        ([], 1, [("e1", "a")], [], []),
    ],
)
def test_plan_keeps_landed_rollouts_up_to_the_target_and_owes_the_rest(
    hashes, rollouts, landed, expected_kept, expected_missing
) -> None:
    tasks = [SimpleNamespace(key="stable-key", hash=content_hash) for content_hash in hashes]
    episodes = []
    for episode_id, content_hash in landed:
        row = _record("math", "stable-key")
        row["id"] = episode_id
        row["task"]["hash"] = content_hash
        episodes.append(vf.WireEpisode.model_validate(row))

    plan = plan_rollouts(tasks, rollouts, episodes)

    assert [task for task, _, _ in plan] == tasks
    assert [[episode.id for episode in kept] for _, kept, _ in plan] == expected_kept
    assert [missing for _, _, missing in plan] == expected_missing
    assert all(len(kept) + missing == rollouts for _, kept, missing in plan)


def test_trigger_queues_only_owed_rollouts() -> None:
    source = EvalSource([_env("math", ["m0", "m1", "m1"], group_size=2), _env("code", ["m1"])])
    episodes = [vf.WireEpisode.model_validate(_record("math", key)) for key in ["m0", "m0", "m1"]]

    fired, restored = source.trigger(0, completed=episodes)

    assert fired == ["math", "code"]
    requests = list(source.queue)
    assert [(request.env_name, request.task.key, request.rollouts) for request in requests] == [
        ("code", "m1", 1),
        ("math", "m1", 1),
        ("math", "m1", 2),
    ]
    assert restored == episodes
    assert len({request.group_id for request in requests}) == 3
    assert restored[-1].group.id == requests[1].group_id
    assert restored[0].group == restored[1].group
    assert restored[0].group != restored[-1].group
    assert all(str(uuid.UUID(request.group_id)) == request.group_id for request in requests)

    # Saved episodes affect only this epoch; later triggers plan every rollout again.
    source.queue.clear()
    assert source.trigger(1) == (["math", "code"], [])
    assert [request.rollouts for request in source.queue] == [2, 1, 2, 2]


def test_take_landed_reads_every_attempt_once(tmp_path) -> None:
    def land(*keys: str) -> None:
        stream = ChunkedJsonl(get_trace_stream(tmp_path), max_bytes=1 << 20, compress=False)
        for key in keys:
            stream.append(orjson.dumps({**_record("math", key), "id": key}, option=orjson.OPT_APPEND_NEWLINE))
        stream.close()

    land("m0", "m1")
    assert [episode.id for episode in resume.take_landed(tmp_path)] == ["m0", "m1"]
    assert not get_file_monitor_dir(tmp_path).exists()
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1"]

    # the resumed attempt re-logged one episode and landed a new one before it died
    land("m0", "m2")
    landed = resume.take_landed(tmp_path)
    source = EvalSource([_env("math", ["m0", "m1", "m2"], group_size=2)])
    _, kept = source.trigger(0, completed=landed)
    assert [episode.id for episode in kept] == ["m0", "m1", "m2"]
    assert [request.rollouts for request in source.queue] == [1, 1, 1]
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1", "file.attempt_2"]
