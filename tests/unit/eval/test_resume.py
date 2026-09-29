from types import SimpleNamespace

import orjson

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
        "env": {"id": env, "name": env},
        "task": {"type": "Task", "data": {"idx": 0}, "key": key, "hash": key},
        "group": {"id": group or f"group-{key}"},
        "ok": ok,
        "traces": [],
    }


def test_plan_keeps_landed_rollouts_up_to_the_target_and_owes_the_rest() -> None:
    envs = [_env("math", ["m0", "m1", "m2"], group_size=2), _env("code", ["c0"])]
    landed = [
        _record("math", "m0"),
        _record("math", "m0"),
        _record("math", "m0"),  # a third rollout of m0 exceeds group_size 2
        _record("math", "m1"),
        _record("math", "m9"),  # no longer selected (num_examples shrank)
        _record("code", "c0", ok=False),  # errored: owed again
    ]

    summaries = [
        resume.Landed(id=str(i), env_name=r["env"]["name"], key=r["task"]["key"], group_id=r["group"]["id"], step=0)
        for i, r in enumerate(landed)
        if r["ok"]
    ]
    kept, owed, groups = resume.plan(summaries, envs)

    assert [(episode.env_name, episode.key) for episode in kept] == [
        ("math", "m0"),
        ("math", "m0"),
        ("math", "m1"),
    ]
    assert owed == {"math": {"m1": 1, "m2": 2}, "code": {"c0": 1}}
    # the owed rollout of m1 completes the group its landed rollout opened
    assert groups == {"math": {"m0": "group-m0", "m1": "group-m1"}}


def test_trigger_queues_only_owed_rollouts() -> None:
    source = EvalSource([_env("math", ["m0", "m1", "m2"], group_size=2), _env("code", ["c0"])])
    source.restore({"math": {"m1": 1, "m2": 2}, "code": {}}, {"math": {"m1": "group-m1"}})

    assert source.trigger(0) == ["math", "code"]
    assert [(request.env_name, request.task.key, request.rollouts, request.group_id) for request in source.queue] == [
        ("math", "m1", 1, "group-m1"),
        ("math", "m2", 2, None),
    ]


def test_groups_per_step_advances_the_step_in_dispatch_order() -> None:
    envs = [_env("math", ["m0", "m1", "m2"], group_size=2), _env("code", ["c0", "c1"])]
    default = EvalSource(envs)
    source = EvalSource(envs, groups_per_step=2)

    default.trigger(0)
    source.trigger(3)

    # same groups in the same order; only the step label advances every two groups
    assert [(r.env_name, r.task.key) for r in source.queue] == [(r.env_name, r.task.key) for r in default.queue]
    assert [request.step for request in source.queue] == [3, 3, 4, 4, 5]
    assert source.planned == {("math", 3): 2, ("code", 3): 1, ("math", 4): 2, ("code", 4): 1, ("math", 5): 2}
    assert {request.step for request in default.queue} == {0}


def test_take_landed_reads_every_attempt_once(tmp_path) -> None:
    def land(*keys: str) -> None:
        stream = ChunkedJsonl(get_trace_stream(tmp_path), max_bytes=1 << 20, compress=False)
        for key in keys:
            stream.append(orjson.dumps({**_record("math", key), "id": key}, option=orjson.OPT_APPEND_NEWLINE))
        stream.close()

    land("m0", "m1")
    assert [episode.id for episode in resume.take_landed(tmp_path, {"math"})] == ["m0", "m1"]
    assert not get_file_monitor_dir(tmp_path).exists()
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1"]

    # the resumed attempt re-logged one episode and landed a new one before it died
    land("m0", "m2")
    landed = resume.take_landed(tmp_path, {"math"})
    assert [episode.id for episode in landed] == ["m0", "m1", "m2"]
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1", "file.attempt_2"]

    # the kept episodes are re-read one at a time, each once, in stream order
    assert [episode.id for episode in resume.replay(tmp_path, [landed[0], landed[2]])] == ["m0", "m2"]

    # envs the resumed run no longer configures are not read into memory
    assert resume.take_landed(tmp_path, {"code"}) == []
