from types import SimpleNamespace

import orjson
import pytest

from prime_rl.configs.eval import EvalConfig
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
        _record("math", "m9"),  # no longer selected (select.limit shrank)
        _record("code", "c0", ok=False),  # errored: owed again
    ]

    kept, owed, groups = resume.plan([record for record in landed if record["ok"]], envs)

    assert [(episode.env.name, episode.task.key) for episode in kept] == [
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


def test_take_landed_reads_every_attempt_once(tmp_path) -> None:
    config = EvalConfig(source=[{"env": {"id": "single_agent"}}], run={"name": "resume-test"})

    def land(*keys: str) -> None:
        resume.stamp_config(tmp_path, config.model_dump(mode="json"))
        stream = ChunkedJsonl(get_trace_stream(tmp_path), max_bytes=1 << 20, compress=False)
        for key in keys:
            stream.append(orjson.dumps({**_record("math", key), "id": key}, option=orjson.OPT_APPEND_NEWLINE))
        stream.close()

    land("m0", "m1")
    assert [record["id"] for record in resume.take_landed(tmp_path, config)] == ["m0", "m1"]
    assert not get_file_monitor_dir(tmp_path).exists()
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1"]

    # the resumed attempt re-logged one episode and landed a new one before it died
    land("m0", "m2")
    assert [record["id"] for record in resume.take_landed(tmp_path, config)] == ["m0", "m1", "m2"]
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1", "file.attempt_2"]


@pytest.mark.parametrize(
    "updates,allowed",
    [
        ({"model": "different-model"}, False),
        ({"sampling": {"temperature": 0.3}}, False),
        ({"select": {"limit": 2}}, False),
        ({"group_size": 2}, False),
        ({"client": {"base_url": "https://different.invalid/v1"}}, False),
        ({"env": {"timeout": {"episode": 60}}}, False),
        ({"log": {"level": "debug"}, "dashboard": False, "monitors": {"prime": None}}, True),
        ({"concurrency": {"min_inflight": 8, "max_inflight": 8}, "tasks_per_minute": 10}, True),
        ({"client": {"wait_for_ready_timeout": 10}}, True),
        ({"source": [{"env": {"id": "single_agent"}, "serve": {"pool": {"type": "elastic", "max_workers": 2}}}]}, True),
    ],
)
def test_take_landed_checks_archived_experiment_before_rotating(tmp_path, updates, allowed) -> None:
    original = {"source": [{"env": {"id": "single_agent"}}], "run": {"name": "resume-test"}}
    saved = EvalConfig.model_validate(original)
    if "client" in updates:
        updates = updates | {"client": saved.client.model_dump() | updates["client"]}
    current = EvalConfig.model_validate(original | updates | {"resume": True})
    resume.stamp_config(tmp_path, saved.model_dump(mode="json"))
    directory = get_file_monitor_dir(tmp_path)
    directory.rename(directory.with_name("file.attempt_1"))
    # The newest attempt matches; an incompatible older attempt must still fail.
    resume.stamp_config(tmp_path, current.model_dump(mode="json"))
    if allowed:
        assert resume.take_landed(tmp_path, current) == []
        assert not directory.exists()
    else:
        with pytest.raises(ValueError, match="file.attempt_1/eval.json"):
            resume.take_landed(tmp_path, current)
        assert directory.is_dir()
        assert len(resume.archives(tmp_path)) == 1


@pytest.mark.parametrize(
    "missing",
    [
        None,
        ("model",),
        ("client", "skip_model_check"),
        ("source", 0, "sampling", "temperature"),
        ("source", 0, "group_size"),
    ],
)
def test_take_landed_requires_saved_experiment(tmp_path, missing) -> None:
    config = EvalConfig(source=[{"env": {"id": "single_agent"}}])
    directory = get_file_monitor_dir(tmp_path)
    directory.mkdir(parents=True)
    if missing is not None:
        saved = config.model_dump(mode="json")
        parent = saved
        for key in missing[:-1]:
            parent = parent[key]
        del parent[missing[-1]]
        resume.stamp_config(tmp_path, saved)
    with pytest.raises(ValueError, match="no saved experiment config|config differs"):
        resume.take_landed(tmp_path, config)
    assert directory.is_dir()
    assert resume.archives(tmp_path) == []


def test_take_landed_compares_effective_sources(tmp_path) -> None:
    saved = EvalConfig(source=[{"env": {"id": "single_agent"}}])
    current = EvalConfig.model_validate(
        saved.model_dump(mode="json")
        | {
            "group_size": 2,
            "sampling": {"temperature": 0.3},
            "select": {"limit": 2},
            "env": {"timeout": {"episode": 60}},
        }
    )
    assert current.source == saved.source
    resume.stamp_config(tmp_path, saved.model_dump(mode="json"))
    assert resume.take_landed(tmp_path, current) == []
