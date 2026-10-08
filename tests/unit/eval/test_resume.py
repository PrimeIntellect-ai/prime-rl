import uuid
from types import SimpleNamespace

import orjson
import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1.utils.eval import plan_rollouts

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


@pytest.mark.parametrize("skip_checks", [False, True])
def test_take_landed_reads_every_attempt_once(tmp_path, skip_checks) -> None:
    config = EvalConfig(
        source=[{"env": {"id": "single_agent"}}], run={"name": "resume-test"}, resume={"skip_checks": skip_checks}
    )

    def land(*keys: str) -> None:
        resume.stamp_config(tmp_path, config.model_dump(mode="json"))
        if skip_checks:
            (get_file_monitor_dir(tmp_path) / resume.CONFIG_NAME).unlink()
        stream = ChunkedJsonl(get_trace_stream(tmp_path), max_bytes=1 << 20, compress=False)
        for key in keys:
            stream.append(orjson.dumps({**_record("math", key), "id": key}, option=orjson.OPT_APPEND_NEWLINE))
        stream.close()

    land("m0", "m1")
    assert [episode.id for episode in resume.take_landed(tmp_path, config)] == ["m0", "m1"]
    assert not get_file_monitor_dir(tmp_path).exists()
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1"]

    # the resumed attempt re-logged one episode and landed a new one before it died
    land("m0", "m2")
    landed = resume.take_landed(tmp_path, config)
    source = EvalSource([_env("math", ["m0", "m1", "m2"], group_size=2)])
    _, kept = source.trigger(0, completed=landed)
    assert [episode.id for episode in kept] == ["m0", "m1", "m2"]
    assert [request.rollouts for request in source.queue] == [1, 1, 1]
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1", "file.attempt_2"]


@pytest.mark.parametrize("ok", [True, False])
@pytest.mark.parametrize("traces,valid", [([], True), (None, False), ([{}], False)])
def test_resume_validates_saved_episodes(tmp_path, ok, traces, valid) -> None:
    config = EvalConfig(source=[{"env": {"id": "single_agent"}}])
    resume.stamp_config(tmp_path, config.model_dump(mode="json"))
    record = {**_record("math", "m0", ok=ok), "id": "saved", "traces": traces}
    stream = ChunkedJsonl(get_trace_stream(tmp_path), max_bytes=1 << 20, compress=False)
    for row in [{key: value for key, value in record.items() if key != "traces"}, record]:
        stream.append(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
    stream.close()

    if not valid:
        with pytest.raises(ValidationError):
            resume.take_landed(tmp_path, config)
        assert get_file_monitor_dir(tmp_path).is_dir()
        assert resume.archives(tmp_path) == []
        return

    landed = resume.take_landed(tmp_path, config)
    source = EvalSource([_env("math", ["m0"], group_size=2)])
    _, restored = source.trigger(0, completed=landed)

    assert [episode.id for episode in restored] == (["saved"] if ok else [])
    assert [request.rollouts for request in source.queue] == [1 if ok else 2]


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"model": "different-model"},
        {"sampling": {"temperature": 0.3}},
        {"select": {"limit": 2}},
        {"group_size": 2},
        {"client": {"base_url": "https://different.invalid/v1"}},
        {"env": {"timeout": {"episode": 60}}},
        {"log": {"level": "debug"}, "dashboard": False},
        {"concurrency": {"min_inflight": 8, "max_inflight": 8}, "tasks_per_minute": 10},
        {"output_dir": "./another-output-location"},
    ],
)
@pytest.mark.parametrize("skip_checks", [False, True])
def test_take_landed_checks_archived_config_before_rotating(tmp_path, updates, skip_checks) -> None:
    original = {"source": [{"env": {"id": "single_agent"}}], "run": {"name": "resume-test"}}
    saved = EvalConfig.model_validate_json(orjson.dumps(original))
    if "client" in updates:
        updates = updates | {"client": saved.client.model_dump() | updates["client"]}
    current = EvalConfig.model_validate(original | updates | {"resume": {"skip_checks": skip_checks}})
    resume.stamp_config(tmp_path, saved.model_dump(mode="json"))
    directory = get_file_monitor_dir(tmp_path)
    directory.rename(directory.with_name("file.attempt_1"))
    # The newest attempt matches; an incompatible older attempt must still fail.
    resume.stamp_config(tmp_path, current.model_dump(mode="json"))
    if not updates or skip_checks:
        assert resume.take_landed(tmp_path, current) == []
        assert not directory.exists()
    else:
        with pytest.raises(ValueError, match="file.attempt_1/eval.json") as error:
            resume.take_landed(tmp_path, current)
        changed = str(error.value).split(" in [", 1)[1].split("]", 1)[0].split(", ")
        assert set(updates) <= set(changed)
        assert "--resume.skip-checks" in str(error.value)
        assert directory.is_dir()
        assert len(resume.archives(tmp_path)) == 1


@pytest.mark.parametrize(
    "missing",
    [
        None,
        ("model",),
        ("source",),
        ("client", "base_url"),
        ("source", 0, "sampling", "temperature"),
        ("source", 0, "group_size"),
    ],
)
@pytest.mark.parametrize("skip_checks", [False, True])
def test_take_landed_requires_saved_config(tmp_path, missing, skip_checks) -> None:
    config = EvalConfig(source=[{"env": {"id": "single_agent"}}], resume={"skip_checks": skip_checks})
    directory = get_file_monitor_dir(tmp_path)
    directory.mkdir(parents=True)
    if missing is not None:
        saved = config.model_dump(mode="json")
        parent = saved
        for key in missing[:-1]:
            parent = parent[key]
        del parent[missing[-1]]
        resume.stamp_config(tmp_path, saved)
    if skip_checks:
        assert resume.take_landed(tmp_path, config) == []
        assert not directory.exists()
        assert len(resume.archives(tmp_path)) == 1
    else:
        with pytest.raises(ValueError, match="no saved config|config differs"):
            resume.take_landed(tmp_path, config)
        assert directory.is_dir()
        assert resume.archives(tmp_path) == []


@pytest.mark.parametrize("reorder", [False, True])
def test_take_landed_checks_source_order_and_shared_defaults(tmp_path, reorder) -> None:
    saved = EvalConfig(
        source=[{"env": {"id": "single_agent"}}, {"env": {"id": "single_agent"}, "name": "other", "group_size": 2}]
    )
    updated = saved.model_dump(mode="json")
    if reorder:
        updated["source"].reverse()
        changed = "source"
    else:
        updated |= {
            "group_size": 2,
            "sampling": {"temperature": 0.3},
            "select": {"limit": 2},
            "env": {"timeout": {"episode": 60}},
        }
        changed = "env, group_size, sampling, select"
    current = EvalConfig.model_validate(updated | {"resume": {}})
    assert current.source == (list(reversed(saved.source)) if reorder else saved.source)
    resume.stamp_config(tmp_path, saved.model_dump(mode="json"))
    with pytest.raises(ValueError, match=rf"in \[{changed}\].*--resume.skip-checks"):
        resume.take_landed(tmp_path, current)
