from types import SimpleNamespace

import orjson
import pytest

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


def test_check_config_allows_selection_and_rollout_timeout_changes() -> None:
    previous = {
        "model": "a",
        "num_examples": 8,
        "group_size": 2,
        "sampling": {"temperature": 1.0},
        "source": [
            {
                "env": {"taskset": {"id": "gsm8k"}, "agent": {"timeout": {"rollout": 7200, "setup": 60}}},
                "group_size": None,
                "serve": {"address": None},
            }
        ],
    }
    resized = {
        **previous,
        "num_examples": 16,
        "group_size": 4,
        "source": [
            {
                "env": {"taskset": {"id": "gsm8k"}, "agent": {"timeout": {"rollout": 21600, "setup": 60}}},
                "group_size": 8,
                "serve": {"address": "tcp://x"},
            }
        ],
    }
    resume.check_config(previous, resized)

    with pytest.raises(ValueError, match="model, sampling.temperature"):
        resume.check_config(previous, {**previous, "model": "b", "sampling": {"temperature": 0.5}})
    with pytest.raises(ValueError, match="source"):
        resume.check_config(previous, {**previous, "source": previous["source"] * 2})
    with pytest.raises(ValueError, match="source.0.env.agent.timeout.setup"):
        resume.check_config(
            previous,
            {
                **previous,
                "source": [
                    {
                        **previous["source"][0],
                        "env": {
                            **previous["source"][0]["env"],
                            "agent": {"timeout": {"rollout": 21600, "setup": 120}},
                        },
                    }
                ],
            },
        )


def test_take_landed_reads_every_attempt_once(tmp_path) -> None:
    def land(*keys: str) -> None:
        stream = ChunkedJsonl(get_trace_stream(tmp_path), max_bytes=1 << 20, compress=False)
        for key in keys:
            stream.append(orjson.dumps({**_record("math", key), "id": key}, option=orjson.OPT_APPEND_NEWLINE))
        stream.close()

    land("m0", "m1")
    assert [record["id"] for record in resume.take_landed(tmp_path)] == ["m0", "m1"]
    assert not get_file_monitor_dir(tmp_path).exists()
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1"]

    # the resumed attempt re-logged one episode and landed a new one before it died
    land("m0", "m2")
    assert [record["id"] for record in resume.take_landed(tmp_path)] == ["m0", "m1", "m2"]
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1", "file.attempt_2"]


def test_take_landed_can_keep_current_attempt_errors(tmp_path) -> None:
    stream = ChunkedJsonl(get_trace_stream(tmp_path), max_bytes=1 << 20, compress=False)
    stream.append(orjson.dumps({**_record("math", "m0"), "id": "m0"}, option=orjson.OPT_APPEND_NEWLINE))
    stream.append(orjson.dumps({**_record("math", "m1", ok=False), "id": "old-m1"}, option=orjson.OPT_APPEND_NEWLINE))
    stream.close()
    resume.take_landed(tmp_path)

    stream = ChunkedJsonl(get_trace_stream(tmp_path), max_bytes=1 << 20, compress=False)
    stream.append(orjson.dumps({**_record("math", "m0"), "id": "m0"}, option=orjson.OPT_APPEND_NEWLINE))
    stream.append(orjson.dumps({**_record("math", "m1", ok=False), "id": "new-m1"}, option=orjson.OPT_APPEND_NEWLINE))
    stream.append(orjson.dumps({**_record("math", "m2", ok=False), "id": "m2"}, option=orjson.OPT_APPEND_NEWLINE))
    stream.close()

    landed = resume.take_landed(tmp_path, keep_failed=True)
    assert [record["id"] for record in landed] == ["m0", "new-m1", "m2"]
    kept, owed, _ = resume.plan(landed, [_env("math", ["m0", "m1", "m2", "m3"])])
    assert [(episode.task.key, episode.ok) for episode in kept] == [
        ("m0", True),
        ("m1", False),
        ("m2", False),
    ]
    assert owed == {"math": {"m3": 1}}


def test_previous_config_is_the_one_stamped_beside_the_results(tmp_path) -> None:
    resume.stamp_config(tmp_path, {"model": "a"})
    resume.take_landed(tmp_path)  # the attempt that ran is archived; a rejected one never stamps
    assert resume.previous_config(tmp_path) == {"model": "a"}
    resume.stamp_config(tmp_path, {"model": "b"})
    assert resume.previous_config(tmp_path) == {"model": "b"}
