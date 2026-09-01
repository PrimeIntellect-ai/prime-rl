import pickle
from types import SimpleNamespace

import pytest

from prime_rl.evals.ckpt import CheckpointManager
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource


def _task(key: str) -> SimpleNamespace:
    return SimpleNamespace(key=key, hash=key)


def _env(name: str, task_keys: list[str], *, interval: int = 1, group_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        examples=[_task(key) for key in task_keys],
        config=SimpleNamespace(interval=interval, group_size=group_size),
    )


def _eval_envs(*envs: SimpleNamespace) -> SimpleNamespace:
    by_name = {env.name: env for env in envs}
    return SimpleNamespace(get=by_name.__getitem__)


def _eval_config() -> SimpleNamespace:
    return SimpleNamespace(skip_first_step=False)


def _queued_tasks(eval_source: EvalSource) -> list[tuple[str, str, int | None]]:
    return [(request.env_name, request.task.key, request.source_index) for request in eval_source.queue]


def test_eval_source_cursor_advances_only_over_completed_prefix() -> None:
    eval_source = EvalSource(
        [_env("math", ["m0", "m1"]), _env("code", ["c0", "c1"])],
        _eval_config(),
    )

    assert eval_source.trigger(0) == ["math", "code"]
    assert _queued_tasks(eval_source) == [
        ("math", "m0", 0),
        ("code", "c0", 1),
        ("math", "m1", 2),
        ("code", "c1", 3),
    ]

    assert eval_source.mark_completed(1) is False
    assert eval_source.cursor == 0

    assert eval_source.mark_completed(3) is False
    assert eval_source.cursor == 0

    assert eval_source.mark_completed(0) is True
    assert eval_source.cursor == 2

    assert eval_source.mark_completed(2) is True
    assert eval_source.cursor == 4


def test_eval_checkpoint_saves_only_cursor_and_resume_skips_completed_prefix(tmp_path) -> None:
    eval_envs = [_env("math", ["m0", "m1"]), _env("code", ["c0", "c1"])]
    eval_source = EvalSource(eval_envs, _eval_config())
    eval_source.load_state_dict({"cursor": 2})

    assert eval_source.trigger(0) == ["math", "code"]
    assert _queued_tasks(eval_source) == [
        ("math", "m1", 2),
        ("code", "c1", 3),
    ]
    assert eval_source.triggered_task_count("math", 0) == 1
    assert eval_source.triggered_task_count("code", 0) == 1
    assert eval_source.skipped_task_count("math", 0) == 1
    assert eval_source.skipped_task_count("code", 0) == 1

    ckpt = CheckpointManager(tmp_path)
    ckpt.save(eval_source)

    with (ckpt.get_ckpt_path(2) / "progress.pt").open("rb") as f:
        assert pickle.load(f) == {"cursor": 2}

    assert ckpt.latest_step() == 2

    restored_source = EvalSource(eval_envs, _eval_config())
    ckpt.load(2, restored_source)

    assert restored_source.state_dict() == {"cursor": 2}


def test_eval_checkpoint_rejects_step_cursor_mismatch(tmp_path) -> None:
    eval_source = EvalSource([_env("math", ["m0"])], _eval_config())
    eval_source.load_state_dict({"cursor": 1})

    ckpt = CheckpointManager(tmp_path)
    ckpt.save(eval_source)

    restored_source = EvalSource([_env("math", ["m0"])], _eval_config())
    with pytest.raises(ValueError, match="contains cursor 1, expected step 2"):
        ckpt.load(2, restored_source, path=ckpt.get_ckpt_path(1))


def test_eval_source_counts_skipped_cursor_prefix_for_progress() -> None:
    group_size = 4
    eval_source = EvalSource(
        [_env("math", ["m0", "m1", "m2"]), _env("code", ["c0", "c1"])],
        _eval_config(),
    )
    eval_source.load_state_dict({"cursor": 3})

    assert eval_source.trigger(0) == ["math", "code"]
    assert _queued_tasks(eval_source) == [
        ("code", "c1", 3),
        ("math", "m2", 4),
    ]
    assert eval_source.skipped_task_count("math", 0) == 2
    assert eval_source.skipped_task_count("code", 0) == 1
    assert eval_source.triggered_task_count("math", 0) == 1
    assert eval_source.triggered_task_count("code", 0) == 1

    shown_arrived = eval_source.skipped_task_count("math", 0) * group_size
    shown_expected = shown_arrived + eval_source.triggered_task_count("math", 0) * group_size
    assert (shown_arrived, shown_expected) == (8, 12)


def test_eval_sink_reports_expected_empty_batch_for_progress() -> None:
    eval_sink = EvalSink(eval_envs=_eval_envs(_env("math", ["m0"], group_size=4)))

    eval_sink.set_batch_size("math", step=0, size=4)

    assert eval_sink.batch_progress() == [("math", 0, 0, 4, 0)]
