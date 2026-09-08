import pickle
from types import SimpleNamespace

import pytest

from prime_rl.evals.ckpt import CheckpointManager
from prime_rl.orchestrator.eval_source import EvalSource


def _task(key: str) -> SimpleNamespace:
    return SimpleNamespace(key=key, hash=key)


def _env(name: str, task_keys: list[str], *, interval: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        examples=[_task(key) for key in task_keys],
        config=SimpleNamespace(interval=interval, group_size=8),
    )


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


def test_eval_checkpoint_resume_skips_completed_prefix(tmp_path) -> None:
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

    ckpt = CheckpointManager(tmp_path)
    ckpt.save(eval_source)

    with (ckpt.get_ckpt_path(2) / "progress.pt").open("rb") as f:
        assert pickle.load(f) == eval_source.state_dict()

    assert ckpt.latest_step() == 2

    restored_source = EvalSource(eval_envs, _eval_config())
    ckpt.load(2, restored_source)

    assert restored_source.state_dict() == eval_source.state_dict()


def test_eval_checkpoint_preserves_partial_and_out_of_order_progress(tmp_path) -> None:
    envs = [_env("math", ["m0", "m1"]), _env("code", ["c0", "c1"])]
    source = EvalSource(envs, _eval_config())
    source.trigger(0)
    for _ in range(8):
        source.record_attempt(3, 8)
    for _ in range(3):
        source.record_attempt(1, 8, "group-one")
    assert source.cursor == 0
    ckpt = CheckpointManager(tmp_path)
    ckpt.save(source)
    restored = EvalSource(envs, _eval_config())
    ckpt.load(0, restored)
    assert restored.group_ids == {1: "group-one"}
    restored.trigger(0)
    assert [(r.source_index, r.num_rollouts) for r in restored.queue] == [(0, 8), (1, 5), (2, 8)]
    assert restored.triggered_rollout_count("code", 0) == 5
    for index, count in [(0, 8), (1, 5), (2, 8)]:
        for _ in range(count):
            restored.record_attempt(index, 8)
    assert restored.cursor == 4
    assert not restored.partial
    assert not restored._completed


def test_eval_checkpoint_rejects_changed_order(tmp_path) -> None:
    source = EvalSource([_env("math", ["m0", "m1"])], _eval_config())
    changed = EvalSource([_env("math", ["m1", "m0"])], _eval_config())
    with pytest.raises(ValueError, match="dataset order"):
        changed.load_state_dict(source.state_dict())


def test_eval_checkpoint_rejects_step_cursor_mismatch(tmp_path) -> None:
    eval_source = EvalSource([_env("math", ["m0"])], _eval_config())
    eval_source.load_state_dict({"cursor": 1})

    ckpt = CheckpointManager(tmp_path)
    ckpt.save(eval_source)

    restored_source = EvalSource([_env("math", ["m0"])], _eval_config())
    with pytest.raises(ValueError, match="contains cursor 1, expected step 2"):
        ckpt.load(2, restored_source, path=ckpt.get_ckpt_path(1))
