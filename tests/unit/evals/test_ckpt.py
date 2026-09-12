import pickle
from types import SimpleNamespace

import pytest

from prime_rl.evals.ckpt import CheckpointManager
from prime_rl.evals.evals import Evals
from prime_rl.orchestrator.eval_source import EvalSource


def _task(key: str) -> SimpleNamespace:
    return SimpleNamespace(key=key, hash=key)


def _env(name: str, task_keys: list[str], *, interval: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        examples=[_task(key) for key in task_keys],
        config=SimpleNamespace(interval=interval),
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

    assert eval_source.mark_completed(1) is True
    assert eval_source.cursor == 0
    assert eval_source.num_completed == 1
    assert eval_source.mark_completed(1) is False

    assert eval_source.mark_completed(3) is True
    assert eval_source.cursor == 0

    assert eval_source.mark_completed(0) is True
    assert eval_source.cursor == 2

    assert eval_source.mark_completed(2) is True
    assert eval_source.cursor == 4


def test_eval_checkpoint_resume_skips_prefix_and_out_of_order_groups(tmp_path) -> None:
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
    assert eval_source.mark_completed(3)

    ckpt = CheckpointManager(tmp_path)
    ckpt.save(eval_source)

    with (ckpt.get_ckpt_path(2) / "progress.pt").open("rb") as f:
        assert pickle.load(f) == {"cursor": 2, "completed": [3]}

    assert ckpt.latest_step() == 2

    restored_source = EvalSource(eval_envs, _eval_config())
    ckpt.load(2, restored_source)

    assert restored_source.state_dict() == {"cursor": 2, "completed": [3]}
    assert restored_source.trigger(0) == ["math"]
    assert _queued_tasks(restored_source) == [("math", "m1", 2)]
    assert restored_source.triggered_task_count("math", 0) == 1
    assert restored_source.triggered_task_count("code", 0) == 0
    assert restored_source.mark_completed(2)
    assert restored_source.cursor == 4


def test_eval_checkpoint_saves_progress_with_zero_prefix(tmp_path) -> None:
    source = EvalSource([_env("math", ["m0", "m1", "m2"])], _eval_config())
    source.trigger(0)
    source.mark_completed(2)
    ckpt = CheckpointManager(tmp_path)
    ckpt.save(source)
    source.mark_completed(1)
    ckpt.save(source)
    restored = EvalSource(source.eval_envs, _eval_config())
    ckpt.load(0, restored)
    assert restored.num_completed == 2
    assert restored.trigger(0) == ["math"]
    assert _queued_tasks(restored) == [("math", "m0", 0)]


def test_eval_checkpoint_interval_counts_out_of_order_completions(tmp_path) -> None:
    source = EvalSource([_env("math", ["m0", "m1", "m2", "m3"])], _eval_config())
    source.trigger(0)
    evals = Evals.__new__(Evals)
    evals.eval_source = source
    evals.config = SimpleNamespace(ckpt=SimpleNamespace(interval=2))
    evals.ckpt_manager = CheckpointManager(tmp_path)
    evals.last_saved_completed = 0
    source.mark_completed(1)
    evals.maybe_save_checkpoint()
    assert not evals.ckpt_manager.get_ckpt_path(0).exists()
    source.mark_completed(3)
    evals.maybe_save_checkpoint()
    with (evals.ckpt_manager.get_ckpt_path(0) / "progress.pt").open("rb") as stream:
        assert pickle.load(stream) == {"cursor": 0, "completed": [1, 3]}
    assert evals.last_saved_completed == 2


@pytest.mark.parametrize("completed", [[0], [-1], [1, 1], [True], ["1"], None])
def test_eval_checkpoint_rejects_invalid_completed_indices(completed) -> None:
    source = EvalSource([_env("math", ["m0", "m1"])], _eval_config())
    with pytest.raises(ValueError, match="Completed indices"):
        source.load_state_dict({"cursor": 0, "completed": completed})


def test_eval_checkpoint_rejects_step_cursor_mismatch(tmp_path) -> None:
    eval_source = EvalSource([_env("math", ["m0"])], _eval_config())
    eval_source.load_state_dict({"cursor": 1})

    ckpt = CheckpointManager(tmp_path)
    ckpt.save(eval_source)

    restored_source = EvalSource([_env("math", ["m0"])], _eval_config())
    with pytest.raises(ValueError, match="contains cursor 1, expected step 2"):
        ckpt.load(2, restored_source, path=ckpt.get_ckpt_path(1))
