import pytest

from prime_rl.configs.shared import ResumeConfig
from prime_rl.utils.pathing import (
    clean_future_steps,
    create_attempt_dirs,
    get_batch_dir,
    get_broadcast_dir,
    get_step_path,
    resolve_latest_ckpt_step,
    resolve_resume,
    validate_run_dir,
)


def test_nonexistent_dir_passes(tmp_path):
    run_dir = tmp_path / "does_not_exist"
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_dir_with_launcher_artifacts_passes(tmp_path):
    run_dir = tmp_path / "submitted"
    run_dir.mkdir()
    config_dir, log_dir = create_attempt_dirs(run_dir)
    (config_dir / "trainer.json").touch()
    next_config_dir, next_log_dir = create_attempt_dirs(run_dir)
    assert next_config_dir.parent.name == "attempt_2"
    assert next_log_dir.name == "attempt_2"
    assert (run_dir / "configs" / "latest").resolve() == next_config_dir.parent
    assert (run_dir / "logs" / "latest").resolve() == next_log_dir
    (run_dir / "launcher").mkdir()
    (run_dir / "launcher" / "rl.sbatch").touch()
    (run_dir / "launcher" / ".trainer.done").touch()
    (run_dir / "launcher" / ".orchestrator.done").touch()
    (run_dir / "launcher" / "logs").mkdir()
    (run_dir / "launcher" / "logs" / "job_1234.log").touch()
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_dir_with_logs_raises(tmp_path):
    run_dir = tmp_path / "has_logs"
    run_dir.mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "trainer.log").touch()
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_dir_with_checkpoints_raises(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_dir_with_checkpoints_passes_when_resuming(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=True, clean=False)


def test_dir_with_checkpoints_cleaned_when_flag_set(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    (run_dir / "logs").mkdir()

    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=True)

    assert not run_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    run_dir = tmp_path / "does_not_exist"
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=True)
    assert not run_dir.exists()


def test_ckpt_output_dir_with_checkpoints_raises(tmp_path):
    run_dir = tmp_path / "fresh"
    ckpt_output_dir = tmp_path / "ckpts"
    ckpt_output_dir.mkdir()
    (ckpt_output_dir / "checkpoints").mkdir()
    (ckpt_output_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False, ckpt_output_dir=ckpt_output_dir)


def test_clean_outside_output_dir_raises(tmp_path):
    escaped = tmp_path / "elsewhere"
    escaped.mkdir()
    with pytest.raises(ValueError, match="remain under output_dir"):
        validate_run_dir(escaped, output_dir=tmp_path / "outputs", resuming=False, clean=True)
    assert escaped.exists()


def test_clean_future_steps_rebuilds_resume_broadcast(tmp_path):
    batch_dir = get_batch_dir(tmp_path)
    broadcast_dir = get_broadcast_dir(tmp_path)
    for parent in (batch_dir, broadcast_dir):
        for step in (1, 2, 3):
            get_step_path(parent, step).mkdir(parents=True)

    clean_future_steps(tmp_path, resume_step=2)

    assert get_step_path(batch_dir, 1).exists()
    assert get_step_path(batch_dir, 2).exists()
    assert not get_step_path(batch_dir, 3).exists()
    assert get_step_path(broadcast_dir, 1).exists()
    assert not get_step_path(broadcast_dir, 2).exists()
    assert not get_step_path(broadcast_dir, 3).exists()


def test_resolve_latest_ckpt_step_skips_incomplete_steps(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    for step, files in {
        10: ["trainer/.metadata", "orchestrator/progress.pt"],
        20: ["trainer/.metadata", "orchestrator/progress.pt"],
        30: ["trainer/__0_0.distcp", "orchestrator/progress.pt"],  # trainer save killed mid-write
        34: ["orchestrator/progress.pt"],  # orchestrator ahead of the trainer
    }.items():
        for file in files:
            path = get_step_path(ckpt_dir, step) / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    assert resolve_latest_ckpt_step(ckpt_dir, ("trainer", "orchestrator")) == 20
    assert resolve_latest_ckpt_step(ckpt_dir, ("trainer",)) == 20
    assert resolve_latest_ckpt_step(ckpt_dir, ("orchestrator",)) == 34
    assert resolve_latest_ckpt_step(tmp_path / "missing", ("trainer",)) is None
    # Steps exist but none is complete for these components: raise instead of starting from scratch
    trainer_only = get_step_path(tmp_path / "trainer_only", 10) / "trainer"
    trainer_only.mkdir(parents=True)
    (trainer_only / ".metadata").touch()
    with pytest.raises(FileNotFoundError):
        resolve_latest_ckpt_step(tmp_path / "trainer_only", ("trainer", "orchestrator"))


def test_resolve_latest_ckpt_step_starts_fresh_without_any_trainer_checkpoint(tmp_path):
    # A crash before the first trainer checkpoint finished: the orchestrator already wrote
    # step_40/orchestrator/progress.pt, the trainer save was killed mid-write (or never started).
    ckpt_dir = tmp_path / "checkpoints"
    for file in ["orchestrator/progress.pt", "trainer/__0_0.distcp"]:
        path = get_step_path(ckpt_dir, 40) / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert resolve_latest_ckpt_step(ckpt_dir, ("trainer", "orchestrator")) is None
    assert resolve_latest_ckpt_step(ckpt_dir, ("trainer",)) is None

    orch_only = get_step_path(tmp_path / "orch_only", 40) / "orchestrator"
    orch_only.mkdir(parents=True)
    (orch_only / "progress.pt").touch()
    assert resolve_latest_ckpt_step(tmp_path / "orch_only", ("trainer", "orchestrator")) is None


def _touch_step(ckpt_dir, step, files=("trainer/.metadata", "orchestrator/progress.pt")):
    for file in files:
        path = get_step_path(ckpt_dir, step) / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_resolve_resume_forks_only_until_the_run_has_its_own_checkpoint(tmp_path):
    fork = tmp_path / "other" / "checkpoints" / "step_200"
    _touch_step(fork.parent, 200)
    ckpt_dir = tmp_path / "run" / "checkpoints"
    resume = ResumeConfig(dir=fork)

    # First attempt: no checkpoint dir yet, then only partial step dirs (orchestrator ahead of the trainer,
    # a trainer save killed mid-write): fork
    assert resolve_resume(resume, ckpt_dir) == (200, fork)
    _touch_step(ckpt_dir, 220, files=("orchestrator/progress.pt", "trainer/__0_0.distcp"))
    _touch_step(ckpt_dir, 240, files=("orchestrator/progress.pt",))
    assert resolve_resume(resume, ckpt_dir) == (200, fork)

    # A restart after the run's own checkpoints completed: resume from its newest complete step, not the fork
    _touch_step(ckpt_dir, 220)
    assert resolve_resume(resume, ckpt_dir) == (220, None)
    _touch_step(ckpt_dir, 240)
    _touch_step(ckpt_dir, 260, files=("trainer/.metadata",))  # final step before the orchestrator's teardown save
    assert resolve_resume(resume, ckpt_dir) == (240, None)


def test_resolve_resume_refuses_a_fork_into_a_dir_with_older_checkpoints(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _touch_step(ckpt_dir, 100)
    with pytest.raises(FileExistsError, match="cannot have produced"):
        resolve_resume(ResumeConfig(dir=tmp_path / "other" / "step_200"), ckpt_dir)


def test_resolve_resume_without_dir_is_unchanged(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    assert resolve_resume(None, ckpt_dir) == (None, None)
    assert resolve_resume(ResumeConfig(), ckpt_dir) == (None, None)
    _touch_step(ckpt_dir, 20)
    _touch_step(ckpt_dir, 40)
    assert resolve_resume(ResumeConfig(), ckpt_dir) == (40, None)
    assert resolve_resume(ResumeConfig(step=20), ckpt_dir) == (20, None)
