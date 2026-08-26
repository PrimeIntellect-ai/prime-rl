import pytest

from prime_rl.utils.pathing import (
    clean_future_steps,
    create_attempt_dirs,
    get_broadcast_dir,
    get_rollout_dir,
    get_step_path,
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
    rollout_dir = get_rollout_dir(tmp_path)
    broadcast_dir = get_broadcast_dir(tmp_path)
    for parent in (rollout_dir, broadcast_dir):
        for step in (1, 2, 3):
            get_step_path(parent, step).mkdir(parents=True)

    clean_future_steps(tmp_path, resume_step=2)

    assert get_step_path(rollout_dir, 1).exists()
    assert get_step_path(rollout_dir, 2).exists()
    assert not get_step_path(rollout_dir, 3).exists()
    assert get_step_path(broadcast_dir, 1).exists()
    assert not get_step_path(broadcast_dir, 2).exists()
    assert not get_step_path(broadcast_dir, 3).exists()
