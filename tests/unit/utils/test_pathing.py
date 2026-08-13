import pytest

from prime_rl.utils.pathing import validate_run_dir


def test_nonexistent_dir_passes(tmp_path):
    run_dir = tmp_path / "does_not_exist"
    validate_run_dir(run_dir, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    validate_run_dir(run_dir, resuming=False, clean=False)


def test_dir_with_launcher_artifacts_passes(tmp_path):
    run_dir = tmp_path / "submitted"
    run_dir.mkdir()
    (run_dir / "configs").mkdir()
    (run_dir / "configs" / "trainer.toml").touch()
    (run_dir / "rl.sbatch").touch()
    (run_dir / "job_1234.log").touch()
    validate_run_dir(run_dir, resuming=False, clean=False)


def test_dir_with_logs_raises(tmp_path):
    run_dir = tmp_path / "has_logs"
    run_dir.mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "trainer.log").touch()
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        validate_run_dir(run_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_raises(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        validate_run_dir(run_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_passes_when_resuming(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    validate_run_dir(run_dir, resuming=True, clean=False)


def test_dir_with_checkpoints_cleaned_when_flag_set(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    (run_dir / "logs").mkdir()

    validate_run_dir(run_dir, resuming=False, clean=True)

    assert not run_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    run_dir = tmp_path / "does_not_exist"
    validate_run_dir(run_dir, resuming=False, clean=True)
    assert not run_dir.exists()


def test_ckpt_output_dir_with_checkpoints_raises(tmp_path):
    run_dir = tmp_path / "fresh"
    ckpt_output_dir = tmp_path / "ckpts"
    ckpt_output_dir.mkdir()
    (ckpt_output_dir / "checkpoints").mkdir()
    (ckpt_output_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_run_dir(run_dir, resuming=False, clean=False, ckpt_output_dir=ckpt_output_dir)
