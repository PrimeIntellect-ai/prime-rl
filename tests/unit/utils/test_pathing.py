import pytest

from prime_rl.utils.pathing import get_log_dir, setup_log_dir, validate_output_dir


def test_nonexistent_dir_passes(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_only_logs_passes(tmp_path):
    output_dir = tmp_path / "has_logs"
    output_dir.mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "logs" / "trainer.log").touch()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_raises(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_passes_when_resuming(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    validate_output_dir(output_dir, resuming=True, clean=False)


def test_dir_with_checkpoints_cleaned_when_flag_set(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    (output_dir / "logs").mkdir()

    validate_output_dir(output_dir, resuming=False, clean=True)

    assert not output_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=True)
    assert not output_dir.exists()


def test_setup_log_dir_creates_symlink(tmp_path):
    """setup_log_dir creates runs/<uuid>/ and symlinks logs to it."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    log_dir = setup_log_dir(output_dir)

    assert log_dir == output_dir / "logs"
    assert log_dir.is_symlink()
    target = log_dir.resolve()
    assert target.parent == (output_dir / "runs").resolve()
    assert target.is_dir()
    # The symlink target should be a UUID directory under runs/
    assert target.parent.name == "runs"
    assert len(target.name) == 32  # uuid4().hex length


def test_setup_log_dir_writable_through_symlink(tmp_path):
    """Files written through the symlink land in the UUID directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    log_dir = setup_log_dir(output_dir)
    (log_dir / "trainer.log").write_text("test")

    target = log_dir.resolve()
    assert (target / "trainer.log").read_text() == "test"
    assert (log_dir / "trainer.log").read_text() == "test"


def test_setup_log_dir_creates_fresh_uuid_each_call(tmp_path):
    """Each call to setup_log_dir creates a new UUID run directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    log_dir_1 = setup_log_dir(output_dir)
    target_1 = log_dir_1.resolve()

    # Write a file to the first run
    (log_dir_1 / "trainer.log").write_text("run1")

    log_dir_2 = setup_log_dir(output_dir)
    target_2 = log_dir_2.resolve()

    # New UUID directory
    assert target_1 != target_2
    # Both directories exist
    assert target_1.is_dir()
    assert target_2.is_dir()
    # Symlink now points to the second run
    assert log_dir_2.resolve() == target_2
    # First run's files are preserved
    assert (target_1 / "trainer.log").read_text() == "run1"


def test_setup_log_dir_resuming_reuses_existing(tmp_path):
    """When resuming and symlink exists, reuse the existing run directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    log_dir_1 = setup_log_dir(output_dir)
    target_1 = log_dir_1.resolve()
    (log_dir_1 / "trainer.log").write_text("run1")

    # Resume should reuse the same directory
    log_dir_2 = setup_log_dir(output_dir, resuming=True)
    assert log_dir_2.resolve() == target_1
    assert (log_dir_2 / "trainer.log").read_text() == "run1"


def test_setup_log_dir_resuming_without_symlink_creates_new(tmp_path):
    """When resuming but no symlink exists, create a new run directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    log_dir = setup_log_dir(output_dir, resuming=True)
    assert log_dir.is_symlink()
    assert log_dir.resolve().is_dir()


def test_get_log_dir_returns_symlink_path(tmp_path):
    """get_log_dir returns output_dir/logs (the symlink path)."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    setup_log_dir(output_dir)
    log_dir = get_log_dir(output_dir)
    assert log_dir == output_dir / "logs"
    assert log_dir.is_symlink()
