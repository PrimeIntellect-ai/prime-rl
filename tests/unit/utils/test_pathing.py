import pytest

from prime_rl.utils.pathing import validate_output_dir


def test_nonexistent_dir_passes(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_empty_dir_raises(tmp_path):
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_only_logs_raises(tmp_path):
    output_dir = tmp_path / "has_logs"
    output_dir.mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "logs" / "trainer").mkdir(parents=True)
    (output_dir / "logs" / "trainer" / "rank_0.log").touch()
    with pytest.raises(FileExistsError, match="already exists"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_existing_dir_raises(tmp_path):
    output_dir = tmp_path / "exists"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_existing_dir_passes_when_resuming(tmp_path):
    output_dir = tmp_path / "exists"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    validate_output_dir(output_dir, resuming=True, clean=False)


def test_existing_dir_cleaned_when_flag_set(tmp_path):
    output_dir = tmp_path / "exists"
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
