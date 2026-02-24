import pytest

from prime_rl.utils.pathing import validate_output_dir


def test_nonexistent_dir_passes(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_nonempty_dir_raises(tmp_path):
    output_dir = tmp_path / "nonempty"
    output_dir.mkdir()
    (output_dir / "checkpoint").touch()
    with pytest.raises(FileExistsError, match="already exists and is not empty"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_nonempty_dir_passes_when_resuming(tmp_path):
    output_dir = tmp_path / "nonempty"
    output_dir.mkdir()
    (output_dir / "checkpoint").touch()
    validate_output_dir(output_dir, resuming=True, clean=False)


def test_nonempty_dir_cleaned_when_flag_set(tmp_path):
    output_dir = tmp_path / "nonempty"
    output_dir.mkdir()
    (output_dir / "checkpoint").touch()
    (output_dir / "subdir").mkdir()
    (output_dir / "subdir" / "file").touch()

    validate_output_dir(output_dir, resuming=False, clean=True)

    assert not output_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=True)
    assert not output_dir.exists()
