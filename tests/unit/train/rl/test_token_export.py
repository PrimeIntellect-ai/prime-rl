from pathlib import Path

import pytest

from prime_rl.trainer.rl import token_export
from prime_rl.trainer.rl.token_export import _mkdir_existing_dir_ok


def test_mkdir_existing_dir_ok_retries_transient_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "token_exports" / "step_31"
    original_mkdir = Path.mkdir
    calls = 0

    def flaky_mkdir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        nonlocal calls
        if self == target and calls == 0:
            calls += 1
            raise FileExistsError(str(self))
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    def create_dir_during_retry(_: float) -> None:
        original_mkdir(target, parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)
    monkeypatch.setattr(token_export.time, "sleep", create_dir_during_retry)

    _mkdir_existing_dir_ok(target)

    assert target.is_dir()
    assert calls == 1


def test_mkdir_existing_dir_ok_raises_when_path_is_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "token_exports" / "step_31"
    target.parent.mkdir(parents=True)
    target.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(token_export.time, "sleep", lambda _: None)

    with pytest.raises(FileExistsError):
        _mkdir_existing_dir_ok(target)
