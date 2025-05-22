from pathlib import Path
from subprocess import Popen
from typing import Callable

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "torchrun", "src/zeroband/train.py", "@configs/training/debug.toml"]


@pytest.fixture(scope="module")
def output_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_infer_debug")


@pytest.fixture(scope="module")
def process(output_path: Path, run_process: Callable[[list[str]], Popen]):
    return run_process(CMD + ["--ckpt.path", str(output_path)])


def test_no_error(process: Popen):
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"


def test_output_directories_exist(output_path: Path):
    assert output_path.exists()
    assert not (output_path / "step_0").exists()
    assert (output_path / "step_1").exists()
    assert (output_path / "step_2").exists()
    assert not (output_path / "step_3").exists()
