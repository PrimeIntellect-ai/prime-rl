from typing import Callable

import pytest

from tests.integration import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "torchrun", "src/zeroband/train.py", "@configs/training/debug.toml"]


@pytest.fixture(scope="module")
def process(run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
    return run_process(CMD, {})


def test_no_error(process: ProcessResult):
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"
