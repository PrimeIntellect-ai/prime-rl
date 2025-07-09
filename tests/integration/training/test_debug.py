from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "trainer", "@configs/trainer/debug.toml"]


@pytest.fixture(scope="module")
def train_process(run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
    return run_process(CMD, {})


def test_no_error(train_process: ProcessResult):
    assert train_process.returncode == 0, f"Train process failed with return code {train_process.returncode}"
