from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

ENV = {"CUDA_VISIBLE_DEVICES": "1"}
CMD = ["uv", "run", "trainer", "@", "configs/debug/train.toml"]


@pytest.fixture(scope="module")
def train_process(run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
    return run_process(CMD, ENV)


def test_no_error(train_process: ProcessResult):
    assert train_process.returncode == 0, f"Train process failed with return code {train_process.returncode}"
