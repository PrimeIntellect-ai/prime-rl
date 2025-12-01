from typing import Callable

import pytest

from tests.integration.conftest import Command, Environment, ProcessResult, check_zero_return_code

pytestmark = [pytest.mark.slow]


@pytest.fixture(scope="module")
def single_env_eval_process(run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
    """Fixture for running single-env eval CI integration test"""
    cmd = ["uv", "run", "eval", "@", "configs/ci/integration/eval/single_env.toml"]
    return run_process(cmd, {})


@pytest.fixture(scope="module")
def multi_env_eval_process(run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
    """Fixture for running multi-env eval CI integration test"""
    cmd = ["uv", "run", "eval", "@", "configs/ci/integration/eval/multi_env.toml"]
    return run_process(cmd, {})


def test_no_error_single_env(single_env_eval_process: ProcessResult):
    """Tests that the single environment eval process does not fail."""
    check_zero_return_code(single_env_eval_process)


def test_no_error_multi_env(multi_env_eval_process: ProcessResult):
    """Tests that the multi environment eval process does not fail."""
    check_zero_return_code(multi_env_eval_process)
