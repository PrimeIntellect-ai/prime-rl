from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow]


@pytest.fixture(scope="module")
def single_env_eval_process(
    run_process: Callable[[Command, Environment], ProcessResult], output_dir: Path
) -> ProcessResult:
    """Fixture for running single-env eval CI integration test"""
    cmd = [
        "uv",
        "run",
        "eval",
        "@",
        "configs/ci/integration/eval/single_env.toml",
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, {})


@pytest.fixture(scope="module")
def multi_env_eval_process(
    run_process: Callable[[Command, Environment], ProcessResult], output_dir: Path
) -> ProcessResult:
    """Fixture for running multi-env eval CI integration test"""
    cmd = [
        "uv",
        "run",
        "eval",
        "@",
        "configs/ci/integration/eval/multi_env.toml",
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, {})


def test_no_error_single_env(single_env_eval_process: ProcessResult):
    """Tests that the single environment eval process does not fail."""
    assert single_env_eval_process.returncode == 0, f"Process has non-zero return code ({single_env_eval_process})"


def test_no_error_multi_env(multi_env_eval_process: ProcessResult):
    """Tests that the multi environment eval process does not fail."""
    assert multi_env_eval_process.returncode == 0, f"Process has non-zero return code ({multi_env_eval_process})"
