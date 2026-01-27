import os
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import Command, Environment, ProcessResult

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]


@pytest.fixture(scope="module")
def single_env_synthesize_process(
    run_process: Callable[[Command, Environment], ProcessResult], output_dir: Path
) -> ProcessResult:
    """Fixture for running single-env synthesize CI integration test"""
    cmd = [
        "uv",
        "run",
        "synthesize",
        "@",
        "configs/ci/integration/synthesize/single_env.toml",
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, {})


@pytest.fixture(scope="module")
def multi_env_synthesize_process(
    run_process: Callable[[Command, Environment], ProcessResult], output_dir: Path
) -> ProcessResult:
    """Fixture for running multi-env synthesize CI integration test"""
    cmd = [
        "uv",
        "run",
        "synthesize",
        "@",
        "configs/ci/integration/synthesize/multi_env.toml",
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, {})


@pytest.fixture(scope="module")
def multi_turn_tool_call_synthesize_process(
    run_process: Callable[[Command, Environment], ProcessResult],
    output_dir: Path,
) -> ProcessResult:
    """Fixture for running multi-turn tool call synthesize CI integration test"""
    cmd = [
        "uv",
        "run",
        "synthesize",
        "@",
        "configs/ci/integration/synthesize/multi_turn_tool_call.toml",
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, {})


def test_no_error_single_env(single_env_synthesize_process: ProcessResult):
    """Tests that the single environment synthesize process does not fail."""
    assert single_env_synthesize_process.returncode == 0, (
        f"Process has non-zero return code ({single_env_synthesize_process})"
    )


def test_no_error_multi_env(multi_env_synthesize_process: ProcessResult):
    """Tests that the multi-env synthesize process does not fail."""
    assert multi_env_synthesize_process.returncode == 0, (
        f"Process has non-zero return code ({multi_env_synthesize_process})"
    )


def test_no_error_multi_turn_tool_call(multi_turn_tool_call_synthesize_process: ProcessResult):
    """Tests that the multi-turn tool call synthesize process does not fail."""
    assert multi_turn_tool_call_synthesize_process.returncode == 0, (
        f"Process has non-zero return code ({multi_turn_tool_call_synthesize_process})"
    )
