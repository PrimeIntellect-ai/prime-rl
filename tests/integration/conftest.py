import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Generator

import pytest


@pytest.fixture(autouse=True, scope="module")
def cleanup_zombies():
    """Cleanup zombies in between module tests."""
    subprocess.run(["pkill", "-f", "torchrun"])
    subprocess.run(["pkill", "-f", "VLLM"])
    yield


@pytest.fixture(scope="session")
def user() -> str:
    """Get currrent user from environment."""
    return os.environ.get("USERNAME_CI", os.environ.get("USER", "none"))


@pytest.fixture(scope="session")
def branch_name() -> str:
    """Get current branch name"""
    branch_name_ = os.environ.get("GITHUB_REF_NAME", None)

    if branch_name_ is None:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    else:
        branch_name = branch_name_.replace("/merge", "")
        branch_name = f"pr-{branch_name}"
    return branch_name


@pytest.fixture(scope="session")
def commit_hash() -> str:
    """Get current commit hash"""
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """Create temporary output directory for tests with automatic cleanup"""
    output_dir = Path(os.environ.get("PYTEST_OUTPUT_DIR", tmp_path_factory.mktemp("outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def get_wandb_project(user: str) -> Callable[[str], str]:
    """Factory function to get W&B project name"""

    def _get_wandb_project(name: str) -> str:
        project = f"ci-{name}"
        if user != "CI_RUNNER":
            project += "-local"
        return project

    return _get_wandb_project


Environment = dict[str, str]
Command = list[str]
DEFAULT_TIMEOUT = 120


class ProcessResult:
    """Result object containing process information and captured output."""

    def __init__(self, process: subprocess.Popen):
        self.returncode = process.returncode
        self.pid = process.pid

    def __repr__(self):
        return f"ProcessResult(returncode={self.returncode}, pid={self.pid})"


@pytest.fixture(scope="module")
def run_process() -> Callable[[Command, Environment, int], ProcessResult]:
    """Factory fixture for running a single process."""

    def _run_process(command: Command, env: Environment = {}, timeout: int = DEFAULT_TIMEOUT) -> ProcessResult:
        """Run a subprocess with given command and environment with a timeout"""
        process = subprocess.Popen(command, env={**os.environ, **env})
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)  # Give it 10 seconds to terminate gracefully
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        return ProcessResult(process)

    return _run_process


def strip_escape_codes(text: str) -> str:
    """Helper to strip escape codes from text"""
    return re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def check_number_goes_up_or_down(
    lines: list[str],
    start_step: int = 0,
    end_step: int = -1,
    pattern: str = r"Reward:\s*(\d+\.\d{4})",
    go_up: bool = True,
):
    """Helper to assert that a number in lines goes up from a specified start to end step"""
    step_lines = [line for line in lines if "Step" in line]
    assert len(step_lines) > 0, f"No step lines found in output ({lines})"
    try:
        start_step_line = step_lines[start_step]
    except IndexError:
        start_step_line = ""
    try:
        end_step_line = step_lines[end_step]
    except IndexError:
        end_step_line = ""
    assert start_step_line, f"Could not find start step {start_step} in output ({lines})"
    assert end_step_line, f"Could not find end step {end_step} in output ({lines})"
    start_step_match = re.search(pattern, start_step_line)
    end_step_match = re.search(pattern, end_step_line)
    assert start_step_match is not None, (
        f"Could not find number for start step {start_step} in line {start_step_line} ({lines})"
    )
    assert end_step_match is not None, (
        f"Could not find number for end step {end_step} in line {end_step_line} ({lines})"
    )
    start_step_number = float(start_step_match.group(1))
    end_step_number = float(end_step_match.group(1))
    if go_up:
        assert start_step_number < end_step_number, (
            f"Number did not go up. Found start_number={start_step_number} <= end_number={end_step_number} "
            f"(start line: {start_step_line}, end line: {end_step_line}) ({lines})"
        )
    else:
        assert start_step_number > end_step_number, (
            f"Number did not go down. Found start_number={start_step_number} >= end_number={end_step_number} "
            f"(start line: {start_step_line}, end line: {end_step_line}) ({lines})"
        )


def check_number_in_range(
    lines: list[str],
    step: int = -1,
    min_threshold: float | None = 0.0,
    max_threshold: float | None = None,
    pattern: str = r"Reward:\s*(\d+\.\d{4})",
):
    """Helper to assert that a number in step logs is within a threshold"""
    step_lines = [line for line in lines if "Step" in line]
    assert len(step_lines) > 0, f"No step lines found in output ({lines})"
    try:
        step_line = step_lines[step]
    except IndexError:
        step_line = ""
    assert step_line, f"Could not find step {step} in output ({lines})"
    step_reward = re.search(pattern, step_line)
    assert step_reward is not None, f"Could not find reward for step {step}. Line: {step_line} ({lines})"
    step_reward = float(step_reward.group(1))
    if min_threshold is not None:
        assert step_reward >= min_threshold, (
            f"Reward did not reach minimum threshold. Found reward={step_reward} < {min_threshold} "
            f"(line: {step_line}) ({lines})"
        )
    if max_threshold is not None:
        assert step_reward <= max_threshold, (
            f"Reward did not reach maximum threshold. Found reward={step_reward} > {max_threshold} "
            f"(line: {step_line}) ({lines})"
        )
