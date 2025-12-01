import os
import re
import subprocess
from typing import Callable

import pytest


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

    def __init__(self, process: subprocess.Popen, stdout: bytes, stderr: bytes):
        self.returncode = process.returncode
        self.stdout = stdout.decode("utf-8", errors="replace")
        self.stderr = stderr.decode("utf-8", errors="replace")
        self._process = process

    @staticmethod
    def strip_escape_codes(text: str) -> str:
        return re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)

    @property
    def readable_stdout(self) -> list[str]:
        return self.strip_escape_codes(self.stdout).splitlines()

    @property
    def readable_stderr(self) -> list[str]:
        return self.strip_escape_codes(self.stderr).splitlines()

    def __repr__(self):
        return (
            f"ProcessResult(returncode={self.returncode}, stdout={self.readable_stdout}, stderr={self.readable_stderr})"
        )


@pytest.fixture(scope="module")
def run_process() -> Callable[[Command, Environment, int], ProcessResult]:
    """Factory fixture for running a single process."""

    def _run_process(command: Command, env: Environment, timeout: int = DEFAULT_TIMEOUT) -> ProcessResult:
        """Run a subprocess with given command and environment with a timeout"""
        process = subprocess.Popen(command, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)  # Give it 10 seconds to terminate gracefully
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        # Read stdout and stderr after process has finished or been killed
        stdout, stderr = process.communicate()

        return ProcessResult(process, stdout, stderr)

    return _run_process


def check_zero_return_code(process: ProcessResult):
    """Helper to assert that a process has a zero return code"""
    assert process.returncode == 0, f"Process has non-zero return code ({process})"


def check_loss_goes_down(process: ProcessResult, loss_pattern: str = r"Loss:\s*(\d+\.\d{4})"):
    """Helper to assert that the last step's loss is less than the first step's loss"""
    lines = process.readable_stdout
    step_lines = [line for line in lines if "Step" in line]
    assert len(step_lines) > 0, f"No step lines found in output ({process})"
    first_match = re.search(loss_pattern, step_lines[0])
    last_match = re.search(loss_pattern, step_lines[-1])
    assert first_match is not None and last_match is not None, (
        f"Could not find loss in step lines. First line: {step_lines[0]}, Last line: {step_lines[-1]} ({process})"
    )
    first_step_loss = float(first_match.group(1))
    last_step_loss = float(last_match.group(1))
    assert first_step_loss > last_step_loss, (
        f"Loss did not go down. Found first_loss={first_step_loss} >= last_loss={last_step_loss} "
        f"(first line: {step_lines[0]}, last line: {step_lines[-1]}) ({process})"
    )
