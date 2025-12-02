import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Callable, Generator

import pytest

from prime_rl.trainer.world import reset_world
from prime_rl.utils.logger import reset_logger, setup_logger


@pytest.fixture(autouse=True)
def setup_logging():
    """Auto-fixture to setup logger between tests"""
    setup_logger("debug")
    yield
    reset_logger()


@pytest.fixture(autouse=True)
def setup_env():
    """Auto-fixture to reset environment variables between tests"""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def setup_world():
    """Auto-fixture to reset the world between tests."""
    yield
    reset_world()


@pytest.fixture(autouse=True, scope="module")
def cleanup_zombies():
    """Auto-fixture to cleanup zombies between module tests. Used in CI to avoid zombie processes from previous tests."""
    subprocess.run(["pkill", "-f", "torchrun"])
    subprocess.run(["pkill", "-f", "VLLM"])
    yield


@pytest.fixture
def free_port() -> int:
    """Fixture to get a free port per tests"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def user() -> str:
    """Fixture for current user from environment for test session."""
    return os.environ.get("USERNAME_CI", os.environ.get("USER", "none"))


@pytest.fixture(scope="session")
def branch_name() -> str:
    """Fixture for current branch name for test session."""
    branch_name_ = os.environ.get("GITHUB_REF_NAME", None)

    if branch_name_ is None:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    else:
        branch_name = branch_name_.replace("/merge", "")
        branch_name = f"pr-{branch_name}"
    return branch_name


@pytest.fixture(scope="session")
def commit_hash() -> str:
    """Fixture for current commit hash for test session."""
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """Fixture for temporary output directory for tests with automatic cleanup"""
    output_dir = Path(os.environ.get("PYTEST_OUTPUT_DIR", tmp_path_factory.mktemp("outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def get_wandb_project(user: str) -> Callable[[str], str]:
    """Factory fixture to get W&B project name. Used to setup shared W&B projects for integration & nightly tests."""

    def _get_wandb_project(wandb_project: str) -> str:
        if user != "CI_RUNNER":
            wandb_project += "-local"
        return wandb_project

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
