import concurrent.futures
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Generator

import pytest
from huggingface_hub import HfApi
from loguru import logger

from prime_rl.trainer.world import reset_world
from prime_rl.utils.logger import reset_logger, set_logger

TIMEOUT = 120


Environment = dict[str, str]
Command = list[str]


@pytest.fixture(autouse=True)
def setup_logger():
    """
    Fixture to set and reset the logger after each test.
    """
    set_logger(logger)  # Use the default loguru.logger
    yield
    reset_logger()


@pytest.fixture(autouse=True)
def setup_env():
    """
    Fixture to reset environment variables after each test.
    """
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def setup_world():
    """
    Fixture to reset the world info after each test.
    """
    yield
    reset_world()


@pytest.fixture(scope="session")
def outputs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    outputs_dir = Path(os.environ.get("PYTEST_OUTPUTS_DIR", tmp_path_factory.mktemp("outputs")))
    yield outputs_dir
    shutil.rmtree(outputs_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def hf_api() -> HfApi:
    """Hugging Face API to use for tests."""
    return HfApi()


@pytest.fixture(scope="module")
def username() -> str:
    return os.environ.get("USERNAME_CI", os.getlogin())


@pytest.fixture(scope="module")
def branch_name() -> str:
    branch_name_ = os.environ.get("GITHUB_REF_NAME", None)

    if branch_name_ is None:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    else:
        branch_name = branch_name_.replace("/merge", "")
        branch_name = f"pr-{branch_name}"
    return branch_name


@pytest.fixture(scope="module")
def commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()


class ProcessResult:
    def __init__(self, returncode: int, pid: int):
        self.returncode = returncode
        self.pid = pid


def run_subprocess(command: Command, env: Environment, timeout: int = TIMEOUT) -> ProcessResult:
    """Run a subprocess with given command and environment with a timeout"""
    try:
        process = subprocess.Popen(command, env={**os.environ, **env})
        process.wait(timeout=timeout)
        return ProcessResult(process.returncode, process.pid)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)  # Give it 10 seconds to terminate gracefully
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return ProcessResult(1, process.pid)


def run_subprocesses_in_parallel(
    commands: list[Command], envs: list[Environment], timeout: int = TIMEOUT
) -> list[ProcessResult]:
    """Start multiple processes in parallel using ProcessPoolExecutor and wait for completion."""
    assert len(commands) == len(envs), "Should have an environment for each command"
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(run_subprocess, cmd, env, timeout) for cmd, env in zip(commands, envs)]
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Process {i} did not complete within {timeout} seconds")

    return results


@pytest.fixture(scope="module")
def run_process() -> Callable[[Command, Environment, int], ProcessResult]:
    """Factory fixture for running a single process."""
    return run_subprocess


@pytest.fixture(scope="module")
def run_processes() -> Callable[[list[Command], list[Environment], int], list[ProcessResult]]:
    """Factory fixture for running multiple processes in parallel."""
    return run_subprocesses_in_parallel


VLLM_SERVER_ENV = {"CUDA_VISIBLE_DEVICES": "0"}
VLLM_SERVER_CMD = ["uv", "run", "inference", "@", "configs/reverse_text/infer.toml", "--max-model-len", "2048"]


@pytest.fixture(scope="session")
def vllm_server() -> Generator[None, None, None]:
    """Start a vLLM server for integration and e2e tests"""
    import asyncio
    import time
    import urllib.error
    import urllib.request

    # Start the server as a subprocess
    env = {**os.environ, **VLLM_SERVER_ENV}
    process = subprocess.Popen(VLLM_SERVER_CMD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Default vLLM server URL
    base_url = "http://localhost:8000"

    async def wait_for_server_health(timeout: int = 180, interval: int = 1) -> bool:
        """Wait for the server to be healthy by checking the /health endpoint."""
        health_url = f"{base_url}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass
            await asyncio.sleep(interval)

        return False

    try:
        # Wait for the server to be healthy
        is_healthy = asyncio.run(wait_for_server_health())

        if not is_healthy:
            raise RuntimeError("vLLM server did not become healthy within timeout")

        # Yield to signal that the server is ready (can be used in tests that depend on it)
        yield
    finally:
        # Shut down the server gracefully
        logger.info("Shutting down vLLM server")
        process.terminate()

        # Wait for the process to terminate (with timeout)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # If it doesn't terminate gracefully, kill it
            process.kill()
            process.wait()
