import concurrent.futures
import os
import subprocess
from typing import Callable

import pytest

TIMEOUT = 120


Environment = dict[str, str]
Command = list[str]


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
    except Exception as e:
        raise e


def run_subprocesses_in_parallel(commands: list[Command], envs: list[Environment]) -> list[ProcessResult]:
    """Start multiple processes in parallel using ProcessPoolExecutor and wait for completion."""
    assert len(commands) == len(envs), "Should have an environment for each command"
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(run_subprocess, cmd, env) for cmd, env in zip(commands, envs)]
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=TIMEOUT)
                results.append(result)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Process {i} did not complete within {TIMEOUT + 30} seconds")

    return results


@pytest.fixture(scope="module")
def run_process() -> Callable[[Command, Environment], ProcessResult]:
    """Factory fixture for running a single process."""
    return run_subprocess


@pytest.fixture(scope="module")
def run_processes() -> Callable[[list[Command], list[Environment]], list[ProcessResult]]:
    """Factory fixture for running multiple processes in parallel. Used for parallel inference tests and RL training tests."""
    return run_subprocesses_in_parallel
