import subprocess
from typing import Callable, Generator

import pytest

TIMEOUT = 120


@pytest.fixture(scope="module")
def run_process() -> Generator[Callable[[list[str]], subprocess.Popen], None, None]:
    """Start a process and wait for it to complete."""

    process = None

    def start_process(command: list[str]) -> subprocess.Popen:
        nonlocal process
        process = subprocess.Popen(command)

        try:
            process.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()
            raise TimeoutError(f"Process did not complete within {TIMEOUT} seconds")

        return process

    yield start_process

    if process is not None and process.poll() is None:
        process.terminate()
        process.wait()
