import atexit
import os
import shutil
import signal
import socket
import subprocess
from pathlib import Path
from typing import Generator

import pytest

from prime_rl.trainer.world import reset_world
from prime_rl.utils.logger import reset_logger, setup_logger


@pytest.fixture(autouse=True)
def setup_logging():
    """Auto-setup logger across tests"""
    setup_logger("debug")
    yield
    reset_logger()


@pytest.fixture(autouse=True)
def setup_env():
    """Reset environment variables across tests"""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def setup_world():
    """Reset world info across tests"""
    yield
    reset_world()


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """Create temporary output directory for tests with automatic cleanup"""
    output_dir = Path(os.environ.get("PYTEST_OUTPUT_DIR", tmp_path_factory.mktemp("outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    shutil.rmtree(output_dir, ignore_errors=True)


VLLM_SERVER_ENV = {"CUDA_VISIBLE_DEVICES": "0"}
VLLM_SERVER_CMD = ["uv", "run", "inference", "@", "configs/reverse_text/rl/infer.toml", "--max-model-len", "2048"]


def cleanup_process(process: subprocess.Popen):
    process.terminate()

    # Wait for the process to terminate (with timeout)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # If it doesn't terminate gracefully, kill it
        process.kill()
        process.wait()


@pytest.fixture(scope="session")
def vllm_server(output_dir: Path) -> Generator[None, None, None]:
    """Start a vLLM server for integration and e2e tests"""
    import asyncio
    import time
    import urllib.error
    import urllib.request

    # Start the server as a subprocess
    env = {**os.environ, **VLLM_SERVER_ENV}
    with open(output_dir / "vllm.stdout", "w") as stdout, open(output_dir / "vllm.stderr", "w") as stderr:
        vllm_process = subprocess.Popen(
            VLLM_SERVER_CMD,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

    # Register cleanup on unexpected termination
    atexit.register(cleanup_process, vllm_process)
    signal.signal(signal.SIGTERM, lambda signum, frame: cleanup_process(vllm_process))
    signal.signal(signal.SIGINT, lambda signum, frame: cleanup_process(vllm_process))

    # Default vLLM server URL
    base_url = "http://localhost:8000"

    async def wait_for_server_health(timeout: int = 180, interval: int = 1) -> bool:
        """Wait for the server to be healthy by checking the /health endpoint."""
        health_url = f"{base_url}/health"
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < timeout:
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
        cleanup_process(vllm_process)


@pytest.fixture(scope="session")
def vllm_server_dynamic_lora_loading(output_dir: Path) -> Generator[None, None, None]:
    """Start a vLLM server for integration and e2e tests"""
    import asyncio
    import time
    import urllib.error
    import urllib.request

    # Start the server as a subprocess
    env = {**os.environ, **VLLM_SERVER_ENV, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    with open(output_dir / "vllm.stdout", "w") as stdout, open(output_dir / "vllm.stderr", "w") as stderr:
        vllm_process = subprocess.Popen(
            VLLM_SERVER_CMD + ["--enable-lora"],
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

    # Register cleanup on unexpected termination
    atexit.register(cleanup_process, vllm_process)
    signal.signal(signal.SIGTERM, lambda signum, frame: cleanup_process(vllm_process))
    signal.signal(signal.SIGINT, lambda signum, frame: cleanup_process(vllm_process))

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
        cleanup_process(vllm_process)


@pytest.fixture()
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
