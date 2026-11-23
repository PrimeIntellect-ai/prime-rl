import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Generator

import httpx
import pytest

from prime_rl.utils.process import cleanup_process

REF_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
REF_PORT = 8001
REF_READY_TIMEOUT_S = 300


@pytest.fixture(scope="module")
def wandb_project(get_wandb_project: Callable[[str], str]) -> str:
    """Fixture for W&B project name for integration tests. This means all integration tests will use the same W&B project."""
    return get_wandb_project("integration-tests")


@pytest.fixture(scope="module")
def ref_inference(output_dir: Path) -> Generator[subprocess.Popen, None, None]:
    """Spawn a `uv run inference` frozen reference server on GPU 0 (shared with the rl-launched
    policy) at 40% gpu_memory_utilization. Tears down at module scope.
    """
    # The rl entrypoint's --clean wipes the rl output_dir on start,
    # so park the reference-server log next to it instead of inside it.
    ref_log_dir = output_dir.parent / f"{output_dir.name}_ref"
    ref_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = ref_log_dir / "ref_inference.log"
    cmd = [
        "uv",
        "run",
        "inference",
        "--vllm.model",
        REF_MODEL,
        "--server.port",
        str(REF_PORT),
        "--vllm.gpu-memory-utilization",
        "0.4",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    try:
        url = f"http://localhost:{REF_PORT}/v1/models"
        deadline = time.time() + REF_READY_TIMEOUT_S
        while True:
            try:
                if httpx.get(url, timeout=2.0).status_code == 200:
                    break
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Reference inference server at {url} did not become ready within {REF_READY_TIMEOUT_S}s"
                )
            time.sleep(1.0)
        yield proc
    finally:
        cleanup_process(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            cleanup_process(proc.pid, signal.SIGKILL)
            proc.wait()
