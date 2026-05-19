import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Generator

import httpx
import pytest

from prime_rl.utils.process import cleanup_process
from tests.conftest import ProcessResult
from tests.utils import check_loss_goes_down, check_no_error, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 600  # 10 minutes
TEACHER_PORT = 8001
TEACHER_READY_TIMEOUT_S = 300


def _wait_for_teacher(port: int, timeout_s: int) -> None:
    """Block until the teacher inference server's /v1/models endpoint is reachable."""
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Teacher inference server at {url} did not become ready within {timeout_s}s")


@pytest.fixture(scope="module")
def teacher_inference(output_dir: Path) -> Generator[subprocess.Popen, None, None]:
    """Spawn a minimal `uv run inference` teacher on GPU 0 (shared with the
    rl-launched student). gpu_memory_utilization is kept low (10%) so the
    student vLLM — which starts after — still has enough headroom for its
    own model + KV cache on the same GPU. Tears down at module scope.
    """
    # The rl entrypoint's --clean-output-dir wipes the rl output_dir on start,
    # so park the teacher log next to it instead of inside it.
    teacher_log_dir = output_dir.parent / f"{output_dir.name}_teacher"
    teacher_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = teacher_log_dir / "teacher_inference.log"
    cmd = [
        "uv",
        "run",
        "inference",
        "--model.name",
        "PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL",
        "--server.port",
        str(TEACHER_PORT),
        "--model.enforce-eager",
        "--gpu-memory-utilization",
        "0.1",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    try:
        _wait_for_teacher(TEACHER_PORT, TEACHER_READY_TIMEOUT_S)
        yield proc
    finally:
        cleanup_process(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            cleanup_process(proc.pid, signal.SIGKILL)
            proc.wait()


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-reverse-text-rl-opd:{branch_name}"


@pytest.fixture(scope="module")
def rl_opd_process(
    teacher_inference,
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    """Run the RL entrypoint with training_mode = "opd"; teacher_inference is
    a fixture-managed external vLLM at http://localhost:8001/v1."""
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse_text_rl_opd/start.toml",
        "--clean-output-dir",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(rl_opd_process: ProcessResult, output_dir: Path):
    check_no_error(rl_opd_process, output_dir)


def test_loss_goes_down(rl_opd_process: ProcessResult, test_no_error, output_dir: Path):
    with open(output_dir / "logs" / "trainer.log", "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)
