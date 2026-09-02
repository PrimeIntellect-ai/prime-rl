import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Generator

import httpx
import pytest

from prime_rl.utils.process import cleanup_process
from tests.conftest import ProcessResult
from tests.utils import check_no_error, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

RUN_NAME = "reverse-text-eval"
TIMEOUT = 600
INFERENCE_PORT = 8000
INFERENCE_READY_TIMEOUT_S = 300


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


def _wait_for_inference(port: int, timeout_s: int) -> None:
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Inference server at {url} did not become ready within {timeout_s}s")


@pytest.fixture(scope="module")
def inference(output_dir: Path) -> Generator[subprocess.Popen, None, None]:
    """A `uv run inference` server for the trained reverse-text model; the eval adapts its
    concurrency to this server's vLLM metrics."""
    log_dir = output_dir.parent / f"{output_dir.name}_inference"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "inference",
        "--vllm.model",
        "PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL",
        "--server.port",
        str(INFERENCE_PORT),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    with open(log_dir / "inference.log", "w") as log_file:
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    try:
        _wait_for_inference(INFERENCE_PORT, INFERENCE_READY_TIMEOUT_S)
        yield proc
    finally:
        cleanup_process(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            cleanup_process(proc.pid, signal.SIGKILL)
            proc.wait()


@pytest.fixture(scope="module")
def eval_process(inference, run_process: Callable[..., ProcessResult], output_dir: Path) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "eval",
        "@",
        "configs/ci/integration/reverse-text-eval.toml",
        "--clean",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(eval_process: ProcessResult, run_dir: Path):
    check_no_error(eval_process, run_dir)


def test_eval_reward(eval_process: ProcessResult, test_no_error, run_dir: Path):
    with open(run_dir / "logs" / "latest" / "eval.log", "r") as f:
        lines = strip_escape_codes(f.read()).splitlines()
    pattern = r"Evaluated reverse-text .*Reward\s+(\d+\.\d{4})"
    matches = [re.search(pattern, line) for line in lines if "SUCCESS" in line]
    matches = [m for m in matches if m]
    assert len(matches) == 1, f"Expected one eval summary line, found {len(matches)}"
    assert float(matches[0].group(1)) >= 0.5


def test_run_artifacts(eval_process: ProcessResult, test_no_error, run_dir: Path):
    assert (run_dir / "configs" / "latest" / "resolved" / "eval.json").is_file()
    assert (run_dir / "monitors" / "file" / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "step_16" / "eval" / "progress.pt").is_file()
