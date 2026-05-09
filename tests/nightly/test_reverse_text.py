from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error, check_reward_goes_up, check_reward_in_range, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def inference_backend(request) -> str:
    return request.param


def pytest_generate_tests(metafunc):
    if "inference_backend" in metafunc.fixturenames:
        metafunc.parametrize("inference_backend", ["vllm", "sglang"], indirect=True)


@pytest.fixture(scope="module")
def backend_output_dir(output_dir: Path, inference_backend: str) -> Path:
    path = output_dir / f"reverse_text_{inference_backend}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def wandb_name(branch_name: str, inference_backend: str) -> str:
    """Fixture for W&B name for RL CI integration tests."""
    return f"reverse-text-{inference_backend}-{branch_name}"


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    backend_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
    inference_backend: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "examples/reverse_text/rl.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--wandb.shared",
        "False",
        "--inference.backend",
        inference_backend,
        "--output-dir",
        backend_output_dir.as_posix(),
    ]
    return run_process(cmd)


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, backend_output_dir: Path):
    """Tests that the RL process does not fail."""
    check_no_error(rl_process, backend_output_dir)


def test_reward_goes_up(rl_process: ProcessResult, test_no_error, backend_output_dir: Path):
    """Tests that the reward goes up in the RL process"""
    with open(backend_output_dir / "logs" / "orchestrator.log", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)


def test_reward_reaches_threshold(rl_process: ProcessResult, test_no_error, backend_output_dir: Path):
    """Tests that the reward goes up in the RL process"""
    with open(backend_output_dir / "logs" / "orchestrator.log", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.65)
