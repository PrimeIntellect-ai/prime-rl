from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_avg_mismatch_kl_in_range, check_no_error, check_reward_goes_up, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 900

# Pull streams each version from the trainer through bounded host buffers into vLLM's loader.
MODE_ARGS = {
    "push": [],
    "pull": ["--weight-broadcast.mode", "pull", "--weight-broadcast.staging-buffer-bytes", "1073741824"],
}


@pytest.fixture(scope="module", params=sorted(MODE_ARGS))
def mode(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def run_name(mode: str) -> str:
    return f"reverse-text-modelexpress-{mode}"


@pytest.fixture(scope="module")
def run_dir(output_dir: Path, run_name: str) -> Path:
    return output_dir / run_name


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    branch_name: str,
    mode: str,
    run_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse-text/start.toml",
        "--weight-broadcast.type",
        "modelexpress",
        *MODE_ARGS[mode],
        "--clean",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        f"test-{run_name}:{branch_name}",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        run_name,
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, run_dir: Path):
    check_no_error(rl_process, run_dir)


def test_reward_goes_up(rl_process: ProcessResult, test_no_error, run_dir: Path):
    with open(run_dir / "logs" / "latest" / "orchestrator.log", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)


def test_mismatch_kl_in_range(rl_process: ProcessResult, test_no_error, run_dir: Path):
    """A low trainer/inference mismatch KL shows inference serves the weights ModelExpress delivered."""
    with open(run_dir / "logs" / "latest" / "trainer.log", "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_avg_mismatch_kl_in_range(trainer_stdout, last_n_steps=3, max_threshold=0.01)
