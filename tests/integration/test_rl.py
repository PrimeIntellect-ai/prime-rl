from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_number_goes_up_or_down, check_number_in_range, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


TIMEOUT = 600  # 10 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str, commit_hash: str) -> str:
    """Fixture for W&B name for RL CI integration tests."""
    return f"rl-{branch_name}-{commit_hash}"


@pytest.fixture(scope="module")
def rl_process(
    vllm_server,  # Can only run with vLLM server
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl/start.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def rl_resume_process(
    vllm_server,  # Can only run with vLLM server
    rl_process,  # Resume training can only start when regular RL process is finished
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    if rl_process.returncode != 0:
        pytest.skip("Full weight RL process failed")
    wandb_name = f"{wandb_name}-resume"
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl/resume.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]

    return run_process(cmd, timeout=TIMEOUT)


check_reward_goes_up = partial(check_number_goes_up_or_down, go_up=True, pattern=r"Reward:\s*(\d+\.\d{4})")
check_reward_in_range = partial(check_number_in_range, pattern=r"Reward:\s*(\d+\.\d{4})")


def test_no_error(rl_process: ProcessResult):
    """Tests that the RL process does not fail."""
    assert rl_process.returncode == 0, f"Process has non-zero return code ({rl_process})"


def test_reward_goes_up(rl_process: ProcessResult, output_dir: Path):
    """Tests that the reward goes up in the RL process"""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)


def test_reward_in_range(rl_process: ProcessResult, output_dir: Path):
    """Tests that the reward is in range in the RL process"""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.65)


def test_no_error_resume(rl_resume_process: ProcessResult):
    """Tests that the RL resume process does not fail."""
    assert rl_resume_process.returncode == 0, f"Process has non-zero return code ({rl_resume_process})"


def test_reward_in_range_resume(rl_resume_process: ProcessResult, output_dir: Path):
    """Tests that the reward is in range in the RL resume process"""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.65)
