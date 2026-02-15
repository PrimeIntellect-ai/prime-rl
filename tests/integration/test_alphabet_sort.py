from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error, check_number_goes_up_or_down, check_reward_in_range, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


TIMEOUT = 600  # 10 minutes


@pytest.fixture(scope="module")
def rl_output_dir(output_dir: Path) -> Path:
    rl_dir = output_dir / "alphabet_sort_start"
    rl_dir.mkdir(parents=True, exist_ok=True)
    return rl_dir


@pytest.fixture(scope="module")
def rl_resume_output_dir(output_dir: Path) -> Path:
    rl_resume_dir = output_dir / "alphabet_sort_resume"
    rl_resume_dir.mkdir(parents=True, exist_ok=True)
    return rl_resume_dir


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for alphabet sort RL CI integration tests."""
    return f"test-alphabet-sort-{branch_name}"


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    rl_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/alphabet_sort/start.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        rl_output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def rl_resume_process(
    rl_process,  # Resume training can only start when regular RL process is finished
    run_process: Callable[..., ProcessResult],
    rl_resume_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    if rl_process.returncode != 0:
        pytest.skip("Alphabet sort RL process failed")
    wandb_name = f"{wandb_name}-resume"
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/alphabet_sort/resume.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        rl_resume_output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


check_reward_goes_up = partial(check_number_goes_up_or_down, go_up=True, pattern=r"Reward:\s*(\d+\.\d{4})")


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, rl_output_dir: Path):
    """Tests that the RL process does not fail."""
    check_no_error(rl_process, rl_output_dir)


def test_reward_goes_up(rl_process: ProcessResult, test_no_error, rl_output_dir: Path):
    """Tests that the reward goes up in the RL process."""
    with open(rl_output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)


def test_reward_in_range(rl_process: ProcessResult, test_no_error, rl_output_dir: Path):
    """Tests that the reward is in range in the RL process."""
    with open(rl_output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.2)


@pytest.fixture(scope="module")
def test_no_error_resume(rl_resume_process: ProcessResult, rl_resume_output_dir: Path):
    """Tests that the RL resume process does not fail."""
    check_no_error(rl_resume_process, rl_resume_output_dir)


def test_reward_in_range_resume(rl_resume_process: ProcessResult, test_no_error_resume, rl_resume_output_dir: Path):
    """Tests that the reward is in range in the RL resume process."""
    with open(rl_resume_output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.2)
