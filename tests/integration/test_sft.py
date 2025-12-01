from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from tests.integration.conftest import ProcessResult, check_number_goes_up_or_down, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

TIMEOUT = 300  # 5 minutes


@pytest.fixture(scope="module")
def wandb_project(get_wandb_project: Callable[[str], str]) -> str:
    """Get W&B project name for SFT integration tests."""
    return get_wandb_project("reverse-text-sft")


@pytest.fixture(scope="module")
def sft_process(
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    output_dir: Path,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    """Fixture for running SFT CI integration test"""
    wandb_name = f"{branch_name}-{commit_hash}"
    cmd = [
        "uv",
        "run",
        "torchrun",
        "--local-ranks-filter",
        "0",
        "--nproc-per-node",
        "2",
        "src/prime_rl/trainer/sft/train.py",
        "@",
        "configs/ci/integration/sft/start.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]

    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def sft_resume_process(
    sft_process,  # Resume training can only start when regular SFT process is finished
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    output_dir: Path,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    """Fixture for resuming SFT CI integration test"""
    wandb_name = f"{branch_name}-{commit_hash}-resume"
    cmd = [
        "uv",
        "run",
        "torchrun",
        "--local-ranks-filter",
        "0",
        "--nproc-per-node",
        "2",
        "src/prime_rl/trainer/sft/train.py",
        "@",
        "configs/ci/integration/sft/resume.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]

    return run_process(cmd, timeout=TIMEOUT)


check_loss_goes_down = partial(check_number_goes_up_or_down, go_up=False, pattern=r"Loss:\s*(\d+\.\d{4})")


def test_no_error(sft_process: ProcessResult):
    """Tests that the SFT process does not fail."""
    assert sft_process.returncode == 0, f"Process has non-zero return code ({sft_process})"


def test_loss_goes_down(sft_process: ProcessResult, output_dir: Path):
    """Tests that the loss goes down in the SFT process"""
    trainer_log_path = output_dir / "logs" / "trainer" / "rank_0.log"
    print(f"Checking trainer path in {trainer_log_path}")
    with open(trainer_log_path, "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)


def test_no_error_resume(sft_resume_process: ProcessResult):
    """Tests that the SFT resume process does not fail."""
    assert sft_resume_process.returncode == 0, f"Process has non-zero return code ({sft_resume_process})"


def test_loss_goes_down_resume(sft_resume_process: ProcessResult, output_dir: Path):
    """Tests that the loss goes down in the SFT resume process"""
    trainer_log_path = output_dir / "logs" / "trainer" / "rank_0.log"
    print(f"Checking trainer path in {trainer_log_path}")
    with open(trainer_log_path, "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)
