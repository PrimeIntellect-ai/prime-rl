from pathlib import Path
from typing import Callable

import pytest

from tests.integration.conftest import Command, Environment, ProcessResult, check_loss_goes_down, check_zero_return_code

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

TIMEOUT = 300  # 5 minutes
ENV = {"CUDA_VISIBLE_DEVICES": "1"}


@pytest.fixture(scope="module")
def wandb_project(get_wandb_project: Callable[[str], str]) -> str:
    """Get W&B project name for SFT integration tests."""
    return get_wandb_project("reverse-text-sft")


@pytest.fixture(scope="module")
def sft_process(
    run_process: Callable[[Command, Environment, int], ProcessResult],
    wandb_project: str,
    output_dir: Path,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    """Fixture for running SFT CI integration test"""
    wandb_name = f"{branch_name}-{commit_hash}"
    sft_cmd = ["uv", "run", "sft", "@", "configs/ci/integration/sft/regular.toml"]

    return run_process(
        sft_cmd + ["--wandb.project", wandb_project, "--wandb.name", wandb_name, "--output-dir", output_dir.as_posix()],
        ENV,
        TIMEOUT,
    )


@pytest.fixture
def sft_resume_process(
    sft_process,  # Resume training can only start when regular SFT process is finished
    run_process: Callable[[Command, Environment, int], ProcessResult],
    wandb_project: str,
    output_dir: Path,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    """Fixture for resuming SFT CI integration test"""
    wandb_name = f"{branch_name}-{commit_hash}-resume"
    sft_resume_cmd = ["uv", "run", "sft", "@", "configs/ci/integration/sft/resume.toml"]

    return run_process(
        sft_resume_cmd
        + [
            "--wandb.project",
            wandb_project,
            "--wandb.name",
            wandb_name,
            "--output-dir",
            output_dir.as_posix(),
        ],
        ENV,
        TIMEOUT,
    )


def test_no_error(sft_process: ProcessResult):
    """Tests that the SFT process does not fail."""
    check_zero_return_code(sft_process)


def test_loss_goes_down(sft_process: ProcessResult):
    """Tests that the loss goes down in the SFT process"""
    check_loss_goes_down(sft_process)


def test_no_error_resume(sft_resume_process: ProcessResult):
    """Tests that the SFT resume process has a zero return code"""
    check_zero_return_code(sft_resume_process)


def test_loss_goes_down_resume(sft_resume_process: ProcessResult):
    """Tests that the loss goes down in the SFT resume process"""
    check_loss_goes_down(sft_resume_process)
