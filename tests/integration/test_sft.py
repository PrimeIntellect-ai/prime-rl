from pathlib import Path
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

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
    wandb_name = f"{branch_name}-{commit_hash}"
    sft_cmd = ["uv", "run", "sft", "@", "configs/ci/integration/sft/regular.toml"]

    return run_process(
        sft_cmd
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
    assert sft_process.returncode == 0, f"SFT process failed with return code {sft_process.returncode}"


@pytest.fixture
def sft_resume_process(
    sft_process,  # Resume training can only start when regular SFT process is finished
    run_process: Callable[[Command, Environment, int], ProcessResult],
    wandb_project: str,
    output_dir: Path,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
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


def test_no_error_resume(sft_resume_process: ProcessResult):
    assert sft_resume_process.returncode == 0, (
        f"SFT resume process failed with return code {sft_resume_process.returncode}"
    )
