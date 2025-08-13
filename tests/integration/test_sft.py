from pathlib import Path
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

TIMEOUT = 300  # 5 minutes
ENV = {"CUDA_VISIBLE_DEVICES": "1"}
SFT_CMD = ["uv", "run", "sft", "@", "configs/reverse_text/sft.toml", "--max-steps", "10", "--ckpt"]
SFT_RESUME_CMD = [
    "uv",
    "run",
    "sft",
    "@",
    "configs/reverse_text/sft.toml",
    "--max-steps",
    "20",
    "--ckpt.resume-step",
    "10",
]


@pytest.fixture(scope="module")
def wandb_project(username: str) -> str:
    project = "ci-reverse-text-sft"
    if username != "CI_RUNNER":
        project += "-local"
    return project


@pytest.fixture(scope="module")
def sft_process(
    run_process: Callable[[Command, Environment], ProcessResult],
    outputs_dir: Path,
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}"

    return run_process(
        SFT_CMD
        + [
            "--monitor.wandb.project",
            wandb_project,
            "--monitor.wandb.name",
            wandb_name,
            "--outputs-dir",
            outputs_dir.as_posix(),
        ],
        ENV,
        TIMEOUT,
    )


@pytest.fixture
def sft_resume_process(
    sft_process,  # Resume training can only start when regular SFT process is finished
    run_process: Callable[[Command, Environment, int], ProcessResult],
    outputs_dir: Path,
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}-resume"

    return run_process(
        SFT_RESUME_CMD
        + [
            "--monitor.wandb.project",
            wandb_project,
            "--monitor.wandb.name",
            wandb_name,
            "--outputs-dir",
            outputs_dir.as_posix(),
        ],
        ENV,
        TIMEOUT,
    )


def test_no_error(sft_process: ProcessResult):
    assert sft_process.returncode == 0, f"SFT process failed with return code {sft_process.returncode}"


def test_no_error_resume(sft_resume_process: ProcessResult):
    assert sft_resume_process.returncode == 0, (
        f"SFT resume process failed with return code {sft_resume_process.returncode}"
    )
