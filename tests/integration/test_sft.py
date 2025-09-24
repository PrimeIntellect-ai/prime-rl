from pathlib import Path
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

TIMEOUT = 300  # 5 minutes
ENV = {"CUDA_VISIBLE_DEVICES": "1"}
SFT_CMD = ["uv", "run", "sft", "@", "configs/reverse_text/sft/train.toml", "--max-steps", "10", "--ckpt"]
SFT_RESUME_CMD = [
    "uv",
    "run",
    "sft",
    "@",
    "configs/reverse_text/sft/train.toml",
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
    run_process: Callable[[Command, Environment, int], ProcessResult],
    output_dir: Path,
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}"

    return run_process(
        SFT_CMD
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


@pytest.fixture
def sft_resume_process(
    sft_process,  # Resume training can only start when regular SFT process is finished
    run_process: Callable[[Command, Environment, int], ProcessResult],
    output_dir: Path,
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}-resume"

    return run_process(
        SFT_RESUME_CMD
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


SFT_CMD_MOE = ["uv", "run", "sft", "@", "configs/debug/moe/sft/train.toml"]


@pytest.fixture
def sft_moe_process(
    run_process: Callable[[Command, Environment, int], ProcessResult],
    output_dir: Path,
) -> ProcessResult:
    return run_process(
        SFT_CMD_MOE
        + [
            "--output-dir",
            output_dir.as_posix(),
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


def test_no_error_moe(sft_moe_process: ProcessResult):
    assert sft_moe_process.returncode == 0, f"SFT MOE process failed with return code {sft_moe_process.returncode}"
