from pathlib import Path
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


TIMEOUT = 600  # 10 minutes
RL_CMD = [
    "uv",
    "run",
    "rl",
    "--trainer",
    "@",
    "configs/reverse_text/train.toml",
    "--orchestrator",
    "@",
    "configs/reverse_text/orch.toml",
    "--orchestrator.sampling.max-tokens",
    "128",
    "--ckpt",
]
RL_RESUME_CMD = [
    "uv",
    "run",
    "rl",
    "--trainer",
    "@",
    "configs/reverse_text/train.toml",
    "--orchestrator",
    "@",
    "configs/reverse_text/orch.toml",
    "--orchestrator.sampling.max-tokens",
    "128",
    "--max-steps",
    "40",
    "--ckpt.resume-step",
    "20",
]


@pytest.fixture(scope="module")
def wandb_project(username: str) -> str:
    project = "ci-reverse-text-rl"
    if username != "CI_RUNNER":
        project += "-local"
    return project


@pytest.fixture(scope="module")
def rl_process(
    vllm_server,  # Can only run with vLLM server
    run_process: Callable[[Command, Environment, int], ProcessResult],
    outputs_dir: Path,
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}"

    return run_process(
        RL_CMD
        + ["--wandb.project", wandb_project, "--wandb.name", wandb_name, "--outputs-dir", outputs_dir.as_posix()],
        {},
        TIMEOUT,
    )


@pytest.fixture
def rl_resume_process(
    vllm_server,  # Can only run with vLLM server
    rl_process,  # Resume training can only start when regular RL process is finished
    run_process: Callable[[Command, Environment, int], ProcessResult],
    outputs_dir: Path,
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}-resume"

    return run_process(
        RL_RESUME_CMD
        + ["--wandb.project", wandb_project, "--wandb.name", wandb_name, "--outputs-dir", outputs_dir.as_posix()],
        {},
        TIMEOUT,
    )


def test_no_error(rl_process: ProcessResult):
    assert rl_process.returncode == 0, f"RL process failed with return code {rl_process.returncode}"


def test_no_error_resume(rl_resume_process: ProcessResult):
    assert rl_resume_process.returncode == 0, (
        f"RL resume process failed with return code {rl_resume_process.returncode}"
    )
