import os
import subprocess
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
    "--trainer.ckpt",
    "--orchestrator.ckpt",
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
    "--trainer.max-steps",
    "40",
    "--orchestrator.max-steps",
    "40",
    "--trainer.ckpt.resume-step",
    "20",
    "--orchestrator.ckpt.resume-step",
    "20",
]


@pytest.fixture(scope="module")
def username():
    return os.environ.get("USERNAME_CI", os.getlogin())


@pytest.fixture(scope="module")
def branch_name():
    branch_name_ = os.environ.get("GITHUB_REF_NAME", None)

    if branch_name_ is None:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    else:
        branch_name = branch_name_.replace("/merge", "")
        branch_name = f"pr-{branch_name}"
    return branch_name


@pytest.fixture(scope="module")
def commit_hash():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()


@pytest.fixture(scope="module")
def wandb_project(username: str):
    project = "ci-reverse-text"
    if username != "CI_RUNNER":
        project += "-local"
    return "ci-reverse-text"


@pytest.fixture(scope="module")
def rl_process(
    _vllm_server,  # Can only run with vLLM server
    run_process: Callable[[Command, Environment, int], ProcessResult],
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}"

    return run_process(
        RL_CMD + ["--wandb.project", wandb_project, "--wandb.name", wandb_name],
        {},
        TIMEOUT,
    )


@pytest.fixture
def rl_resume_process(
    _vllm_server,  # Can only run with vLLM server
    _rl_process,  # Resume training can only start when regular RL process is finished
    run_process: Callable[[Command, Environment, int], ProcessResult],
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
):
    wandb_name = f"{branch_name}-{commit_hash}-resume"

    return run_process(
        RL_RESUME_CMD + ["--wandb.project", wandb_project, "--wandb.name", wandb_name],
        {},
        TIMEOUT,
    )


def test_no_error(rl_process: ProcessResult):
    assert rl_process.returncode == 0, f"RL process failed with return code {rl_process.returncode}"


def test_no_error_resume(rl_resume_process: ProcessResult):
    assert rl_resume_process.returncode == 0, (
        f"RL resume process failed with return code {rl_resume_process.returncode}"
    )
