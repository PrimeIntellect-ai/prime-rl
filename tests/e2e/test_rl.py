import os
import subprocess
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TRAINING_ENV = {"CUDA_VISIBLE_DEVICES": "0"}
TRAINING_CMD = [
    "uv",
    "run",
    "torchrun",
    "src/zeroband/train.py",
    "@configs/training/simple_reverse_two_gpu.toml",
]

INFERENCE_ENV = {"CUDA_VISIBLE_DEVICES": "1"}
INFERENCE_CMD = [
    "uv",
    "run",
    "python",
    "src/zeroband/infer.py",
    "@configs/inference/simple_reverse_two_gpus.toml",
]


@pytest.fixture(scope="module")
def processes(run_processes: Callable[[list[Command], list[Environment]], list[ProcessResult]]) -> list[ProcessResult]:
    username = os.environ.get("USERNAME_CI", os.getlogin())

    branch_name_ = os.environ.get("GITHUB_REF_NAME", None)
    if branch_name_ is None:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    else:
        branch_name = branch_name_.replace("/merge", "")
        branch_name = f"pr-{branch_name}"

    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

    if username == "CI_RUNNER":
        project = "ci_run_prime_rl"
        wandb_run_name = f"{branch_name}-{commit_hash}"
    else:
        project = "ci_run_prime_rl_local"
        wandb_run_name = f"{username}-{branch_name}-{commit_hash}"

    training_cmd = TRAINING_CMD + ["--monitor.wandb.name", wandb_run_name, "--monitor.wandb.project", project]
    inference_cmd = INFERENCE_CMD

    return run_processes([training_cmd, inference_cmd], [TRAINING_ENV, INFERENCE_ENV], timeout=600)


def test_no_error(processes: list[ProcessResult]):
    training_process, inference_process = processes

    assert training_process.returncode == 0, f"Training process failed with return code {training_process.returncode}"
    assert inference_process.returncode == 0, f"Inference process failed with return code {inference_process.returncode}"
