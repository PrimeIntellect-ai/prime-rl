import os
import subprocess
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


TIMEOUT = 600  # 10 minutes
CMD = [
    "uv",
    "run",
    "rl",
    "--trainer.model.name",
    "willcb/Qwen2.5-0.5B-Reverse-SFT",
    "--trainer.optim.lr",
    "3e-6",
    "--orchestrator.data.name",
    "mikasenghaas/reverse_text_dataset_debug_50_seq_len",
    "--orchestrator.sampling.n",
    "16",
    "--orchestrator.sampling.max_seq_len",
    "128",
    "--orchestrator.batch-size",
    "128",
    "--orchestrator.micro-batch-size",
    "16",
    "--orchestrator.seq-len",
    "128",
    "--trainer.max-steps",
    "30",
    "--orchestrator.monitor.wandb.log_samples",
]
ENV = {"CUDA_VISIBLE_DEVICES": "1"}


@pytest.fixture(scope="module")
def train_process(vllm_server: str, run_process: Callable[[Command, Environment, int], ProcessResult]) -> ProcessResult:
    # Parse git information
    username = os.environ.get("USERNAME_CI", os.getlogin())
    branch_name_ = os.environ.get("GITHUB_REF_NAME", None)

    if branch_name_ is None:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    else:
        branch_name = branch_name_.replace("/merge", "")
        branch_name = f"pr-{branch_name}"

    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

    # Setup W&B project and run name
    project = "ci-reverse-text"
    if username != "CI_RUNNER":
        project += "-local"
    group_name = f"{branch_name}-{commit_hash}"

    return run_process(
        CMD + ["--trainer.monitor.wandb.project", project, "--trainer.monitor.wandb.group", group_name],
        ENV,
        TIMEOUT,
    )


def test_no_error(train_process: ProcessResult):
    assert train_process.returncode == 0, f"Train process failed with return code {train_process.returncode}"
