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
    "--trainer",
    "@",
    "configs/trainer/reverse_text.toml",
    "--orchestrator",
    "@",
    "configs/orchestrator/reverse_text.toml",
    "--orchestrator.monitor.wandb.log_samples",
]


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
    run_name = f"{branch_name}-{commit_hash}"

    return run_process(
        CMD + ["--trainer.monitor.wandb.project", project, "--trainer.monitor.wandb.name", run_name],
        {},
        TIMEOUT,
    )


def test_no_error(train_process: ProcessResult):
    assert train_process.returncode == 0, f"Train process failed with return code {train_process.returncode}"
