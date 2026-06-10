from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import (
    check_avg_reward_in_range,
    check_no_error,
    check_reward_goes_up,
    check_reward_in_range,
    strip_escape_codes,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


TIMEOUT = 600  # 10 minutes per leg


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for RL CI integration tests."""
    return f"test-reverse-text-lora-nccl:{branch_name}"


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse_text_lora_nccl/start.toml",
        "--clean-output-dir",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, output_dir: Path):
    """Tests that the RL process does not fail (a hang at end-of-run fails via TIMEOUT)."""
    check_no_error(rl_process, output_dir)


@pytest.fixture(scope="module")
def start_orchestrator_log(rl_process: ProcessResult, output_dir: Path) -> list[str]:
    """Snapshot the start leg's orchestrator log before the resume leg overwrites it."""
    with open(output_dir / "logs" / "orchestrator.log", "r") as f:
        return strip_escape_codes(f.read()).splitlines()


@pytest.fixture(scope="module")
def resume_process(
    start_orchestrator_log: list[str],
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    """Resume from the start leg's end-of-training checkpoint (step 15) and train to step 20.

    Exercises the NCCL LoRA resume bootstrap: the orchestrator arms a receive for the
    restored step and the trainer answers with a one-shot broadcast of the restored adapter.
    """
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse_text_lora_nccl/resume.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"{wandb_name}-resume",
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_resume_no_error(resume_process: ProcessResult, output_dir: Path):
    """Tests that the resume leg does not fail (a bootstrap deadlock fails via TIMEOUT)."""
    check_no_error(resume_process, output_dir)


def test_reward_goes_up(test_no_error, start_orchestrator_log: list[str]):
    """Tests that the reward goes up in the RL process"""
    check_reward_goes_up(start_orchestrator_log)


def test_reward_in_range(test_no_error, start_orchestrator_log: list[str]):
    """Tests that the reward is in range in the RL process"""
    check_reward_in_range(start_orchestrator_log, min_threshold=0.65)


def test_resume_reward_continuous(test_resume_no_error, output_dir: Path):
    """Tests that the resumed steps train and serve the restored adapter: the average
    reward over steps 15-19 stays at the pre-checkpoint level instead of collapsing
    toward the base model (which would indicate the adapter was reset or never served)."""
    with open(output_dir / "logs" / "orchestrator.log", "r") as f:
        resume_stdout = strip_escape_codes(f.read()).splitlines()
    check_avg_reward_in_range(resume_stdout, last_n_steps=5, min_threshold=0.6)
