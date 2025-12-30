from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_number_goes_up_or_down, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

TIMEOUT = 300  # 5 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for SFT LoRA CI integration tests."""
    return f"test-sft-lora-{branch_name}"


@pytest.fixture(scope="module")
def sft_lora_process(
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    """Fixture for running SFT LoRA CI integration test"""
    cmd = [
        "uv",
        "run",
        "torchrun",
        "--local-ranks-filter",
        "0",
        "--nproc-per-node",
        "2",
        "src/prime_rl/trainer/sft/train.py",
        "@",
        "configs/ci/integration/sft_lora/start.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]

    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def sft_lora_resume_process(
    sft_lora_process: ProcessResult,  # Resume training can only start when regular SFT LoRA process is finished
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    """Fixture for resuming SFT LoRA CI integration test"""
    if sft_lora_process.returncode != 0:
        pytest.skip("SFT LoRA process failed")
    wandb_name += "-resume"
    cmd = [
        "uv",
        "run",
        "torchrun",
        "--local-ranks-filter",
        "0",
        "--nproc-per-node",
        "2",
        "src/prime_rl/trainer/sft/train.py",
        "@",
        "configs/ci/integration/sft_lora/resume.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]

    return run_process(cmd, timeout=TIMEOUT)


check_loss_goes_down = partial(check_number_goes_up_or_down, go_up=False, pattern=r"Loss:\s*(\d+\.\d{4})")


def test_no_error(sft_lora_process: ProcessResult) -> None:
    """Tests that the SFT LoRA process does not fail."""
    assert sft_lora_process.returncode == 0, f"Process has non-zero return code ({sft_lora_process})"


def test_loss_goes_down(sft_lora_process: ProcessResult, output_dir: Path) -> None:
    """Tests that the loss goes down in the SFT LoRA process"""
    trainer_log_path = output_dir / "logs" / "trainer" / "rank_0.log"
    print(f"Checking trainer path in {trainer_log_path}")
    with open(trainer_log_path, "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)


def test_no_error_resume(sft_lora_resume_process: ProcessResult) -> None:
    """Tests that the SFT LoRA resume process does not fail."""
    assert sft_lora_resume_process.returncode == 0, f"Process has non-zero return code ({sft_lora_resume_process})"
