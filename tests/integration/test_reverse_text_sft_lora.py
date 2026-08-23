from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.integration.dashboard_smoke import check_dashboard_smoke
from tests.utils import check_loss_goes_down, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

RUN_NAME = "reverse-text-sft-lora"


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


TIMEOUT = 300  # 5 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for SFT LoRA CI integration tests."""
    return f"test-reverse-text-sft-lora:{branch_name}"


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
        "sft",
        "@",
        "configs/ci/integration/reverse-text-sft-lora/start.toml",
        "--deployment.num-train-gpus",
        "2",
        "--clean",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]

    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def sft_lora_resume_process(
    sft_lora_process,  # Resume training can only start when regular SFT LoRA process is finished
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    """Fixture for resuming SFT LoRA CI integration test"""
    wandb_name += "-resume"
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        "configs/ci/integration/reverse-text-sft-lora/resume.toml",
        "--deployment.num-train-gpus",
        "2",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]

    return run_process(cmd, timeout=TIMEOUT)


def test_no_error(sft_lora_process: ProcessResult):
    """Tests that the SFT LoRA process does not fail."""
    assert sft_lora_process.returncode == 0, f"Process has non-zero return code ({sft_lora_process})"


def test_loss_goes_down(sft_lora_process: ProcessResult, run_dir: Path):
    """Tests that the loss goes down in the SFT LoRA process"""
    trainer_log_path = run_dir / "logs" / "latest" / "trainer.log"
    print(f"Checking trainer path in {trainer_log_path}")
    with open(trainer_log_path, "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)


def test_no_error_resume(sft_lora_resume_process: ProcessResult):
    """Tests that the SFT LoRA resume process does not fail."""
    assert sft_lora_resume_process.returncode == 0, f"Process has non-zero return code ({sft_lora_resume_process})"


def test_loss_goes_down_resume(sft_lora_resume_process: ProcessResult, run_dir: Path):
    """Tests that the loss goes down in the SFT LoRA resume process"""
    trainer_log_path = run_dir / "logs" / "latest" / "trainer.log"
    print(f"Checking trainer path in {trainer_log_path}")
    with open(trainer_log_path, "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)


def test_dashboard(sft_lora_process: ProcessResult, test_no_error, output_dir: Path):
    check_dashboard_smoke(output_dir, RUN_NAME)
