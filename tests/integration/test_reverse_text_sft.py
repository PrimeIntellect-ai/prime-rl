import hashlib
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_loss_goes_down, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

RUN_NAME = "reverse-text-sft"


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


TIMEOUT = 300  # 5 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for SFT CI integration tests."""
    return f"test-reverse-text-sft:{branch_name}"


@pytest.fixture(scope="module")
def sft_process(
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    """Fixture for running SFT CI integration test"""
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        "configs/ci/integration/reverse-text-sft/start.toml",
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
def sft_resume_process(
    sft_process,  # Resume training can only start when regular SFT process is finished
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    """Fixture for resuming SFT CI integration test"""
    wandb_name += "-resume"
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        "configs/ci/integration/reverse-text-sft/resume.toml",
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


@pytest.fixture(scope="module")
def sft_full_offload_model_only_resume_process(
    sft_resume_process: ProcessResult,
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    """Resume without optimizer state using full CPU offload."""
    if sft_resume_process.returncode != 0:
        pytest.skip("Regular SFT resume failed")
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        "configs/ci/integration/reverse-text-sft/full-offload-resume.toml",
        "--deployment.num-train-gpus",
        "2",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        f"{wandb_name}-full-offload-model-only-resume",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]

    return run_process(cmd, timeout=TIMEOUT)


def test_no_error(sft_process: ProcessResult):
    """Tests that the SFT process does not fail."""
    assert sft_process.returncode == 0, f"Process has non-zero return code ({sft_process})"


def test_loss_goes_down(sft_process: ProcessResult, run_dir: Path):
    """Tests that the loss goes down in the SFT process"""
    trainer_log_path = run_dir / "logs" / "latest" / "trainer.log"
    print(f"Checking trainer path in {trainer_log_path}")
    with open(trainer_log_path, "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)


def test_no_error_resume(sft_resume_process: ProcessResult):
    """Tests that the SFT resume process does not fail."""
    assert sft_resume_process.returncode == 0, f"Process has non-zero return code ({sft_resume_process})"


def test_loss_goes_down_resume(sft_resume_process: ProcessResult, run_dir: Path):
    """Tests that the loss goes down in the SFT resume process"""
    trainer_log_path = run_dir / "logs" / "latest" / "trainer.log"
    print(f"Checking trainer path in {trainer_log_path}")
    with open(trainer_log_path, "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)


def test_full_offload_model_only_resume_preserves_weights(
    sft_full_offload_model_only_resume_process: ProcessResult,
    run_dir: Path,
):
    assert sft_full_offload_model_only_resume_process.returncode == 0, (
        f"Process has non-zero return code ({sft_full_offload_model_only_resume_process})"
    )
    before_dir = run_dir / "weights" / "step_5"
    after_dir = run_dir / "weights" / "step_6"
    before_files = sorted(before_dir.glob("*.safetensors"))
    after_files = sorted(after_dir.glob("*.safetensors"))
    assert before_files
    assert [path.name for path in before_files] == [path.name for path in after_files]
    for before, after in zip(before_files, after_files):
        with before.open("rb") as before_handle, after.open("rb") as after_handle:
            assert (
                hashlib.file_digest(before_handle, "sha256").digest()
                == hashlib.file_digest(after_handle, "sha256").digest()
            )
