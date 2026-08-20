from pathlib import Path
from typing import Callable

import pytest

from prime_rl.utils.weights import load_state_dict
from tests.conftest import ProcessResult
from tests.utils import check_loss_goes_down, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

RUN_NAME = "reverse-text-sft-lora"


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


TIMEOUT = 300  # 5 minutes


def convert_adapter(run_process: Callable[..., ProcessResult], run_dir: Path, step: int) -> Path:
    """Convert a LoRA DCP checkpoint into a PEFT adapter via scripts/dcp_to_hf.py."""
    adapter_dir = run_dir / "weights_hf" / f"step_{step}"
    convert = run_process(
        [
            "uv",
            "run",
            "torchrun",
            "--nproc-per-node",
            "2",
            "scripts/dcp_to_hf.py",
            "--model.name",
            "PrimeIntellect/Qwen3-0.6B",
            "--model.lora.rank",
            "8",
            "--ckpt-dir",
            (run_dir / "checkpoints" / f"step_{step}" / "trainer").as_posix(),
            "--output-dir",
            adapter_dir.as_posix(),
        ],
        timeout=TIMEOUT,
    )
    assert convert.returncode == 0, f"dcp_to_hf failed for step {step} ({convert})"
    return adapter_dir


def assert_adapter_checkpoint(adapter_dir: Path) -> None:
    assert (adapter_dir / "adapter_config.json").exists()
    state_dict = load_state_dict(adapter_dir)
    assert state_dict
    assert all(".0.weight" not in key for key in state_dict)
    assert any(key.endswith("lora_A.weight") for key in state_dict)
    assert all(key.startswith("base_model.model.") for key in state_dict)


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


def test_adapter_checkpoint_written(
    sft_lora_process: ProcessResult, run_process: Callable[..., ProcessResult], run_dir: Path
):
    """Tests that the DCP checkpoint converts to a valid PEFT-compatible adapter."""
    adapter_dir = convert_adapter(run_process, run_dir, step=5)
    assert_adapter_checkpoint(adapter_dir)


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


def test_adapter_checkpoint_written_resume(
    sft_lora_resume_process: ProcessResult, run_process: Callable[..., ProcessResult], run_dir: Path
):
    """Tests that the resumed run's DCP checkpoint converts to a valid PEFT-compatible adapter."""
    adapter_dir = convert_adapter(run_process, run_dir, step=10)
    assert_adapter_checkpoint(adapter_dir)
