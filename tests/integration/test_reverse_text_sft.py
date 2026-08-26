from pathlib import Path
from typing import Callable

import pytest
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from tests.conftest import ProcessResult
from tests.integration.dashboard_smoke import make_dashboard_test
from tests.utils import check_hf_load, check_loss_goes_down, convert_checkpoint, strip_escape_codes

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
    tmp_path: Path,
):
    assert sft_full_offload_model_only_resume_process.returncode == 0, (
        f"Process has non-zero return code ({sft_full_offload_model_only_resume_process})"
    )

    def model_state(step: int) -> dict[str, torch.Tensor]:
        torch_save_path = tmp_path / f"step_{step}.pt"
        dcp_to_torch_save(run_dir / "checkpoints" / f"step_{step}" / "trainer", torch_save_path)
        return torch.load(torch_save_path, map_location="cpu", weights_only=False)["app"]["model"]

    before, after = model_state(5), model_state(6)
    assert before.keys() == after.keys()
    # The runs train in different compute dtypes (fp32 vs bf16 under full offload),
    # so compare in the dtype the resumed model actually loaded.
    for key in before:
        assert torch.equal(before[key].to(torch.bfloat16), after[key].to(torch.bfloat16)), (
            f"Weight mismatch after model-only resume: {key}"
        )


def test_convert_final_checkpoint(sft_resume_process: ProcessResult, run_dir: Path, tmp_path: Path):
    """The final DCP checkpoint converts to bf16 and fp8, and the exports reload in HF."""
    assert sft_resume_process.returncode == 0
    step_dir = run_dir / "checkpoints" / "step_10"
    convert_checkpoint("dcp_to_bf16", step_dir)
    convert_checkpoint("dcp_to_fp8", step_dir)
    check_hf_load(step_dir / "weights")
    convert_checkpoint("fp8_to_bf16", step_dir / "weights-FP8", tmp_path / "dequant")
    check_hf_load(tmp_path / "dequant")


test_dashboard = make_dashboard_test("sft_process", RUN_NAME)
