from pathlib import Path
from typing import Callable

import pytest

from prime_rl.trainer.weights import load_state_dict
from tests.conftest import ProcessResult
from tests.utils import check_loss_goes_down, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

TIMEOUT = 300  # 5 minutes


def assert_process_succeeded(process: ProcessResult) -> None:
    assert process.returncode == 0, f"Process has non-zero return code ({process})"


def assert_adapter_checkpoint(adapter_dir: Path) -> None:
    assert (adapter_dir / "adapter_config.json").exists()
    state_dict = load_state_dict(adapter_dir)
    assert state_dict
    assert all(".0.weight" not in key for key in state_dict)
    assert any(key.endswith("lora_A.weight") for key in state_dict)
    assert all(key.startswith("base_model.model.") for key in state_dict)


def assert_no_adapter_checkpoint(adapter_dir: Path) -> None:
    assert not adapter_dir.exists()


def assert_full_checkpoint(weights_dir: Path) -> None:
    state_dict = load_state_dict(weights_dir)
    assert state_dict
    assert all("lora_A" not in key and "lora_B" not in key for key in state_dict)
    assert all(".base_layer." not in key for key in state_dict)
    assert all(not key.startswith("base_model.model.") for key in state_dict)


def get_trainer_log_lines(run_output_dir: Path) -> list[str]:
    trainer_log_path = run_output_dir / "logs" / "trainer.log"
    print(f"Checking trainer path in {trainer_log_path}")
    with open(trainer_log_path, "r") as f:
        return strip_escape_codes(f.read()).splitlines()


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for SFT LoRA CI integration tests."""
    return f"test-sft-lora-{branch_name}"


@pytest.fixture(scope="module")
def separate_adapter_output_dir(output_dir: Path) -> Path:
    return output_dir / "separate_adapter"


@pytest.fixture(scope="module")
def merged_only_output_dir(output_dir: Path) -> Path:
    return output_dir / "merged_only"


@pytest.fixture(scope="module")
def run_sft_lora(
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    wandb_name: str,
):
    def _run_sft_lora(config_path: str, run_output_dir: Path, run_name_suffix: str, clean_output_dir: bool) -> ProcessResult:
        cmd = [
            "uv",
            "run",
            "sft",
            "@",
            config_path,
        ]
        if clean_output_dir:
            cmd.append("--clean-output-dir")
        cmd.extend(
            [
                "--deployment.num-gpus",
                "2",
                "--wandb.project",
                wandb_project,
                "--wandb.name",
                f"{wandb_name}-{run_name_suffix}",
                "--output-dir",
                run_output_dir.as_posix(),
            ]
        )

        return run_process(cmd, timeout=TIMEOUT)

    return _run_sft_lora


@pytest.fixture(scope="module")
def sft_lora_process(
    run_sft_lora,
    separate_adapter_output_dir: Path,
) -> ProcessResult:
    """Fixture for running SFT LoRA CI integration test with separate adapter exports."""
    return run_sft_lora(
        "configs/ci/integration/sft_lora/start.toml",
        separate_adapter_output_dir,
        "separate-adapter",
        clean_output_dir=True,
    )


@pytest.fixture(scope="module")
def sft_lora_resume_process(
    sft_lora_process,  # Resume training can only start when regular SFT LoRA process is finished
    run_sft_lora,
    separate_adapter_output_dir: Path,
) -> ProcessResult:
    """Fixture for resuming the SFT LoRA CI integration test with separate adapter exports."""
    if sft_lora_process.returncode != 0:
        pytest.skip("Initial SFT LoRA process failed")
    return run_sft_lora(
        "configs/ci/integration/sft_lora/resume.toml",
        separate_adapter_output_dir,
        "separate-adapter-resume",
        clean_output_dir=False,
    )


@pytest.fixture(scope="module")
def sft_lora_merged_only_process(
    run_sft_lora,
    merged_only_output_dir: Path,
) -> ProcessResult:
    """Fixture for running SFT LoRA CI integration test without separate adapter exports."""
    return run_sft_lora(
        "configs/ci/integration/sft_lora/start_merged_only.toml",
        merged_only_output_dir,
        "merged-only",
        clean_output_dir=True,
    )


def test_no_error(sft_lora_process: ProcessResult):
    """Tests that the SFT LoRA process does not fail."""
    assert_process_succeeded(sft_lora_process)


def test_loss_goes_down(sft_lora_process: ProcessResult, separate_adapter_output_dir: Path):
    """Tests that the loss goes down in the SFT LoRA process."""
    assert_process_succeeded(sft_lora_process)
    check_loss_goes_down(get_trainer_log_lines(separate_adapter_output_dir))


def test_adapter_checkpoint_written(sft_lora_process: ProcessResult, separate_adapter_output_dir: Path):
    """Tests that the adapter checkpoint is written with valid PEFT-compatible keys."""
    assert_process_succeeded(sft_lora_process)
    adapter_dir = separate_adapter_output_dir / "weights" / "step_10" / "lora_adapters"
    assert_adapter_checkpoint(adapter_dir)


def test_full_checkpoint_written(sft_lora_process: ProcessResult, separate_adapter_output_dir: Path):
    """Tests that the full checkpoint stays HF-compatible when adapters are also exported."""
    assert_process_succeeded(sft_lora_process)
    weights_dir = separate_adapter_output_dir / "weights" / "step_10"
    assert_full_checkpoint(weights_dir)


def test_no_error_merged_only(sft_lora_merged_only_process: ProcessResult):
    """Tests that the merged-only SFT LoRA process does not fail."""
    assert_process_succeeded(sft_lora_merged_only_process)


def test_loss_goes_down_merged_only(sft_lora_merged_only_process: ProcessResult, merged_only_output_dir: Path):
    """Tests that the loss goes down when only merged full checkpoints are exported."""
    assert_process_succeeded(sft_lora_merged_only_process)
    check_loss_goes_down(get_trainer_log_lines(merged_only_output_dir))


def test_full_checkpoint_written_merged_only(
    sft_lora_merged_only_process: ProcessResult,
    merged_only_output_dir: Path,
):
    """Tests that merged-only LoRA exports remain HF-compatible."""
    assert_process_succeeded(sft_lora_merged_only_process)
    weights_dir = merged_only_output_dir / "weights" / "step_10"
    assert_full_checkpoint(weights_dir)


def test_adapter_checkpoint_not_written_merged_only(
    sft_lora_merged_only_process: ProcessResult,
    merged_only_output_dir: Path,
):
    """Tests that merged-only LoRA exports do not create an adapter sidecar directory."""
    assert_process_succeeded(sft_lora_merged_only_process)
    adapter_dir = merged_only_output_dir / "weights" / "step_10" / "lora_adapters"
    assert_no_adapter_checkpoint(adapter_dir)


def test_no_error_resume(sft_lora_resume_process: ProcessResult):
    """Tests that the SFT LoRA resume process does not fail."""
    assert_process_succeeded(sft_lora_resume_process)


def test_loss_goes_down_resume(sft_lora_resume_process: ProcessResult, separate_adapter_output_dir: Path):
    """Tests that the loss goes down in the SFT LoRA resume process."""
    assert_process_succeeded(sft_lora_resume_process)
    check_loss_goes_down(get_trainer_log_lines(separate_adapter_output_dir))


def test_adapter_checkpoint_written_resume(sft_lora_resume_process: ProcessResult, separate_adapter_output_dir: Path):
    """Tests that the adapter checkpoint is written after resuming with valid PEFT-compatible keys."""
    assert_process_succeeded(sft_lora_resume_process)
    adapter_dir = separate_adapter_output_dir / "weights" / "step_20" / "lora_adapters"
    assert_adapter_checkpoint(adapter_dir)


def test_full_checkpoint_written_resume(sft_lora_resume_process: ProcessResult, separate_adapter_output_dir: Path):
    """Tests that the resumed full checkpoint stays HF-compatible when adapters are also exported."""
    assert_process_succeeded(sft_lora_resume_process)
    weights_dir = separate_adapter_output_dir / "weights" / "step_20"
    assert_full_checkpoint(weights_dir)
