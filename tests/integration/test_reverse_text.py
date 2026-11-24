from pathlib import Path
from typing import Callable

import pytest
import torch

from prime_rl.utils.weights import load_state_dict
from tests.conftest import ProcessResult
from tests.integration.dashboard_smoke import make_dashboard_test
from tests.utils import (
    check_avg_mismatch_kl_in_range,
    check_hf_load,
    check_no_error,
    check_reward_goes_up,
    check_reward_in_range,
    convert_checkpoint,
    strip_escape_codes,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

RUN_NAME = "reverse-text"


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


TIMEOUT = 900  # 15 minutes (was 600s — can be tight on contended CI runners
# after verifiers per-call tracing overhead was added)


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for RL CI integration tests."""
    return f"test-reverse-text:{branch_name}"


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
        "configs/ci/integration/reverse-text/start.toml",
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
def rl_resume_process(
    rl_process,  # Resume training can only start when regular RL process is finished
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    if rl_process.returncode != 0:
        pytest.skip("Full weight RL process failed")
    wandb_name = f"{wandb_name}-resume"
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse-text/resume.toml",
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
def test_no_error(rl_process: ProcessResult, run_dir: Path):
    """Tests that the RL process does not fail."""
    check_no_error(rl_process, run_dir)


def test_reward_goes_up(rl_process: ProcessResult, test_no_error, run_dir: Path):
    """Tests that the reward goes up in the RL process"""
    with open(run_dir / "logs" / "latest" / "orchestrator.log", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)


def test_reward_in_range(rl_process: ProcessResult, test_no_error, run_dir: Path):
    """Tests that the reward is in range in the RL process"""
    with open(run_dir / "logs" / "latest" / "orchestrator.log", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.6)


def test_mismatch_kl_in_range(rl_process: ProcessResult, test_no_error, run_dir: Path):
    """Tests that the average mismatch KL is below 0.01 across all steps"""
    with open(run_dir / "logs" / "latest" / "trainer.log", "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_avg_mismatch_kl_in_range(trainer_stdout, last_n_steps=3, max_threshold=0.01)


@pytest.fixture(scope="module")
def test_no_error_resume(rl_resume_process: ProcessResult, run_dir: Path):
    """Tests that the RL resume process does not fail."""
    check_no_error(rl_resume_process, run_dir)


def test_reward_in_range_resume(rl_resume_process: ProcessResult, test_no_error_resume, run_dir: Path):
    """Tests that the reward is in range in the RL resume process"""
    with open(run_dir / "logs" / "latest" / "orchestrator.log", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.6)


def test_convert_final_checkpoint(rl_resume_process: ProcessResult, run_dir: Path, tmp_path: Path):
    """The final DCP checkpoint converts to bf16 and fp8 (direct == chained), and the exports reload in HF."""
    assert rl_resume_process.returncode == 0
    step_dir = run_dir / "checkpoints" / "step_20"
    convert_checkpoint("dcp_to_bf16", step_dir)
    convert_checkpoint("dcp_to_fp8", step_dir)
    check_hf_load(step_dir / "weights")

    # Quantizing the bf16 export reproduces the direct dcp->fp8 export byte-for-byte.
    convert_checkpoint("bf16_to_fp8", step_dir / "weights", tmp_path / "fp8-chained")
    direct = load_state_dict(step_dir / "weights-FP8")
    chained = load_state_dict(tmp_path / "fp8-chained")
    assert set(direct) == set(chained)
    for key in direct:
        assert torch.equal(direct[key].view(torch.uint8), chained[key].view(torch.uint8)), key

    convert_checkpoint("fp8_to_bf16", step_dir / "weights-FP8", tmp_path / "dequant")
    check_hf_load(tmp_path / "dequant")


test_dashboard = make_dashboard_test("rl_process", RUN_NAME)
