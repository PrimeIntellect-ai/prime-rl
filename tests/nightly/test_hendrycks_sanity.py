from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import (
    check_mismatch_kl_in_range,
    check_no_error,
    check_reward_goes_up,
    check_reward_in_range,
    strip_escape_codes,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for RL CI integration tests."""
    return f"hendrycks-sanity-{branch_name}"


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
        "examples/hendrycks_sanity/rl.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
        "--max-steps",
        "1000",  # do less steps to finish in time
    ]
    return run_process(cmd)


MIN_TRAIN_REWARD = 0.75
MISMATCH_KL_MIN = 0.0
MISMATCH_KL_MAX = 0.0005


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, output_dir: Path):
    """Tests that the RL process does not fail."""
    check_no_error(rl_process, output_dir)


def test_reward_goes_up(rl_process: ProcessResult, test_no_error, output_dir: Path):
    """Tests that the train reward goes up during training"""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)


def test_reward_reaches_threshold(rl_process: ProcessResult, test_no_error, output_dir: Path):
    """Tests that the train reward reaches a minimum threshold"""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=MIN_TRAIN_REWARD)


def test_mismatch_kl_in_band(rl_process: ProcessResult, test_no_error, output_dir: Path):
    """Tests that mismatch KL stays within the expected band."""
    with open(output_dir / "logs" / "trainer.stdout", "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_mismatch_kl_in_range(trainer_stdout, min_threshold=MISMATCH_KL_MIN, max_threshold=MISMATCH_KL_MAX)
