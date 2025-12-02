from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_number_goes_up_or_down, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def wandb_name(branch_name: str, commit_hash: str) -> str:
    """Fixture for W&B name for RL CI integration tests."""
    return f"reverse-text-{branch_name}-{commit_hash}"


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
        "configs/ci/nightly/reverse_text.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd)


check_reward_goes_up = partial(check_number_goes_up_or_down, go_up=True, pattern=r"Reward:\s*(\d+\.\d{4})")


def test_no_error(rl_process: ProcessResult, output_dir: Path):
    """Tests that the RL process does not fail."""
    if rl_process.returncode != 0:
        print("=== Inference Outputs ===")
        with open(output_dir / "logs" / "inference.stdout", "r") as f:
            print(*f.readlines()[-100:], sep="\n")
        print("=== Orchestrator Outputs ===")
        with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
            print(*f.readlines()[-100:], sep="\n")
    assert rl_process.returncode == 0, f"Process has non-zero return code ({rl_process})"


def test_reward_goes_up(rl_process: ProcessResult, output_dir: Path):
    """Tests that the reward goes up in the RL process"""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)
