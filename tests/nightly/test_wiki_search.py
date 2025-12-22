from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error, check_number_goes_up_or_down, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for RL CI integration tests."""
    return f"wiki-search-{branch_name}"


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
        "examples/wiki_search/rl.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd)


check_reward_goes_up = partial(check_number_goes_up_or_down, go_up=True, pattern=r"Reward:\s*(\d+\.\d{4})")


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, output_dir: Path):
    """Tests that the RL process does not fail."""
    check_no_error(rl_process, output_dir)


def test_reward_goes_up(rl_process: ProcessResult, test_no_error, output_dir: Path):
    """Tests that the reward goes up in the RL process"""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)
