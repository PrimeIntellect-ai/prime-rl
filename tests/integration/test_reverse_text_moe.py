from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 900  # 15 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-reverse-text-moe:{branch_name}"


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse_text_moe/start.toml",
        "--trainer.model.impl",
        "custom",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"{wandb_name}-custom",
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, output_dir: Path):
    check_no_error(rl_process, output_dir)


def test_moe_runs(rl_process: ProcessResult, test_no_error):
    """MoE RL with custom model impl completes without error."""
