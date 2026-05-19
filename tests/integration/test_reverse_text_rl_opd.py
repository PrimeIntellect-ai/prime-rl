from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_loss_goes_down, check_no_error, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 600  # 10 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-reverse-text-rl-opd:{branch_name}"


@pytest.fixture(scope="module")
def rl_opd_process(
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    """Run the RL entrypoint with training_mode = "opd" and overlapped student+teacher inference."""
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse_text_rl_opd/start.toml",
        "--clean-output-dir",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(rl_opd_process: ProcessResult, output_dir: Path):
    check_no_error(rl_opd_process, output_dir)


def test_loss_goes_down(rl_opd_process: ProcessResult, test_no_error, output_dir: Path):
    with open(output_dir / "logs" / "trainer.log", "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_loss_goes_down(trainer_stdout)
