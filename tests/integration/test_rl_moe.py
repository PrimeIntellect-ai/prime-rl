from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 900  # 15 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-rl-moe-{branch_name}"


# --- MoE with HF impl (default) ---


@pytest.fixture(scope="module")
def moe_hf_output_dir(output_dir: Path) -> Path:
    d = output_dir / "rl_moe_hf"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def moe_hf_process(
    run_process: Callable[..., ProcessResult],
    moe_hf_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl_moe/start.toml",
        "--trainer.model.impl",
        "hf",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"{wandb_name}-hf",
        "--output-dir",
        moe_hf_output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error_hf(moe_hf_process: ProcessResult, moe_hf_output_dir: Path):
    check_no_error(moe_hf_process, moe_hf_output_dir)


def test_moe_hf_runs(moe_hf_process: ProcessResult, test_no_error_hf):
    """MoE RL with HF model impl completes without error."""


# --- MoE with custom impl ---


@pytest.fixture(scope="module")
def moe_custom_output_dir(output_dir: Path) -> Path:
    d = output_dir / "rl_moe_custom"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def moe_custom_process(
    run_process: Callable[..., ProcessResult],
    moe_custom_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl_moe/start.toml",
        "--trainer.model.impl",
        "custom",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"{wandb_name}-custom",
        "--output-dir",
        moe_custom_output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error_custom(moe_custom_process: ProcessResult, moe_custom_output_dir: Path):
    check_no_error(moe_custom_process, moe_custom_output_dir)


def test_moe_custom_runs(moe_custom_process: ProcessResult, test_no_error_custom):
    """MoE RL with custom model impl completes without error."""
