import os
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 900  # 15 minutes
RUN_SONIC_SMOKE = os.environ.get("PRIME_RL_TEST_SONIC") == "1"


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


# --- MoE with custom impl + sonic backend ---


@pytest.fixture(scope="module")
def moe_custom_sonic_output_dir(output_dir: Path) -> Path:
    d = output_dir / "rl_moe_custom_sonic"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def moe_custom_sonic_process(
    run_process: Callable[..., ProcessResult],
    moe_custom_sonic_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    if not RUN_SONIC_SMOKE:
        pytest.skip("Set PRIME_RL_TEST_SONIC=1 on Hopper lane to run SonicMoE integration smoke.")

    cmd = [
        "uv",
        "run",
        "--extra",
        "sonic-moe",
        "rl",
        "@",
        "configs/ci/integration/rl_moe/start.toml",
        "--trainer.model.impl",
        "custom",
        "--trainer.model.moe_backend",
        "sonic",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"{wandb_name}-custom-sonic",
        "--output-dir",
        moe_custom_sonic_output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.mark.skipif(
    not RUN_SONIC_SMOKE,
    reason="Set PRIME_RL_TEST_SONIC=1 on Hopper lane to run SonicMoE integration smoke.",
)
def test_moe_custom_sonic_runs(moe_custom_sonic_process: ProcessResult, moe_custom_sonic_output_dir: Path):
    check_no_error(moe_custom_sonic_process, moe_custom_sonic_output_dir)
