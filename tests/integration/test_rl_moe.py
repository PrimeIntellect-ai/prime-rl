from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 900  # 15 minutes

MOE_CONFIGS = [
    ("glm4_moe", "configs/ci/integration/rl_moe/glm4_moe.toml"),
    ("qwen3_moe", "configs/ci/integration/rl_moe/qwen3_moe.toml"),
    ("minimax_m2", "configs/ci/integration/rl_moe/minimax_m2.toml"),
]

MOE_IMPLS = ["hf", "custom"]


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-rl-moe-{branch_name}"


@pytest.fixture(scope="module", params=MOE_CONFIGS, ids=lambda x: x[0])
def moe_config(request):
    return request.param


@pytest.fixture(scope="module", params=MOE_IMPLS, ids=lambda x: x)
def model_impl(request):
    return request.param


@pytest.fixture(scope="module")
def moe_output_dir(output_dir: Path, moe_config, model_impl) -> Path:
    arch_name, _ = moe_config
    d = output_dir / f"rl_moe_{arch_name}_{model_impl}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def moe_process(
    run_process: Callable[..., ProcessResult],
    moe_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
    moe_config,
    model_impl,
) -> ProcessResult:
    arch_name, config_path = moe_config
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        config_path,
        "--trainer.model.impl",
        model_impl,
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"{wandb_name}-{arch_name}-{model_impl}",
        "--output-dir",
        moe_output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(moe_process: ProcessResult, moe_output_dir: Path):
    check_no_error(moe_process, moe_output_dir)


def test_moe_runs(moe_process: ProcessResult, test_no_error):
    """MoE RL completes without error."""
