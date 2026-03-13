from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 900  # 15 minutes

MOE_LORA_CONFIGS = [
    ("glm4_moe", "configs/ci/integration/rl_moe_lora/glm4_moe.toml"),
    ("kimi_k25", "configs/ci/integration/rl_moe_lora/kimi_k25.toml"),
    ("qwen3_moe", "configs/ci/integration/rl_moe_lora/qwen3_moe.toml"),
    ("minimax_m2", "configs/ci/integration/rl_moe_lora/minimax_m2.toml"),
]

MOE_IMPLS = ["hf", "custom"]


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-rl-moe-lora-{branch_name}"


@pytest.fixture(scope="module", params=MOE_LORA_CONFIGS, ids=lambda x: x[0])
def moe_lora_config(request):
    return request.param


@pytest.fixture(scope="module", params=MOE_IMPLS, ids=lambda x: x)
def model_impl(request):
    return request.param


@pytest.fixture(scope="module")
def moe_lora_output_dir(output_dir: Path, moe_lora_config, model_impl) -> Path:
    arch_name, _ = moe_lora_config
    d = output_dir / f"rl_moe_lora_{arch_name}_{model_impl}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def moe_lora_process(
    run_process: Callable[..., ProcessResult],
    moe_lora_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
    moe_lora_config,
    model_impl,
) -> ProcessResult:
    arch_name, config_path = moe_lora_config
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
        moe_lora_output_dir.as_posix(),
    ]
    return run_process(cmd, env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(moe_lora_process: ProcessResult, moe_lora_output_dir: Path):
    check_no_error(moe_lora_process, moe_lora_output_dir)


def test_moe_lora_runs(moe_lora_process: ProcessResult, test_no_error):
    """MoE RL with LoRA completes without error."""
