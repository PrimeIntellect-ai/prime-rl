from pathlib import Path
from typing import Callable

import pytest
import torch

from tests.conftest import ProcessResult
from tests.utils import check_no_error

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 900  # 15 minutes


@pytest.fixture(scope="module", autouse=True)
def require_hopper_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for reverse-text FP8 integration test.")

    major, _ = torch.cuda.get_device_capability(device=0)
    if major < 9:
        pytest.skip("Reverse-text FP8 integration test requires Hopper GPUs (SM90+).")


@pytest.fixture(scope="module")
def rl_output_dir(output_dir: Path) -> Path:
    rl_dir = output_dir / "reverse_text_fp8"
    rl_dir.mkdir(parents=True, exist_ok=True)
    return rl_dir


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-reverse-text-fp8-{branch_name}"


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    rl_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl/start.toml",
        "--inference.model.quantization",
        "fp8",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        rl_output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


def test_reverse_text_fp8_no_error(rl_process: ProcessResult, rl_output_dir: Path):
    check_no_error(rl_process, rl_output_dir)
