from pathlib import Path
from typing import Callable

import pytest
import torch
from transformers import AutoModelForCausalLM

from tests.conftest import ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

RUN_NAME = "dcp-to-hf"

TIMEOUT = 900  # 15 minutes


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


@pytest.fixture(scope="module")
def sft_process(run_process: Callable[..., ProcessResult], output_dir: Path) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        "configs/debug/fake/sft.toml",
        "--model.name",
        "samsja/mini-glm-moe",
        "--max-steps",
        "2",
        "--ckpt.interval",
        "2",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
        "--clean",
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def convert_process(
    sft_process: ProcessResult, run_process: Callable[..., ProcessResult], run_dir: Path
) -> ProcessResult:
    assert sft_process.returncode == 0
    cmd = [
        "uv",
        "run",
        "python",
        "tools/converters/dcp_to_hf.py",
        (run_dir / "checkpoints" / "step_2").as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


def test_export_loads_as_hf_model(convert_process: ProcessResult, run_dir: Path):
    """The exported dir loads as an HF model with no missing/unexpected tensors."""
    assert convert_process.returncode == 0
    weights_dir = run_dir / "checkpoints" / "step_2" / "weights"

    model, loading_info = AutoModelForCausalLM.from_pretrained(weights_dir, output_loading_info=True)
    assert not loading_info["missing_keys"], f"missing keys: {loading_info['missing_keys'][:5]}"
    assert not loading_info["unexpected_keys"], f"unexpected keys: {loading_info['unexpected_keys'][:5]}"
    assert not loading_info["mismatched_keys"], f"mismatched keys: {loading_info['mismatched_keys'][:5]}"
    assert all(torch.isfinite(param).all() for param in model.parameters())
