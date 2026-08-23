from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult

ARCHS = ["glm4_moe", "laguna", "minimax_m2", "qwen3_5_moe_vlm"]

TIMEOUT = 600


@pytest.fixture(scope="module", params=ARCHS)
def arch(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def run_dir(arch: str, output_dir: Path, run_process: Callable[..., ProcessResult]) -> Path:
    """Fixture + all four converter runs for one arch, in a single subprocess."""
    run_dir = output_dir / f"converters-{arch}"
    cmd = ["uv", "run", "python", "tests/converters/run_chain.py", arch, run_dir.as_posix()]
    result = run_process(cmd, timeout=TIMEOUT)
    assert result.returncode == 0, "converter chain failed"
    return run_dir


@pytest.fixture(scope="module")
def source_dir(run_dir: Path) -> Path:
    return run_dir / "source"


@pytest.fixture(scope="module")
def bf16_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "step_1" / "weights"


@pytest.fixture(scope="module")
def fp8_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "step_1" / "weights-FP8"


@pytest.fixture(scope="module")
def fp8_chained_dir(run_dir: Path) -> Path:
    return run_dir / "weights-FP8-chained"


@pytest.fixture(scope="module")
def dequant_dir(run_dir: Path) -> Path:
    return run_dir / "weights-dequant"
