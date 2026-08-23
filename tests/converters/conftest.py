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
def run_converter(run_process: Callable[..., ProcessResult]) -> Callable[..., None]:
    def _run(script: str, *args: Path | str) -> None:
        cmd = ["uv", "run", "python", script, *[str(arg) for arg in args]]
        result = run_process(cmd, timeout=TIMEOUT)
        assert result.returncode == 0, f"{script} failed"

    return _run


@pytest.fixture(scope="module")
def run_dir(arch: str, output_dir: Path, run_converter: Callable[..., None]) -> Path:
    run_dir = output_dir / f"converters-{arch}"
    run_converter("tests/converters/make_fixture.py", arch, run_dir)
    return run_dir


@pytest.fixture(scope="module")
def source_dir(run_dir: Path) -> Path:
    return run_dir / "source"


@pytest.fixture(scope="module")
def bf16_dir(run_dir: Path, run_converter: Callable[..., None]) -> Path:
    step_dir = run_dir / "checkpoints" / "step_1"
    run_converter("tools/converters/dcp_to_bf16.py", step_dir)
    return step_dir / "weights"


@pytest.fixture(scope="module")
def fp8_dir(run_dir: Path, run_converter: Callable[..., None]) -> Path:
    step_dir = run_dir / "checkpoints" / "step_1"
    run_converter("tools/converters/dcp_to_fp8.py", step_dir)
    return step_dir / "weights-FP8"


@pytest.fixture(scope="module")
def fp8_chained_dir(run_dir: Path, bf16_dir: Path, run_converter: Callable[..., None]) -> Path:
    out_dir = run_dir / "weights-FP8-chained"
    run_converter("tools/converters/bf16_to_fp8.py", bf16_dir, out_dir)
    return out_dir


@pytest.fixture(scope="module")
def dequant_dir(run_dir: Path, fp8_dir: Path, run_converter: Callable[..., None]) -> Path:
    out_dir = run_dir / "weights-dequant"
    run_converter("tools/converters/fp8_to_bf16.py", fp8_dir, out_dir)
    return out_dir
