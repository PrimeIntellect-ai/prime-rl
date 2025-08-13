from pathlib import Path
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = [
    "uv",
    "run",
    "orchestrator",
    "@",
    "configs/debug/orch.toml",
    "--model.name",
    "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
]


@pytest.fixture(scope="module")
def orchestrator_process(
    vllm_server, run_process: Callable[[Command, Environment], ProcessResult], outputs_dir: Path
) -> ProcessResult:
    return run_process(CMD + ["--outputs-dir", outputs_dir.as_posix()], {})


def test_no_error(orchestrator_process: ProcessResult):
    assert orchestrator_process.returncode == 0, (
        f"Orchestrator process failed with return code {orchestrator_process.returncode}"
    )
