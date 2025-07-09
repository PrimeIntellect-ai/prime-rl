from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = [
    "uv",
    "run",
    "orchestrator",
    "@configs/training/orchestrator/debug.toml",
    "--model.name",
    "willcb/Qwen2.5-0.5B-Reverse-SFT",
]


@pytest.fixture(scope="module")
def orchestrator_process(
    vllm_server: str, run_process: Callable[[Command, Environment], ProcessResult]
) -> ProcessResult:
    return run_process(CMD, {})


def test_no_error(orchestrator_process: ProcessResult):
    assert orchestrator_process.returncode == 0, (
        f"Orchestrator process failed with return code {orchestrator_process.returncode}"
    )
