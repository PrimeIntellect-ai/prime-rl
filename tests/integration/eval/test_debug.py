from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = [
    "uv",
    "run",
    "eval",
    "@configs/eval/debug.toml",
    "--model.name",
    "willcb/Qwen2.5-0.5B-Reverse-SFT",
]


@pytest.fixture(scope="module")
def eval_process(vllm_server, run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
    return run_process(CMD, {})


def test_no_error(eval_process: ProcessResult):
    assert eval_process.returncode == 0, f"Eval process failed with return code {eval_process.returncode}"
