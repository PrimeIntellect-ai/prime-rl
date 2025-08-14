from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = [
    "uv",
    "run",
    "eval",
    "@",
    "configs/debug/eval.toml",
    "--model.name",
    "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
]
ENV = {"CUDA_VISIBLE_DEVICES": "1"}


@pytest.fixture(scope="module")
def eval_process(vllm_server, run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
    return run_process(CMD, ENV)


def test_no_error(eval_process: ProcessResult):
    assert eval_process.returncode == 0, f"Eval process failed with return code {eval_process.returncode}"
