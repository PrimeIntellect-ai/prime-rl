import re
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import strip_escape_codes

pytestmark = [pytest.mark.slow]

RUN_NAME = "reverse-text-eval"
TIMEOUT = 600


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


@pytest.fixture(scope="module")
def eval_process(run_process: Callable[..., ProcessResult], output_dir: Path) -> ProcessResult:
    """`uv run eval` against Prime Inference (the default client and model), so the test
    needs `PRIME_API_KEY` and no GPU."""
    cmd = [
        "uv",
        "run",
        "eval",
        "@",
        "configs/ci/integration/reverse-text-eval.toml",
        "--clean",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(eval_process: ProcessResult, run_dir: Path):
    if eval_process.returncode != 0:
        print("=== Eval Outputs ===")
        with open(run_dir / "logs" / "latest" / "eval.log", "r") as f:
            print(*f.readlines()[-200:], sep="")
    assert eval_process.returncode == 0, f"Process has non-zero return code ({eval_process})"


def test_eval_reward(eval_process: ProcessResult, test_no_error, run_dir: Path):
    with open(run_dir / "logs" / "latest" / "eval.log", "r") as f:
        lines = strip_escape_codes(f.read()).splitlines()
    pattern = r"Evaluated reverse-text .*Reward\s+(\d+\.\d{4})"
    matches = [re.search(pattern, line) for line in lines if "SUCCESS" in line]
    matches = [m for m in matches if m]
    assert len(matches) == 1, f"Expected one eval summary line, found {len(matches)}"
    assert float(matches[0].group(1)) >= 0.5


def test_run_artifacts(eval_process: ProcessResult, test_no_error, run_dir: Path):
    assert (run_dir / "configs" / "latest" / "resolved" / "eval.json").is_file()
    assert (run_dir / "monitors" / "file" / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "step_16" / "eval" / "progress.pt").is_file()
