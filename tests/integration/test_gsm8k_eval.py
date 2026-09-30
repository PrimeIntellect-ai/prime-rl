import re
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import strip_escape_codes

pytestmark = [pytest.mark.slow]

RUN_NAME = "gsm8k-eval"
TIMEOUT = 900


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


@pytest.fixture(scope="module")
def eval_process(run_process: Callable[..., ProcessResult], output_dir: Path) -> ProcessResult:
    """`uv run eval` of single-turn null-harness rollouts against Prime Inference (the
    default client and model); needs `PRIME_API_KEY`, no GPU, no sandbox."""
    cmd = [
        "uv",
        "run",
        "eval",
        "@",
        "configs/ci/integration/gsm8k-eval.toml",
        "--clean",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def summary(eval_process: ProcessResult, run_dir: Path) -> re.Match:
    log = run_dir / "logs" / "latest" / "eval.log"
    if eval_process.returncode != 0:
        print("=== Eval Outputs ===")
        print(*log.read_text().splitlines()[-200:], sep="\n")
    assert eval_process.returncode == 0, f"Process has non-zero return code ({eval_process})"
    lines = strip_escape_codes(log.read_text()).splitlines()
    pattern = r"Evaluated gsm8k .*Reward\s+(?P<reward>\d+\.\d+) \| Turns\s+(?P<turns>\d+\.\d+) \| Branches\s+(?P<branches>\d+\.\d+)"
    matches = [m for m in (re.search(pattern, line) for line in lines if "SUCCESS" in line) if m]
    assert len(matches) == 1, f"Expected one eval summary line, found {len(matches)}"
    return matches[0]


def test_eval_reward(summary: re.Match):
    assert float(summary["reward"]) >= 0.5


def test_rollouts_are_one_branch(summary: re.Match):
    """A linear rollout is exactly one branch."""
    assert float(summary["branches"]) == 1.0
