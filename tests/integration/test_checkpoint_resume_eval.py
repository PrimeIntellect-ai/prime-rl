import re
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import strip_escape_codes

pytestmark = [pytest.mark.slow]

RUN_NAME = "checkpoint-resume-eval"
TIMEOUT = 900
FIXTURES = Path("deps/verifiers/tests/integration/fixtures")


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


@pytest.fixture(scope="module")
def eval_process(run_process: Callable[..., ProcessResult], output_dir: Path) -> ProcessResult:
    """`uv run eval` of one bash-harness rollout in a prime sandbox whose first attempt
    fails mid-rollout; needs `PRIME_API_KEY` (sandbox + Prime Inference), no GPU."""
    cmd = [
        "uv",
        "run",
        "eval",
        "@",
        "configs/ci/integration/checkpoint-resume-eval.toml",
        "--clean",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]
    return run_process(cmd, env={"PYTHONPATH": FIXTURES.resolve().as_posix()}, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def logs(eval_process: ProcessResult, run_dir: Path) -> str:
    """The eval log and the env server's (the rollout itself runs in the env server)."""
    log_dir = run_dir / "logs" / "latest"
    text = "\n".join(strip_escape_codes(path.read_text()) for path in sorted(log_dir.rglob("*.log")))
    if eval_process.returncode != 0:
        print("=== Eval Outputs ===")
        print(*text.splitlines()[-200:], sep="\n")
    assert eval_process.returncode == 0, f"Process has non-zero return code ({eval_process})"
    return text


@pytest.fixture(scope="module")
def summary(logs: str) -> re.Match:
    pattern = (
        r"Evaluated checkpoint-resume-v1 .*Reward\s+(?P<reward>\d+\.\d+) \| Turns\s+(?P<turns>\d+\.\d+)"
        r" \| Branches\s+(?P<branches>\d+\.\d+) \| Error\s+(?P<error>\d+\.\d+)%"
    )
    matches = [m for m in (re.search(pattern, line) for line in logs.splitlines() if "SUCCESS" in line) if m]
    assert len(matches) == 1, f"Expected one eval summary line, found {len(matches)}"
    return matches[0]


def test_retry_resumed_from_checkpoint(logs: str):
    """The failed attempt's retry restored the sandbox from its newest checkpoint instead
    of starting over."""
    assert re.search(r"retrying agent rollout \(retry 1/1\).*resuming from checkpoint", logs)
    assert re.search(r"rollout resume: .*runtime=prime checkpoint=\S+ turns=1", logs)


def test_resumed_rollout_solved_the_task(summary: re.Match):
    """The resumed attempt built on the turn the checkpoint preserved: full reward, no
    error on the final trace, one branch."""
    assert float(summary["reward"]) == 1.0
    assert float(summary["turns"]) >= 2.0
    assert float(summary["branches"]) == 1.0
    assert float(summary["error"]) == 0.0
