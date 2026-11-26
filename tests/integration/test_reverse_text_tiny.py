from pathlib import Path
from typing import Callable

import orjson
import pytest

from prime_rl.monitors.file.traces import get_index_path, get_trace_stream
from tests.conftest import ProcessResult
from tests.utils import check_avg_mismatch_kl_in_range, check_avg_reward_in_range, check_no_error, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.tiny]

RUN_NAME = "reverse-text-tiny"

TIMEOUT = 300


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    branch_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse-text-tiny/start.toml",
        "--clean",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        f"test-reverse-text-tiny:{branch_name}",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(rl_process: ProcessResult, run_dir: Path):
    check_no_error(rl_process, run_dir)


def test_reward_in_range(rl_process: ProcessResult, test_no_error, run_dir: Path):
    """Five steps are too few for the reward to rise reliably, but the SFT-warmed policy scores well above a broken one."""
    with open(run_dir / "logs" / "latest" / "orchestrator.log", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_avg_reward_in_range(orchestrator_stdout, last_n_steps=5, min_threshold=0.1)


def test_mismatch_kl_in_range(rl_process: ProcessResult, test_no_error, run_dir: Path):
    """A low trainer/inference mismatch KL shows inference serves the weights the trainer broadcast."""
    with open(run_dir / "logs" / "latest" / "trainer.log", "r") as f:
        trainer_stdout = strip_escape_codes(f.read()).splitlines()
    check_avg_mismatch_kl_in_range(trainer_stdout, last_n_steps=3, max_threshold=0.01)


def test_trace_records_name_their_source_and_group(rl_process: ProcessResult, test_no_error, run_dir: Path):
    """Env workers write the records: each names the source and the rollout group it was dispatched for."""
    rows = [orjson.loads(line) for line in get_index_path(get_trace_stream(run_dir)).read_bytes().splitlines()]
    assert rows
    assert {row["env"] for row in rows} == {"reverse-text"}
    assert all(row["group"] for row in rows)
