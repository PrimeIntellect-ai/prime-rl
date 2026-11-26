from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.integration.dashboard_smoke import make_dashboard_test
from tests.utils import check_final_eval_reward_above, check_no_error, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

RUN_NAME = "reverse-text-rl-sft"


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


TIMEOUT = 900  # 15 minutes (was 600s — same overhead as reverse_text)


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-reverse-text-rl-sft:{branch_name}"


@pytest.fixture(scope="module")
def rl_sft_process(
    ref_inference,
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    """Run the RL entrypoint with the sft algorithm; ref_inference is
    a fixture-managed external vLLM at http://localhost:8001/v1."""
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "tests/integration/configs/reverse-text-rl-sft/start.toml",
        "--clean",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_no_error(rl_sft_process: ProcessResult, run_dir: Path):
    check_no_error(rl_sft_process, run_dir)


def test_eval_reward_converges(rl_sft_process: ProcessResult, test_no_error, run_dir: Path):
    with open(run_dir / "logs" / "latest" / "orchestrator.log", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_final_eval_reward_above(orchestrator_stdout, env_name="reverse-text", min_threshold=0.5)


test_dashboard = make_dashboard_test("rl_sft_process", RUN_NAME)
