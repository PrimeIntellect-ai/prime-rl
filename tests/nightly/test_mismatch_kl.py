"""Mismatch KL between the trainer and vLLM for every architecture that fits on one 8-GPU node.

Each model runs a short, fully on-policy reverse-text RL job (`configs/ci/nightly-kl/base.toml` plus the
model's overlay), so mismatch KL measures only the numerical gap between the trainer's modeling code and
vLLM's: step 1 on the checkpoint, the mean over all steps on the updated weights vLLM received from the
trainer, which also covers the weight broadcast and conversion.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import (
    check_avg_mismatch_kl_in_range,
    check_mismatch_kl_in_range,
    check_no_error,
    strip_escape_codes,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

CONFIG_DIR = Path("configs/ci/nightly-kl")
NUM_STEPS = 10


@dataclass(frozen=True)
class KLBudget:
    step1: float
    mean: float


# Budgets sit a few times above on-policy KL measured on H200s, with a floor that absorbs run-to-run noise.
KL_BUDGETS = {
    "llama-3.1-8b": KLBudget(step1=0.005, mean=0.005),
    "qwen3-8b": KLBudget(step1=0.005, mean=0.005),
    "qwen3.5-9b": KLBudget(step1=0.005, mean=0.005),
    "qwen3-30b-a3b": KLBudget(step1=0.01, mean=0.01),
    "qwen3.5-35b-a3b": KLBudget(step1=0.005, mean=0.005),
    "gpt-oss-20b": KLBudget(step1=0.005, mean=0.005),
    "nemotron-3.5-lightning": KLBudget(step1=0.02, mean=0.02),
    # Laguna's own run-to-run KL on real text is ~0.1, so this only catches gross breakage.
    "laguna-xs.2": KLBudget(step1=0.2, mean=0.2),
}


@pytest.fixture(scope="module", params=list(KL_BUDGETS))
def model(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def run_dir(output_dir: Path, model: str) -> Path:
    return output_dir / f"mismatch-kl-{model}"


@pytest.fixture(scope="module")
def rl_process(
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    run_dir: Path,
    wandb_project: str,
    branch_name: str,
    model: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        (CONFIG_DIR / "base.toml").as_posix(),
        "@",
        (CONFIG_DIR / f"{model}.toml").as_posix(),
        "--max-steps",
        str(NUM_STEPS),
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        f"mismatch-kl-{model}-{branch_name}",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        run_dir.name,
    ]
    return run_process(cmd)


@pytest.fixture(scope="module")
def trainer_lines(rl_process: ProcessResult, run_dir: Path) -> list[str]:
    check_no_error(rl_process, run_dir)
    with open(run_dir / "logs" / "latest" / "trainer.log") as f:
        return strip_escape_codes(f.read()).splitlines()


def test_on_policy_mismatch_kl(trainer_lines: list[str], model: str):
    check_mismatch_kl_in_range(trainer_lines, step=1, max_threshold=KL_BUDGETS[model].step1)


def test_mean_mismatch_kl(trainer_lines: list[str], model: str):
    check_avg_mismatch_kl_in_range(trainer_lines, last_n_steps=NUM_STEPS, max_threshold=KL_BUDGETS[model].mean)
