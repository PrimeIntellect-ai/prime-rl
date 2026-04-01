from pathlib import Path
from typing import Callable

import pytest

from prime_rl.trainer.weights import load_state_dict
from tests.conftest import ProcessResult
from tests.utils import (
    check_loss_goes_down,
    check_metric_in_range,
    check_no_error,
    check_reward_goes_up,
    check_reward_in_range,
    strip_escape_codes,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


TIMEOUT = 600
SFT_STEP = 100
RL_STEP = 20


def read_log_lines(*paths: Path) -> list[str]:
    for path in paths:
        if path.exists():
            with open(path, "r") as f:
                return strip_escape_codes(f.read()).splitlines()
    raise FileNotFoundError(f"None of the expected log paths exist: {[p.as_posix() for p in paths]}")


def assert_adapter_checkpoint(adapter_dir: Path) -> None:
    assert (adapter_dir / "adapter_config.json").exists()
    state_dict = load_state_dict(adapter_dir)
    assert state_dict
    assert all(key.startswith("base_model.model.") for key in state_dict)
    assert any(key.endswith("lora_A.weight") for key in state_dict)
    assert any(key.endswith("lora_B.weight") for key in state_dict)


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-rl-lora-init-continuation-{branch_name}"


@pytest.fixture(scope="module")
def sft_output_dir(output_dir: Path) -> Path:
    path = output_dir / "sft_lora"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def rl_output_dir(output_dir: Path) -> Path:
    path = output_dir / "rl_lora_init"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def sft_lora_process(
    run_process: Callable[..., ProcessResult],
    wandb_project: str,
    branch_name: str,
    sft_output_dir: Path,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        "configs/ci/integration/rl_lora_init/sft.toml",
        "--deployment.num-gpus",
        "2",
        "--clean-output-dir",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"test-sft-lora-init-continuation-{branch_name}",
        "--output-dir",
        sft_output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def init_adapter_dir(sft_lora_process: ProcessResult, sft_output_dir: Path) -> Path:
    assert sft_lora_process.returncode == 0, f"SFT process has non-zero return code ({sft_lora_process})"
    adapter_dir = sft_output_dir / "weights" / f"step_{SFT_STEP}" / "lora_adapters"
    assert_adapter_checkpoint(adapter_dir)
    return adapter_dir


@pytest.fixture(scope="module")
def rl_process(
    init_adapter_dir: Path,
    run_process: Callable[..., ProcessResult],
    rl_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl_lora_init/start.toml",
        "--trainer.model.lora.init_adapter_path",
        init_adapter_dir.as_posix(),
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        rl_output_dir.as_posix(),
    ]
    return run_process(cmd, env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def rl_resume_process(
    rl_process,
    init_adapter_dir: Path,
    run_process: Callable[..., ProcessResult],
    rl_output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    assert rl_process.returncode == 0, f"RL init process has non-zero return code ({rl_process})"
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl_lora_init/resume.toml",
        "--trainer.model.lora.init_adapter_path",
        init_adapter_dir.as_posix(),
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"{wandb_name}-resume",
        "--output-dir",
        rl_output_dir.as_posix(),
    ]
    return run_process(cmd, env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}, timeout=TIMEOUT)


def test_sft_lora_no_error(sft_lora_process: ProcessResult):
    assert sft_lora_process.returncode == 0, f"Process has non-zero return code ({sft_lora_process})"


def test_sft_lora_loss_goes_down(sft_lora_process: ProcessResult, sft_output_dir: Path):
    trainer_stdout = read_log_lines(
        sft_output_dir / "logs" / "trainer.log",
        sft_output_dir / "logs" / "trainer" / "rank_0.log",
    )
    check_loss_goes_down(trainer_stdout)
    check_metric_in_range(trainer_stdout, metric_name="Loss", pattern=r"Loss:\s*(\d+\.\d{4})", min_threshold=None, max_threshold=1.5)


def test_init_adapter_checkpoint_written(init_adapter_dir: Path):
    assert_adapter_checkpoint(init_adapter_dir)


@pytest.fixture(scope="module")
def rl_no_error(rl_process: ProcessResult, rl_output_dir: Path):
    check_no_error(rl_process, rl_output_dir)


def test_reward_goes_up(rl_process: ProcessResult, rl_no_error, rl_output_dir: Path):
    orchestrator_stdout = read_log_lines(
        rl_output_dir / "logs" / "orchestrator.log",
        rl_output_dir / "logs" / "orchestrator.stdout",
    )
    check_reward_goes_up(orchestrator_stdout)


def test_reward_in_range(rl_process: ProcessResult, rl_no_error, rl_output_dir: Path):
    orchestrator_stdout = read_log_lines(
        rl_output_dir / "logs" / "orchestrator.log",
        rl_output_dir / "logs" / "orchestrator.stdout",
    )
    check_reward_in_range(orchestrator_stdout, min_threshold=0.65)


def test_rl_adapter_checkpoint_written(rl_process: ProcessResult, rl_no_error, rl_output_dir: Path):
    adapter_dir = rl_output_dir / "weights" / f"step_{RL_STEP}" / "lora_adapters"
    assert_adapter_checkpoint(adapter_dir)


@pytest.fixture(scope="module")
def rl_resume_no_error(rl_resume_process: ProcessResult, rl_output_dir: Path):
    check_no_error(rl_resume_process, rl_output_dir)


def test_reward_in_range_resume(rl_resume_process: ProcessResult, rl_resume_no_error, rl_output_dir: Path):
    orchestrator_stdout = read_log_lines(
        rl_output_dir / "logs" / "orchestrator.log",
        rl_output_dir / "logs" / "orchestrator.stdout",
    )
    check_reward_in_range(orchestrator_stdout, min_threshold=0.65)
