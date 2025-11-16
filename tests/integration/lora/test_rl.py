import json
from pathlib import Path
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


TIMEOUT = 600  # 10 minutes
RL_CMD = [
    "uv",
    "run",
    "rl",
    "--trainer",
    "@",
    "configs/reverse_text/rl/train.toml",
    "--orchestrator",
    "@",
    "configs/reverse_text/rl/orch.toml",
    "--orchestrator.sampling.max-tokens",
    "128",
    "--trainer.model.experimental.lora.rank",
    "8",
    "--trainer.weights.save-adapter-separately",
    "--orchestrator.lora-name",
    "r8-1e-4",
    "--trainer.optim.lr",
    "1e-4",
    "--ckpt",
]
RL_RESUME_CMD = [
    "uv",
    "run",
    "rl",
    "--trainer",
    "@",
    "configs/reverse_text/rl/train.toml",
    "--orchestrator",
    "@",
    "configs/reverse_text/rl/orch.toml",
    "--orchestrator.sampling.max-tokens",
    "128",
    "--trainer.model.experimental.lora.rank",
    "8",
    "--trainer.weights.save-adapter-separately",
    "--orchestrator.lora-name",
    "r8-1e-4",
    "--trainer.optim.lr",
    "1e-4",
    "--max-steps",
    "25",
    "--ckpt.resume-step",
    "20",
]


@pytest.fixture(scope="module")
def wandb_project(username: str) -> str:
    project = "ci-reverse-text-rl"
    if username != "CI_RUNNER":
        project += "-local"
    return project


@pytest.fixture(scope="module")
def lora_rl_process(
    vllm_server_dynamic_lora_loading,  # Can only run with vLLM server
    run_process: Callable[[Command, Environment, int], ProcessResult],
    output_dir: Path,
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}"

    return run_process(
        RL_CMD + ["--wandb.project", wandb_project, "--wandb.name", wandb_name, "--output-dir", output_dir.as_posix()],
        {},
        TIMEOUT,
    )


@pytest.fixture(scope="module")
def lora_rl_resume_process(
    vllm_server,  # Can only run with vLLM server
    lora_rl_process,  # Resume training can only start when regular RL process is finished
    run_process: Callable[[Command, Environment, int], ProcessResult],
    output_dir: Path,
    wandb_project: str,
    branch_name: str,
    commit_hash: str,
) -> ProcessResult:
    wandb_name = f"{branch_name}-{commit_hash}-resume"

    return run_process(
        RL_RESUME_CMD
        + ["--wandb.project", wandb_project, "--wandb.name", wandb_name, "--output-dir", output_dir.as_posix()],
        {},
        TIMEOUT,
    )


def test_no_error(lora_rl_process: ProcessResult):
    assert lora_rl_process.returncode == 0, f"RL process failed with return code {lora_rl_process.returncode}"


def test_no_error_resume(lora_rl_resume_process: ProcessResult):
    assert lora_rl_resume_process.returncode == 0, (
        f"RL resume process failed with return code {lora_rl_resume_process.returncode}"
    )


def test_check_reward(output_dir: Path, lora_rl_resume_process: ProcessResult):
    wandb_paths = [i for i in output_dir.glob("run-*")]
    wandb_summaries = [json.load(open(i / "final_summary.json")) for i in wandb_paths]
    assert len(wandb_paths) == 2
    for wandb_summary in wandb_summaries:
        assert "reward/mean" in wandb_summary
        assert wandb_summary["reward/mean"] > 0.65


# would need the setup a vllm server with the nccl broadcast enabled to make this work
@pytest.mark.skip(reason="Skipping NCCL broadcast as it fail only in ci")
def test_rl_nccl(run_process):
    process = run_process(
        RL_CMD + ["--weight-broadcast.type", "nccl"],
        {},
        TIMEOUT,
    )
    assert process.returncode == 0, f"RL process failed with return code {process.returncode}"
