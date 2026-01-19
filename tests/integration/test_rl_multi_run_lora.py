"""Integration tests for multi-run RL training with LoRA adapters."""

import os
import shutil
import signal
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Generator

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_number_goes_up_or_down, check_number_in_range, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TIMEOUT = 600  # 15 minutes
ORCHESTRATOR_NAMES = ["alpha", "beta", "gamma"]


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for multi-run RL CI integration tests."""
    return f"test-rl-multi-run-{branch_name}"


@pytest.fixture(scope="module")
def multi_run_result(
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> Generator[tuple[dict[str, ProcessResult], str], None, None]:
    """
    Start trainer, inference, and 3 orchestrators.
    Kill one orchestrator halfway and delete its directory.
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env_base = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    processes: list[subprocess.Popen] = []

    # Start inference server
    inference_log = log_dir / "inference.stdout"
    with open(inference_log, "w") as f:
        inference_proc = subprocess.Popen(
            ["uv", "run", "inference", "@", "configs/ci/integration/rl_multi_run/inference.toml"],
            stdout=f,
            stderr=f,
            env={**env_base, "CUDA_VISIBLE_DEVICES": "0"},
        )
    processes.append(inference_proc)

    # Start trainer
    trainer_log = log_dir / "trainer.stdout"
    with open(trainer_log, "w") as f:
        trainer_proc = subprocess.Popen(
            [
                "uv",
                "run",
                "torchrun",
                "--nproc-per-node",
                "1",
                "-m",
                "prime_rl.trainer.rl.train",
                "@",
                "configs/ci/integration/rl_multi_run/trainer.toml",
                "--output-dir",
                output_dir.as_posix(),
                "--wandb.project",
                wandb_project,
                "--wandb.name",
                f"{wandb_name}-trainer",
                "--log.level",
                "debug",
            ],
            stdout=f,
            stderr=f,
            env={**env_base, "CUDA_VISIBLE_DEVICES": "1"},
        )
    processes.append(trainer_proc)
    time.sleep(10)

    # Wait for inference to be ready
    ready_indicators = ["Application startup complete", "Uvicorn running on", "Started server process"]
    start_time = time.time()
    while time.time() - start_time < 300:
        if inference_log.exists():
            content = inference_log.read_text()
            if any(ind in content for ind in ready_indicators):
                break
        time.sleep(2)
    else:
        for p in processes:
            p.terminate()
        pytest.fail("Inference server did not start in time")

    # Wait for trainer to be ready
    ready_indicators = ["Starting training loop"]
    start_time = time.time()
    while time.time() - start_time < 300:
        if trainer_log.exists():
            content = trainer_log.read_text()
            if any(ind in content for ind in ready_indicators):
                break
        time.sleep(2)
    else:
        for p in processes:
            p.terminate()
        pytest.fail("Trainer did not start in time")

    # Start orchestrators
    orch_procs: dict[str, subprocess.Popen] = {}
    for name in ORCHESTRATOR_NAMES:
        run_dir = output_dir / f"run_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        orch_log_dir = run_dir / "logs"
        orch_log_dir.mkdir(parents=True, exist_ok=True)

        with open(orch_log_dir / "orchestrator.stdout", "w") as f:
            proc = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "orchestrator",
                    "@",
                    "configs/ci/integration/rl_multi_run/orchestrator.toml",
                    "--output-dir",
                    run_dir.as_posix(),
                    "--model.lora.name",
                    name,
                    "--wandb.project",
                    wandb_project,
                    "--wandb.name",
                    f"{wandb_name}-{name}",
                ],
                stdout=f,
                stderr=f,
                env=env_base,
            )
        orch_procs[name] = proc
        processes.append(proc)
        time.sleep(2)

    # Wait for alpha to reach step 10, then kill it
    killed_name = "alpha"
    killed_log = output_dir / f"run_{killed_name}" / "logs" / "orchestrator.stdout"
    start_time = time.time()
    while time.time() - start_time < 300:
        if killed_log.exists():
            content = killed_log.read_text()
            if "Step 10" in content or "Step 11" in content or "Step 12" in content:
                break
        time.sleep(2)

    # Kill alpha and delete its directory
    orch_procs[killed_name].send_signal(signal.SIGTERM)
    try:
        orch_procs[killed_name].wait(timeout=30)
    except subprocess.TimeoutExpired:
        orch_procs[killed_name].kill()

    run_dir = output_dir / f"run_{killed_name}"
    while run_dir.exists():
        shutil.rmtree(run_dir)

    # Wait for remaining orchestrators to complete
    remaining_names = [n for n in ORCHESTRATOR_NAMES if n != killed_name]
    for name in remaining_names:
        try:
            orch_procs[name].wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            orch_procs[name].terminate()

    # Build results
    results = {name: ProcessResult(orch_procs[name]) for name in remaining_names}

    yield results, killed_name

    # Cleanup
    for p in processes:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


check_reward_goes_up = partial(check_number_goes_up_or_down, go_up=True, pattern=r"Reward:\s*(\d+\.\d{4})")


def test_remaining_orchestrators_complete(
    multi_run_result: tuple[dict[str, ProcessResult], str],
    output_dir: Path,
):
    """Test that remaining orchestrators complete successfully."""
    results, killed_name = multi_run_result

    for name, result in results.items():
        if result.returncode != 0:
            log_file = output_dir / f"run_{name}" / "logs" / "orchestrator.stdout"
            if log_file.exists():
                print(f"=== {name} Orchestrator Outputs ===")
                print(log_file.read_text()[-5000:])
        assert result.returncode == 0, f"Orchestrator {name} failed with code {result.returncode}"


def test_reward_goes_up(multi_run_result: tuple[dict[str, ProcessResult], str], output_dir: Path):
    """Test that reward goes up for remaining orchestrators."""
    results, _ = multi_run_result

    for name in results.keys():
        log_file = output_dir / f"run_{name}" / "logs" / "orchestrator.stdout"
        with open(log_file, "r") as f:
            lines = strip_escape_codes(f.read()).splitlines()
        check_reward_goes_up(lines)


def test_reward_in_range(multi_run_result: tuple[dict[str, ProcessResult], str], output_dir: Path):
    """Test that final reward is in acceptable range for remaining orchestrators."""
    results, _ = multi_run_result

    for name in results.keys():
        log_file = output_dir / f"run_{name}" / "logs" / "orchestrator.stdout"
        with open(log_file, "r") as f:
            lines = strip_escape_codes(f.read()).splitlines()
        check_number_in_range(lines, step=7, min_threshold=0.2, max_threshold=0.6, pattern=r"Reward:\s*(\d+\.\d{4})")
        check_number_in_range(lines, min_threshold=0.65, pattern=r"Reward:\s*(\d+\.\d{4})")
