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

TIMEOUT = 300  # 5 minutes
ORCHESTRATOR_NAMES = ["alpha", "beta", "gamma"]


def wait_for_log(
    log_file: Path,
    conditions: list[str],
    proc: subprocess.Popen,
    timeout: int = 300,
    poll_interval: float = 0.1,
    sigterm: bool = False,
    kill: bool = False,
) -> None:
    """Wait for any of the conditions to appear in log file, then optionally send SIGTERM and kill the process.

    Args:
        log_file: Path to the log file.
        conditions: List of substrings to wait for.
        proc: Process to kill.
        timeout: Timeout waiting for conditions in seconds.
        poll_interval: Interval in seconds to poll the log file.
        sigterm: Whether to send SIGTERM to the process.
        kill: Whether to kill the process right after sending SIGTERM.
    """
    start_time = time.time()
    print(f"Waiting for conditions {conditions} in {proc.pid}")
    while time.time() - start_time < timeout:
        if log_file.exists():
            content = log_file.read_text()
            if any(cond in content for cond in conditions):
                break
        time.sleep(poll_interval)

    if sigterm:
        print(f"Sending SIGTERM to process {proc.pid}")
        proc.send_signal(signal.SIGTERM)
    if kill:
        print(f"Killing process {proc.pid}")
        proc.kill()
    try:
        proc.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for multi-run RL CI integration tests."""
    return f"test-rl-multi-run-{branch_name}"


@pytest.fixture(scope="module")
def multi_run_result(
    output_dir: Path, wandb_project: str, wandb_name: str, tmp_path_factory
) -> Generator[tuple[dict[str, ProcessResult], str], None, None]:
    """
    Start trainer, inference, and 3 orchestrators.
    Kill one orchestrator halfway and delete its directory.
    """
    tmp_path: Path = tmp_path_factory.mktemp("prime_rl_test_rl_multi_run_lora")
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
                "2",
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
            env={**env_base, "CUDA_VISIBLE_DEVICES": "2,3"},
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

    def start_orchestrator(name: str, max_steps: int, proc_name: str | None = None):
        print(f"Starting orchestrator {name} with proc name {proc_name}")
        if proc_name is None:
            proc_name = name
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
                    "--max-steps",
                    str(max_steps),
                    "--model.lora.name",
                    name,
                    "--wandb.project",
                    wandb_project,
                    "--wandb.name",
                    f"{wandb_name}-{proc_name}",
                ],
                stdout=f,
                stderr=f,
                env=env_base,
            )
        orch_procs[proc_name] = proc
        processes.append(proc)

    # Start orchestrators
    orch_procs: dict[str, subprocess.Popen] = {}
    for name in ORCHESTRATOR_NAMES:
        start_orchestrator(name, max_steps=20)
        time.sleep(5)

    # Wait for alpha to reach step 11, then kill it
    # There is a checkpoint at step 10, so we need to wait for step 11
    killed_name = "alpha"
    killed_log = output_dir / f"run_{killed_name}" / "logs" / "orchestrator.stdout"
    wait_for_log(
        killed_log,
        conditions=["Step 11", "Step 12", "Step 13"],
        proc=orch_procs[killed_name],
        sigterm=True,
    )

    run_dir = output_dir / f"run_{killed_name}"
    alpha_ckpt_dir = run_dir / "checkpoints" / "step_10"
    while not (alpha_ckpt_dir / "trainer").exists():
        print(f"Waiting for {alpha_ckpt_dir / 'trainer'} to exist")
        time.sleep(1)
    while not (alpha_ckpt_dir / "weight").exists():
        print(f"Waiting for {alpha_ckpt_dir / 'weight'} to exist")
        time.sleep(1)
    shutil.copy(run_dir / "logs" / "orchestrator.stdout", log_dir / "alpha_orchestrator.stdout")
    shutil.copytree(alpha_ckpt_dir, tmp_path / "alpha_ckpt_step_10")
    print(f"Copied alpha checkpoint to {tmp_path / 'alpha_ckpt_step_10'}")
    shutil.rmtree(run_dir)

    # Queue alpha's resume proc
    # We cant use the same dir in case the trainer misses the change
    run_dir = output_dir / "run_alpha_resume"
    ckpt_dir = run_dir / "checkpoints" / "step_10"
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp_path / "alpha_ckpt_step_10", ckpt_dir)
    print(f"Copied alpha checkpoint to {ckpt_dir}")
    start_orchestrator("alpha_resume", max_steps=20)

    # Wait for beta to finish
    wait_for_log(
        output_dir / "run_beta" / "logs" / "orchestrator.stdout",
        conditions=["Orchestrator finished."],
        proc=orch_procs["beta"],
        poll_interval=1,
    )

    run_dir = output_dir / "run_beta"
    beta_ckpt_dir = run_dir / "checkpoints" / "step_20"
    while not (beta_ckpt_dir / "trainer").exists():
        time.sleep(1)
        print(f"Waiting for {beta_ckpt_dir / 'trainer'} to exist")
    while not (beta_ckpt_dir / "weight").exists():
        time.sleep(1)
        print(f"Waiting for {beta_ckpt_dir / 'weight'} to exist")
    shutil.copy(run_dir / "logs" / "orchestrator.stdout", log_dir / "beta_orchestrator.stdout")
    shutil.copytree(beta_ckpt_dir, tmp_path / "beta_ckpt_step_20")
    print(f"Copied {beta_ckpt_dir} to {tmp_path / 'beta_ckpt_step_20'}")
    shutil.rmtree(run_dir)

    # Queue beta's resume
    run_dir = output_dir / "run_beta_resume"
    ckpt_dir = run_dir / "checkpoints" / "step_20"
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp_path / "beta_ckpt_step_20", ckpt_dir)
    print(f"Copied beta checkpoint to {ckpt_dir}")
    start_orchestrator("beta_resume", max_steps=25)

    # Wait for gamma to finish
    wait_for_log(
        output_dir / "run_gamma" / "logs" / "orchestrator.stdout",
        conditions=["Orchestrator finished."],
        proc=orch_procs["gamma"],
        timeout=TIMEOUT,
    )
    shutil.copy(output_dir / "run_gamma" / "logs" / "orchestrator.stdout", log_dir / "gamma_orchestrator.stdout")
    shutil.rmtree(output_dir / "run_gamma")

    # Wait for remaining orchestrators to finish
    for name in orch_procs.keys():
        try:
            orch_procs[name].wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            orch_procs[name].terminate()

    # Build results
    results = {name: ProcessResult(orch_procs[name]) for name in orch_procs.keys()}

    # Copy logs from remaining orchestrators to log_dir
    for name in ["alpha_resume", "beta_resume"]:
        src_log = output_dir / f"run_{name}" / "logs" / "orchestrator.stdout"
        if src_log.exists():
            shutil.copy(src_log, log_dir / f"{name}_orchestrator.stdout")

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
    log_dir = output_dir / "logs"

    for name, result in results.items():
        if name == "alpha":  # We sigtermed alpha
            continue
        if result.returncode != 0:
            log_file = log_dir / f"{name}_orchestrator.stdout"
            if log_file.exists():
                print(f"=== {name} Orchestrator Outputs ===")
                print(log_file.read_text()[-5000:])
        assert result.returncode == 0, f"Orchestrator {name} failed with code {result.returncode}"


def test_reward_goes_up(multi_run_result: tuple[dict[str, ProcessResult], str], output_dir: Path):
    """Test that reward goes up for remaining orchestrators."""
    results, _ = multi_run_result
    log_dir = output_dir / "logs"

    print("Test reward goes up", results.keys())
    for name in results.keys():
        if name == "beta_resume":  # Beta is resumed after saturation
            continue
        log_file = log_dir / f"{name}_orchestrator.stdout"
        with open(log_file, "r") as f:
            lines = strip_escape_codes(f.read()).splitlines()
        check_reward_goes_up(lines)


def test_reward_in_range(multi_run_result: tuple[dict[str, ProcessResult], str], output_dir: Path):
    """Test that final reward is in acceptable range for remaining orchestrators."""
    results, _ = multi_run_result
    log_dir = output_dir / "logs"

    print("Test reward in range", results.keys())
    for name in results.keys():
        log_file = log_dir / f"{name}_orchestrator.stdout"
        with open(log_file, "r") as f:
            lines = strip_escape_codes(f.read()).splitlines()
        if name in ["beta", "gamma"]:
            check_number_in_range(
                lines, step=7, min_threshold=0.2, max_threshold=0.6, pattern=r"Reward:\s*(\d+\.\d{4})"
            )
            check_number_in_range(lines, min_threshold=0.65, pattern=r"Reward:\s*(\d+\.\d{4})")
        elif name in ["alpha_resume", "beta_resume"]:
            check_number_in_range(lines, min_threshold=0.65, pattern=r"Reward:\s*(\d+\.\d{4})")
        elif name == "alpha":  # Only had 10 steps, so it's lower
            check_number_in_range(lines, min_threshold=0.4, pattern=r"Reward:\s*(\d+\.\d{4})")
        else:
            pytest.fail(f"Unknown orchestrator {name}")
