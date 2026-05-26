"""Integration coverage for multi-run token export routing."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest

from tests.conftest import ProcessResult
from tests.integration.test_reverse_text_multi_run import INFERENCE_BASE_URLS, INFERENCE_PORTS, TIMEOUT

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

RUN_NAMES = ["alpha", "beta"]


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-reverse-text-multi-run-token-export:{branch_name}"


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        **os.environ,
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache"),
        "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
    }
    if extra:
        env.update(extra)
    return env


def _command(name: str) -> list[str]:
    if os.environ.get("PRIME_RL_TEST_USE_VENV_BIN") == "1":
        return [str(Path(".venv/bin") / name)]
    return ["uv", "run", name]


def _write_token_export_trainer_config(tmp_path: Path) -> Path:
    source = Path("configs/ci/integration/reverse_text_multi_run/trainer.toml")
    config_path = tmp_path / "trainer_token_export.toml"
    config_path.write_text(source.read_text() + "\n[experimental.token_export]\n")
    return config_path


def _start_inference_and_trainer(
    log_dir: Path, output_dir: Path, trainer_config: Path, wandb_project: str, wandb_name: str
) -> tuple[subprocess.Popen, list[subprocess.Popen]]:
    inference_procs: list[subprocess.Popen] = []
    inference_logs: list[Path] = []
    for i, port in enumerate(INFERENCE_PORTS):
        inference_log = log_dir / f"inference_{i}.log"
        inference_logs.append(inference_log)
        with open(inference_log, "w") as f:
            proc = subprocess.Popen(
                [
                    *_command("inference"),
                    "@",
                    "configs/ci/integration/reverse_text_multi_run/inference.toml",
                    "--server.port",
                    str(port),
                ],
                stdout=f,
                stderr=f,
                env=_env({"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "CUDA_VISIBLE_DEVICES": str(i)}),
            )
        inference_procs.append(proc)

    trainer_log = log_dir / "trainer.log"
    with open(trainer_log, "w") as f:
        trainer_proc = subprocess.Popen(
            [
                *_command("torchrun"),
                "--nproc-per-node",
                "2",
                "-m",
                "prime_rl.trainer.rl.train",
                "@",
                trainer_config.as_posix(),
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
            env=_env({"CUDA_VISIBLE_DEVICES": "2,3"}),
        )

    for inference_proc, inference_log in zip(inference_procs, inference_logs):
        _wait_for_log_or_exit(
            inference_log,
            ["Application startup complete", "Uvicorn running on", "Started server process"],
            inference_proc,
        )
    _wait_for_log_or_exit(trainer_log, ["Starting training loop"], trainer_proc)
    return trainer_proc, inference_procs


def _start_orchestrator(name: str, output_dir: Path, wandb_project: str, wandb_name: str) -> subprocess.Popen:
    run_dir = output_dir / f"run_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_command("orchestrator"),
        "@",
        "configs/ci/integration/reverse_text_multi_run/orchestrator.toml",
        "--output-dir",
        run_dir.as_posix(),
        "--max-steps",
        "3",
        "--model.lora.name",
        name,
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        f"{wandb_name}-{name}",
    ]
    cmd.append("--model.client.base-url")
    cmd.extend(INFERENCE_BASE_URLS)

    with open(log_dir / "orchestrator.log", "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=f, env=_env())


def _wait_for_export_files(run_dir: Path) -> list[Path]:
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        export_steps = sorted((run_dir / "token_exports").glob("step_*"))
        export_files = [file for step_dir in export_steps for file in sorted(step_dir.glob("rank_*.jsonl"))]
        if export_files and any((step_dir / "STABLE").exists() for step_dir in export_steps):
            return export_files
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for token exports under {run_dir}")


def _wait_for_log_or_exit(log_file: Path, conditions: list[str], proc: subprocess.Popen) -> None:
    start_time = time.time()
    while True:
        if time.time() - start_time > TIMEOUT:
            raise TimeoutError(f"Timed out waiting for conditions {conditions} in {log_file} after {TIMEOUT}s")
        if log_file.exists():
            content = log_file.read_text()
            if any(condition in content for condition in conditions):
                return
        returncode = proc.poll()
        if returncode is not None:
            tail = log_file.read_text()[-5000:] if log_file.exists() else ""
            raise RuntimeError(f"Process {proc.pid} exited with code {returncode} before {conditions}:\n{tail}")
        time.sleep(0.5)


@pytest.fixture(scope="module")
def multi_run_token_export_result(
    output_dir: Path, wandb_project: str, wandb_name: str, tmp_path_factory: pytest.TempPathFactory
) -> Generator[dict[str, ProcessResult], None, None]:
    tmp_path = tmp_path_factory.mktemp("prime_rl_test_multi_run_token_export")
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer_config = _write_token_export_trainer_config(tmp_path)

    processes: list[subprocess.Popen] = []
    trainer_proc, inference_procs = _start_inference_and_trainer(
        log_dir, output_dir, trainer_config, wandb_project, wandb_name
    )
    processes.append(trainer_proc)
    processes.extend(inference_procs)

    orchestrators = {name: _start_orchestrator(name, output_dir, wandb_project, wandb_name) for name in RUN_NAMES}
    processes.extend(orchestrators.values())

    for name, proc in orchestrators.items():
        _wait_for_log_or_exit(
            output_dir / f"run_{name}" / "logs" / "orchestrator.log", ["Orchestrator finished."], proc
        )
        proc.wait(timeout=TIMEOUT)
        _wait_for_export_files(output_dir / f"run_{name}")

    yield {name: ProcessResult(proc) for name, proc in orchestrators.items()}

    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_token_exports_are_run_local(multi_run_token_export_result: dict[str, ProcessResult], output_dir: Path) -> None:
    assert not (output_dir / "token_exports").exists()

    for name, result in multi_run_token_export_result.items():
        assert result.returncode == 0
        run_id = f"run_{name}"
        export_steps = sorted((output_dir / run_id / "token_exports").glob("step_*"))
        stable_steps = [step_dir for step_dir in export_steps if (step_dir / "STABLE").exists()]
        assert stable_steps, f"No stable token export steps found for {run_id}"

        export_files = sorted((output_dir / run_id / "token_exports").glob("step_*/rank_*.jsonl"))
        assert export_files, f"No token exports found for {run_id}"

        records = []
        deadline = time.time() + TIMEOUT
        while not records and time.time() < deadline:
            for export_file in export_files:
                records.extend(json.loads(line) for line in export_file.read_text().splitlines() if line)
            if not records:
                time.sleep(1)
        assert records, f"No token export records found for {run_id}"

        for record in records:
            assert record["run_id"] == run_id
            assert record["export_step"] >= 0
            assert len(record["token_ids"]) == len(record["importance_ratio"])
            assert len(record["token_ids"]) == len(record["mismatch_kl"])
            assert len(record["token_ids"]) == len(record["prob_delta"])
