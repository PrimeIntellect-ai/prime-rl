import contextlib
import os
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable

import psutil
import pynvml
import pytest

from tests.regression.distributed.sft.cases import (
    CASES,
    Case,
    assert_training_rows,
    completed_step,
    failure_report,
    graph_break_reports,
    load_metrics,
    ranks_without_dynamo_logs,
    sft_args,
)

pytestmark = [pytest.mark.distributed, pytest.mark.gpu]

TRAIN_STEPS, RESUME_STEPS, TIMEOUT_SEC = 5, 3, 600
LAUNCH_MARKER = "PRL_TEST_LAUNCH"
BUSY_GPU_MIB = 1024
MAX_TIMEOUTS_IN_A_ROW = 2


@dataclass
class LaunchAttempt:
    label: str
    returncode: int
    timed_out: bool
    log_dir: Path | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def describe(self) -> str:
        outcome = f"timed out after {TIMEOUT_SEC} s" if self.timed_out else f"exited with code {self.returncode}"
        report = failure_report(self.log_dir) if self.log_dir else "no logs: the launch ended before creating them"
        return f"{self.label} launch {outcome}\n{report}"


@dataclass
class Launch:
    run_dir: Path
    attempts: list[LaunchAttempt]


def busy_gpus() -> list[str]:
    """Visible GPUs that already hold memory, as `GPU <id>: <GiB> GiB used`."""
    pynvml.nvmlInit()
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpus = visible.split(",") if visible else [str(index) for index in range(pynvml.nvmlDeviceGetCount())]
        busy = []
        for gpu in gpus:
            handle = (
                pynvml.nvmlDeviceGetHandleByIndex(int(gpu)) if gpu.isdigit() else pynvml.nvmlDeviceGetHandleByUUID(gpu)
            )
            used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            if used > BUSY_GPU_MIB * 2**20:
                busy.append(f"GPU {gpu}: {used / 2**30:.1f} GiB used")
        return busy
    finally:
        pynvml.nvmlShutdown()


def started_by_launch(process: psutil.Process, launch_id: str) -> bool:
    try:
        return process.environ().get(LAUNCH_MARKER) == launch_id
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def kill_leftover_processes(launch_id: str) -> None:
    """Kill every process a finished launch left behind, such as torchrun workers orphaned in their own session."""
    leftovers = [process for process in psutil.process_iter() if started_by_launch(process, launch_id)]
    for process in leftovers:
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill()
    psutil.wait_procs(leftovers, timeout=30)


@pytest.fixture(scope="module")
def launch(tiny_checkpoint, tokenizer_files, run_process, output_dir) -> Callable[[Case], Launch]:
    """Run each case's `uv run sft` launches once, shared by every test that checks that case."""
    timed_out_in_a_row: list[str] = []

    @cache
    def _launch(case: Case) -> Launch:
        model_name = tiny_checkpoint(case.arch.name).as_posix()
        tokenizer_files(case.arch.name)
        run_dir = output_dir / case.id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        env = {LAUNCH_MARKER: case.id, "HF_HUB_OFFLINE": "1"}
        if case.check_graph_breaks:
            env["TORCH_LOGS"] = "graph_breaks,dynamo"

        def sft(label: str, *extra: str) -> LaunchAttempt:
            if busy := busy_gpus():
                pytest.fail(f"GPUs already hold memory before the {label} launch:\n" + "\n".join(busy), pytrace=False)
            cmd = ["uv", "run", "--no-sync", "sft", *sft_args(case, model_name), "--output-dir", output_dir.as_posix()]
            cmd += ["--run.name", case.id, *extra]
            result = run_process(cmd, env=env, timeout=TIMEOUT_SEC)
            kill_leftover_processes(case.id)
            latest_log_dir = run_dir / "logs" / "latest"
            attempt = LaunchAttempt(
                label,
                result.returncode,
                result.timed_out,
                latest_log_dir.resolve() if latest_log_dir.exists() else None,
            )
            timed_out_in_a_row[:] = [*timed_out_in_a_row, f"{case.id} ({label})"] if attempt.timed_out else []
            if len(timed_out_in_a_row) >= MAX_TIMEOUTS_IN_A_ROW:
                pytest.exit(
                    f"{len(timed_out_in_a_row)} launches in a row timed out, which points to a hang in shared code: "
                    + ", ".join(timed_out_in_a_row),
                    returncode=1,
                )
            return attempt

        attempts = [sft("train", "--max-steps", str(TRAIN_STEPS), *(["--ckpt"] if case.resume else []))]
        if case.resume and attempts[0].ok:
            attempts.append(
                sft("resume", "--max-steps", str(TRAIN_STEPS + RESUME_STEPS), "--resume.step", str(TRAIN_STEPS))
            )
        return Launch(run_dir, attempts)

    return _launch


@pytest.mark.parametrize("case", [case.param() for case in CASES])
def test_sft(case, launch):
    result = launch(case)
    for attempt in result.attempts:
        if not attempt.ok:
            pytest.fail(f"run dir: {result.run_dir}\n{attempt.describe()}", pytrace=False)
    last_step = TRAIN_STEPS + RESUME_STEPS if case.resume else TRAIN_STEPS
    assert_training_rows(load_metrics(result.run_dir), range(1, last_step + 1))


@pytest.mark.parametrize("case", [case.param() for case in CASES if case.check_graph_breaks])
def test_no_graph_breaks(case, launch):
    result = launch(case)
    metrics = result.run_dir / "monitors" / "file" / "metrics.jsonl"
    if not metrics.exists() or not completed_step(load_metrics(result.run_dir), 1):
        pytest.skip("launch ended before step 1; graph breaks not checked")
    missing = ranks_without_dynamo_logs(result.run_dir)
    assert not missing, f"dynamo logging did not reach {missing}, so graph breaks cannot be checked"
    breaks = graph_break_reports(result.run_dir)
    assert not breaks, f"{len(breaks)} graph break(s):\n" + "\n".join(breaks)
