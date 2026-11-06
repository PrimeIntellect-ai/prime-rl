import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import psutil
import pytest

from prime_rl.utils.process import cleanup_process
from tests.conftest import ProcessResult
from tests.utils import check_no_error, check_reward_goes_up, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

RUN_NAME = "reverse-text-recovery"
TIMEOUT = 1200
KILL_AFTER_STEP = 4


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


def _kill_trainer_after_step(launcher: subprocess.Popen, run_dir: Path, killed: list[int]) -> None:
    """SIGKILL the trainer worker once it logs the step, like a crashed GPU process."""
    log = run_dir / "logs" / "latest" / "trainer.log"
    while launcher.poll() is None:
        if log.exists() and re.search(rf"Step {KILL_AFTER_STEP} \|", strip_escape_codes(log.read_text())):
            for proc in psutil.Process(launcher.pid).children(recursive=True):
                if "PRL::Trainer" in " ".join(proc.cmdline()):
                    proc.send_signal(signal.SIGKILL)
                    killed.append(proc.pid)
            return
        time.sleep(0.2)


@pytest.fixture(scope="module")
def rl_process(output_dir: Path, run_dir: Path) -> tuple[ProcessResult, list[int]]:
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/reverse-text-recovery/start.toml",
        "--clean",
        "--no-monitors.prime",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]
    launcher = subprocess.Popen(cmd)
    killed: list[int] = []
    killer = threading.Thread(target=_kill_trainer_after_step, args=(launcher, run_dir, killed), daemon=True)
    killer.start()
    try:
        launcher.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        cleanup_process(launcher.pid, signal.SIGTERM)
        try:
            launcher.wait(timeout=30)
        except subprocess.TimeoutExpired:
            cleanup_process(launcher.pid, signal.SIGKILL)
            launcher.wait()
    return ProcessResult(launcher), killed


def test_no_error(rl_process, run_dir: Path):
    check_no_error(rl_process[0], run_dir)


def test_trainer_was_replaced(rl_process, run_dir: Path):
    assert rl_process[1], "The trainer was never killed"
    orchestrator = strip_escape_codes((run_dir / "logs" / "latest" / "orchestrator.log").read_text())
    assert "Admitting replacement trainer attempt" in orchestrator


def test_replayed_step_reproduces_metrics(rl_process, run_dir: Path):
    """The replacement restores the step-3 checkpoint and retrains step 4 on the journaled batch."""
    trainer = strip_escape_codes((run_dir / "logs" / "latest" / "trainer.log").read_text())
    replays = re.findall(rf"Step {KILL_AFTER_STEP} \|.*?Loss (-?\d+\.\d+).*?Grad\. Norm (\d+\.\d+)", trainer)
    assert len(replays) == 2 and replays[0] == replays[1], replays
    assert re.search(r"Step 15 \|", trainer)


def test_reward_goes_up(rl_process, run_dir: Path):
    orchestrator = strip_escape_codes((run_dir / "logs" / "latest" / "orchestrator.log").read_text())
    check_reward_goes_up(orchestrator.splitlines())
