"""Runtime control-file helpers for multi-run LoRA sweeps."""

from pathlib import Path
from typing import Any

EXIT_CODE_FILENAME = "exit_code"
EVICTED_FILENAME = "evicted.txt"


def _mark_failed_orchestrator_evicted(run_dir: Path, code: int) -> None:
    """Hide a failed run from the shared trainer's future directory scans."""
    if code == 0:
        return

    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    evicted_path = control_dir / EVICTED_FILENAME
    if not evicted_path.exists():
        evicted_path.write_text(f"orchestrator exited with code {code}\n")


def write_orchestrator_exit_code(run_dir: Path, returncode: int | None) -> None:
    """Write a per-orchestrator returncode for the sweep controller to reconcile.

    The sweep controller reads each ``<run_dir>/control/exit_code`` after the
    multi-run invocation exits, so it can attribute failures to the actual
    orchestrator that crashed instead of marking every trial in the wave
    failed. ``None`` means "the launcher tore down the orchestrator before it
    produced an exit code"; we record ``-1`` so the controller treats it as an
    infrastructure failure. Non-zero exits also write ``evicted.txt`` when the
    controller/trainer did not already create one, so the shared trainer
    forgets crashed runs instead of waiting on them for the rest of the wave.
    """
    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    code = -1 if returncode is None else int(returncode)
    (control_dir / EXIT_CODE_FILENAME).write_text(f"{code}\n")
    _mark_failed_orchestrator_evicted(run_dir, code)


def record_orchestrator_exit_codes(orchestrator_processes: list[Any], run_dirs: list[Path]) -> None:
    """Best-effort: write exit_code for every run dir, swallowing per-run write errors.

    A failure to write one exit_code must not prevent the others from being
    recorded; the controller falls back to "infrastructure failure" when the
    file is missing, which is at least diagnosable.
    """
    for proc, run_dir in zip(orchestrator_processes, run_dirs):
        try:
            write_orchestrator_exit_code(run_dir, proc.returncode)
        except OSError:
            continue


def record_finished_orchestrator_exit_codes(
    orchestrator_processes: list[Any],
    run_dirs: list[Path],
    recorded_run_dirs: set[Path],
) -> None:
    """Publish per-run exit codes as orchestrators finish during a wave."""
    for proc, run_dir in zip(orchestrator_processes, run_dirs):
        if run_dir in recorded_run_dirs or proc.poll() is None:
            continue
        try:
            write_orchestrator_exit_code(run_dir, proc.returncode)
        except OSError:
            continue
        recorded_run_dirs.add(run_dir)


def finished_orchestrator_failures(
    orchestrator_labels: list[str],
    orchestrator_processes: list[Any],
    stop_events: dict[str, Any],
) -> list[tuple[str, int | None]]:
    """Return failed orchestrators only once every orchestrator has stopped."""
    if not all(stop_events[label].is_set() for label in orchestrator_labels):
        return []
    return [
        (label, proc.returncode)
        for label, proc in zip(orchestrator_labels, orchestrator_processes)
        if proc.returncode != 0
    ]
