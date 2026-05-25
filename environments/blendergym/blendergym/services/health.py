"""Service health sentinel protocol.

Owns the entire sentinel lifecycle:
- Server side: clear_sentinels(), report_ready(), report_crash()
- Client side: wait_for_service(), ensure_service_ready()
- Runtime:     diagnose_service_down()

SENTINEL_DIR is the shared path convention between service processes
(write .ready/.crash) and training processes (read them).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SENTINEL_DIR = "/tmp/blendergym_services"


# ---- Server-side helpers (called by BaseService._startup) ----------------


def clear_sentinels(service_id: str) -> None:
    """Remove stale sentinel files before service startup."""
    d = Path(SENTINEL_DIR)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{service_id}.crash").unlink(missing_ok=True)
    (d / f"{service_id}.ready").unlink(missing_ok=True)


def report_ready(service_id: str, pid: int | None = None) -> None:
    """Write .ready sentinel with current PID for validation."""
    content = str(pid) if pid else "ok"
    (Path(SENTINEL_DIR) / f"{service_id}.ready").write_text(content)


def report_crash(service_id: str, tb: str) -> None:
    """Write .crash sentinel with traceback — service startup failed."""
    (Path(SENTINEL_DIR) / f"{service_id}.crash").write_text(tb)


# ---- Client-side helpers (called by env.py / rubric.py) ------------------


def ensure_service_ready(base_url: str, service_name: str, **kwargs) -> None:
    """Wait for service, auto-reading PID/stderr_log from env vars.

    Env var convention: {SERVICE_NAME}_PID, {SERVICE_NAME}_STDERR_LOG
    (matching launcher.py output).
    """
    prefix = service_name.upper()
    pid_str = os.environ.get(f"{prefix}_PID", "0")
    wait_for_service(
        base_url,
        service_name,
        pid=int(pid_str) or None,
        stderr_log=os.environ.get(f"{prefix}_STDERR_LOG"),
        **kwargs,
    )


def wait_for_service(
    base_url: str,
    service_name: str = "service",
    timeout: float = 300,
    poll_interval: float = 2,
    pid: int | None = None,
    stderr_log: str | None = None,
) -> None:
    """Block until service /health returns ok, or fail fast on crash.

    Modeled after vllm-manager's wait_for_ready() and LangChain's
    wait_for_server_healthy(): check process liveness first, then HTTP.
    """
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    crash_path = Path(SENTINEL_DIR) / f"{service_name}.crash"
    ready_path = Path(SENTINEL_DIR) / f"{service_name}.ready"

    while time.monotonic() < deadline:
        # Fast-fail: process already exited?
        if pid is not None:
            try:
                os.kill(pid, 0)
            except OSError:
                diag = _read_diagnostics(crash_path, stderr_log)
                raise RuntimeError(
                    f"{service_name} process (pid={pid}) exited{diag}"
                )

        # Fast-fail: crash sentinel?
        if crash_path.exists():
            diag = crash_path.read_text()
            raise RuntimeError(f"{service_name} startup failed:\n{diag}")

        # Check ready sentinel with PID validation to avoid stale .ready
        startup_ready = pid is None
        if ready_path.exists():
            content = ready_path.read_text().strip()
            if pid is None or content == str(pid):
                startup_ready = True

        # HTTP health check — only accept when startup_ready is confirmed
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5)
            if startup_ready and resp.json().get("status") == "ok":
                logger.info("%s ready at %s", service_name, base_url)
                return
        except Exception as e:
            last_err = e

        time.sleep(poll_interval)

    diag = _read_diagnostics(crash_path, stderr_log)
    raise RuntimeError(
        f"{service_name} at {base_url} not ready after {timeout}s: "
        f"{last_err}{diag}"
    )


# ---- Runtime diagnostics (called by clients on connection failure) -------


def diagnose_service_down(
    service_name: str,
    original_error: Exception,
) -> RuntimeError:
    """Enrich a connection error with PID liveness + stderr tail.

    Called by RenderClient/ScoreClient when HTTP request fails with
    ConnectError/ConnectTimeout after retries exhausted.
    """
    prefix = service_name.upper()
    pid_str = os.environ.get(f"{prefix}_PID", "0")
    try:
        pid = int(pid_str) or None
    except ValueError:
        pid = None
    stderr_log = os.environ.get(f"{prefix}_STDERR_LOG")
    crash_path = Path(SENTINEL_DIR) / f"{service_name}.crash"

    parts = [f"{service_name} unreachable: {original_error}"]

    if pid is not None:
        try:
            os.kill(pid, 0)
            parts.append(f"(process pid={pid} still alive — may be hung)")
        except OSError:
            parts.append(f"(process pid={pid} EXITED)")

    diag = _read_diagnostics(crash_path, stderr_log)
    if diag:
        parts.append(diag)

    return RuntimeError("\n".join(parts))


# ---- Internal ------------------------------------------------------------


def _read_diagnostics(crash_path: Path, stderr_log: str | None) -> str:
    """Read crash file and/or stderr log tail for error context."""
    parts: list[str] = []
    if crash_path.exists():
        parts.append(f"\n--- crash report ---\n{crash_path.read_text()}")
    if stderr_log and Path(stderr_log).exists():
        lines = Path(stderr_log).read_text().splitlines()[-50:]
        parts.append("\n--- stderr (last 50 lines) ---\n" + "\n".join(lines))
    return "".join(parts)
