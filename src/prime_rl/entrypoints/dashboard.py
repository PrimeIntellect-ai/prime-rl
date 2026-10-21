"""Launcher-side dashboard auto-start.

Every launcher registers its output dir in a shared registry and makes sure one
dashboard is running: every dashboard instance serves its own dirs plus the
registry (re-read live), so each new run shows up on one URL - one dashboard
per host per user. If a live daemon already exists, its URL is reused instead
of starting another. Discovery goes through
``~/.cache/prime-rl/dashboard/daemon.json`` (pid + actual url), which survives
port spillover. The daemon also carries the process title ``PRL::Dashboard``.

Kept free of ``dashboard``-extra imports on purpose: the launcher must work
without the extra (then it registers the dir and points at the missing extra
instead of spawning). ``main`` is the ``dashboard`` console script, delegating
to the actual server.
"""

import fcntl
import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import CACHE_DIR, get_platform_run_path

STATE_DIR = CACHE_DIR / "dashboard"
DAEMON_FILE = STATE_DIR / "daemon.json"
DIRS_FILE = STATE_DIR / "dirs.json"
DAEMON_LOG = STATE_DIR / "daemon.log"
SPAWN_TIMEOUT_S = 10.0
# The components register the run on the platform once they are up, which takes
# seconds to a few minutes (imports, model and tokenizer loads).
PLATFORM_LINK_TIMEOUT_S = 300.0
PLATFORM_LINK_POLL_S = 0.5


@contextmanager
def registry_lock():
    """Serialize read-modify-write cycles on the shared registry files: concurrent
    launchers (or dashboard starts) must not drop each other's entries."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def register_output_dir(output_dir: Path) -> None:
    """Add the dir to the daemon's registry (idempotent, atomic)."""
    with registry_lock():
        try:
            dirs = json.loads(DIRS_FILE.read_text())
        except (OSError, ValueError):
            dirs = []
        entry = str(output_dir.resolve())
        if entry in dirs:
            return
        dirs.append(entry)
        tmp = DIRS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(dirs))
        tmp.replace(DIRS_FILE)


def find_daemon(timeout: float = 1.0) -> dict | None:
    """The live daemon's record ({pid, url, ...}), or None."""
    try:
        info = json.loads(DAEMON_FILE.read_text())
        os.kill(info["pid"], 0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        with urllib.request.urlopen(f"{info['url']}/api/runs", timeout=timeout):
            return info
    except OSError:
        return None


def ensure_dashboard(output_dir: Path, logger) -> str | None:
    """Register the run's output dir and make sure one dashboard daemon serves it.

    Returns the dashboard URL (log it with ``log_run_links``), or None when no
    daemon could be found or started (missing extra, non-interactive session, or
    startup failure).
    """
    # The dir may not exist yet on a first run: create it up front, because the
    # daemon only serves registered dirs that exist (and refuses to start with none).
    output_dir.mkdir(parents=True, exist_ok=True)
    register_output_dir(output_dir)
    daemon = find_daemon()
    if daemon is not None:
        return daemon["url"]
    if not sys.stdout.isatty():
        logger.warning("Non-interactive session - not auto-starting a dashboard (the output dir stays registered)")
        return None
    # The console script exists even without the extra, so check the actual imports.
    if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None:
        logger.warning("Dashboard extra not installed - install with `uv sync --extra dashboard`")
        return None
    binary = shutil.which("dashboard")
    if binary is None:
        logger.warning("Dashboard entry point not found - install with `uv sync --extra dashboard`")
        return None
    with open(DAEMON_LOG, "ab") as log_file:
        process = subprocess.Popen([binary], stdout=log_file, stderr=log_file, start_new_session=True)
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while time.monotonic() < deadline:
        daemon = find_daemon()
        if daemon is not None:
            return daemon["url"]
        if process.poll() is not None:
            break
        time.sleep(0.25)
    logger.warning(f"Dashboard daemon did not come up - see {DAEMON_LOG}")
    return None


def log_run_links(dashboard_url: str | None, platform: Callable[[], list[str]] | None = None) -> None:
    """Banner pointing at the dashboard and the run's platform pages - meant as the
    launcher's last startup log. ``platform`` blocks until the platform links are known
    (the components register the run after they start). Bound with ``console`` so it
    reaches a console that only shows errors while the run is in flight."""
    logger = get_logger().bind(console=True).opt(raw=True, colors=True)
    if dashboard_url is not None:
        logger.info(f"\n  <b>Dashboard</b> · <green><u>{dashboard_url}</u></green>\n")
    links = platform() if platform is not None else []
    for i, url in enumerate(links):
        lead = "\n" if dashboard_url is None and i == 0 else ""
        logger.info(f"{lead}  <b>Platform</b>  · <green><u>{url}</u></green>\n")
    if dashboard_url is not None or links:
        logger.info("\n")


def platform_links(
    run_dir: Path, attempt_log_dir: Path, expected: int, stop: Callable[[], bool]
) -> Callable[[], list[str]] | None:
    """The ``platform`` argument of ``log_run_links`` for a run whose prime monitor is
    on, or None when the monitor opens no platform pages (no API key, or
    ``$PRIME_RUNS_MODE=disabled``)."""
    from prime_rl.monitors.prime import platform_mode

    if platform_mode() != "online":
        return None
    return functools.partial(wait_for_platform_links, run_dir, attempt_log_dir, expected, stop)


def wait_for_platform_links(
    output_dir: Path,
    attempt_log_dir: Path,
    expected: int,
    stop: Callable[[], bool],
    timeout: float = PLATFORM_LINK_TIMEOUT_S,
) -> list[str]:
    """The platform links this attempt's prime monitors leave in the run's platform
    record: the training run's, or one per eval source. Polls until ``expected`` links
    are in, ``stop()`` turns true or ``timeout`` passes, and returns what is in by then.
    Links stamped with another attempt are a previous launch's and never count."""
    path = get_platform_run_path(output_dir)
    attempt = str(attempt_log_dir)
    deadline = time.monotonic() + timeout
    links: list[str] = []
    while True:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            record = {}
        entries = [record, *(record.get("evaluations") or {}).values()]
        links = [entry["url"] for entry in entries if entry.get("url") and entry.get("attempt") == attempt]
        if len(links) >= expected or stop() or time.monotonic() >= deadline:
            return links
        time.sleep(PLATFORM_LINK_POLL_S)


def main() -> None:
    from prime_rl.dashboard.server import main as server_main

    server_main()
