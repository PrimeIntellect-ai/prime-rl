"""Launcher-side dashboard auto-start.

Every launcher registers its output dir in a shared registry and makes sure one
dashboard is running: every dashboard instance serves its own dirs plus the
registry (re-read live), so each new run shows up on one URL - one dashboard
per host per user. If a live daemon already
exists, its URL is logged instead of starting another. Discovery goes through
``~/.cache/prime-rl/dashboard/daemon.json`` (pid + actual url), which survives
port spillover — never probe port 7788 directly. The daemon also carries the
process title ``PRIME-RL::Dashboard``.

Stdlib-only on purpose: the launcher must work without the ``dashboard`` extra
(then it registers the dir and points at the missing extra instead of spawning).
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from prime_rl.utils.pathing import CACHE_DIR

STATE_DIR = CACHE_DIR / "dashboard"
DAEMON_FILE = STATE_DIR / "daemon.json"
DIRS_FILE = STATE_DIR / "dirs.json"
DAEMON_LOG = STATE_DIR / "daemon.log"
SPAWN_TIMEOUT_S = 10.0


def register_output_dir(output_dir: Path) -> None:
    """Add the dir to the daemon's registry (idempotent, atomic)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
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

    Returns the dashboard URL, or None when no daemon could be found or started
    (missing extra, non-interactive session, or startup failure).
    """
    register_output_dir(output_dir)
    daemon = find_daemon()
    if daemon is not None:
        logger.info(f"Dashboard running at {daemon['url']}")
        return daemon["url"]
    if not sys.stdout.isatty():
        return None  # never spawn daemons from CI or scripted launches
    binary = shutil.which("dashboard")
    if binary is None:
        logger.warning("Dashboard entry point not found - install with `uv sync --extra dashboard`")
        return None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DAEMON_LOG, "ab") as log_file:
        subprocess.Popen([binary], stdout=log_file, stderr=log_file, start_new_session=True)
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while time.monotonic() < deadline:
        daemon = find_daemon()
        if daemon is not None:
            logger.info(f"Dashboard started at {daemon['url']}")
            return daemon["url"]
        time.sleep(0.25)
    logger.warning(f"Dashboard daemon did not come up - see {DAEMON_LOG}")
    return None
