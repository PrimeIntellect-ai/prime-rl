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

from prime_rl.utils.links import PRIME_EVAL_LINKS, PRIME_LINKS, WANDB_LINKS
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import CACHE_DIR

STATE_DIR = CACHE_DIR / "dashboard"
DAEMON_FILE = STATE_DIR / "daemon.json"
DIRS_FILE = STATE_DIR / "dirs.json"
DAEMON_LOG = STATE_DIR / "daemon.log"
SPAWN_TIMEOUT_S = 10.0


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


LINK_NAMES = {PRIME_LINKS: "Prime", PRIME_EVAL_LINKS: "Prime", WANDB_LINKS: "W&B"}


def log_run_links(dashboard_url: str | None, links: Callable[[], dict[str, dict[str, str]]] | None = None) -> None:
    """Banner pointing at the dashboard and each monitor's run pages - meant as the
    launcher's last startup log. ``links`` blocks until the monitors published their
    links (the components open their pages after they start). Bound with ``console``
    so it reaches a console that only shows errors while the run is in flight."""
    logger = get_logger().bind(console=True).opt(raw=True, colors=True)
    printed = False

    def show(name: str, url: str, suffix: str) -> None:
        nonlocal printed
        lead = "" if printed else "\n"
        logger.info(f"{lead}  <b>{name:<9}</b> · <green><u>{url}</u></green>{suffix}\n")
        printed = True

    if dashboard_url is not None:
        show("Dashboard", dashboard_url, "")
    for monitor, urls in (links() if links is not None else {}).items():
        for label, url in urls.items():
            show(LINK_NAMES[monitor], url, f" ({label})" if len(urls) > 1 else "")
    if printed:
        logger.info("\n")


def run_links(
    run_dir: Path, attempt_log_dir: Path, monitors: dict[str, object | None], stop: Callable[[], bool]
) -> Callable[[], dict[str, dict[str, str]]] | None:
    """The ``links`` argument of ``log_run_links``: waits for every configured monitor
    in ``monitors`` (links-record key -> its config, None when off) that can open a
    page. A monitor turned off by its mode variable (``$PRIME_RUNS_MODE``, an offline
    or disabled ``$WANDB_MODE``) opens none. One without credentials opens none either:
    its component skips it, and the warning goes to the launcher's console here."""
    from prime_rl.utils.credentials import PRIME_MODE_VAR, missing_prime_credentials, missing_wandb_credentials
    from prime_rl.utils.links import wait_for_links

    def opens_pages(key: str, config: object) -> bool:
        if key == WANDB_LINKS:
            if getattr(config, "offline", False) or os.environ.get("WANDB_MODE") in ("offline", "disabled"):
                return False
            reason = missing_wandb_credentials()
        else:
            if os.environ.get(PRIME_MODE_VAR, "online") != "online":
                return False
            reason = missing_prime_credentials()
        if reason:
            get_logger().bind(console=True).warning(reason)
        return reason is None

    waited = [key for key, config in monitors.items() if config is not None and opens_pages(key, config)]
    if not waited:
        return None
    return functools.partial(wait_for_links, run_dir, attempt_log_dir, waited, stop)


def main() -> None:
    from prime_rl.dashboard.server import main as server_main

    server_main()
