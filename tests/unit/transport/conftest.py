"""Session-scoped Model Express server brought up via docker-compose.

Skips the test session entirely if Docker isn't on the PATH so transport
tests don't false-fail in environments without Docker.

Uses a fixed compose project name (``prime-rl-mx-test``) so a previous
session that crashed before teardown doesn't leak port 29501 — ``up -d
--wait`` is idempotent against an already-healthy stack.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

COMPOSE_FILE = Path(__file__).resolve().parents[3] / "docker" / "modelexpress" / "docker-compose.yml"
COMPOSE_PROJECT = "prime-rl-mx-test"


@pytest.fixture(scope="session")
def mx_server() -> str:
    """Bring up the ME stack (server + redis), tear it down at session end. Yields ``host:port``."""
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")
    if not COMPOSE_FILE.is_file():
        pytest.skip(f"compose file not found at {COMPOSE_FILE}")

    up = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", COMPOSE_PROJECT, "up", "-d", "--build", "--wait"],
        capture_output=True,
        text=True,
    )
    if up.returncode != 0:
        pytest.skip(f"docker compose up failed:\n{up.stderr}")

    try:
        yield "localhost:29501"
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", COMPOSE_PROJECT, "down", "--remove-orphans"],
            capture_output=True,
            text=True,
        )
