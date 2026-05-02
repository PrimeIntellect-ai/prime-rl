"""Session-scoped Model Express server brought up via docker-compose.

Skips the test session entirely if Docker isn't on the PATH so transport
tests don't false-fail in environments without Docker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

COMPOSE_FILE = Path(__file__).resolve().parents[3] / "docker" / "modelexpress" / "docker-compose.yml"


@pytest.fixture(scope="session")
def mx_server() -> str:
    """Bring up the ME stack (server + redis), tear it down at session end. Yields ``host:port``."""
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")
    if not COMPOSE_FILE.is_file():
        pytest.skip(f"compose file not found at {COMPOSE_FILE}")

    project = f"prime-rl-mx-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    up = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", project, "up", "-d", "--build", "--wait"],
        capture_output=True,
        text=True,
    )
    if up.returncode != 0:
        pytest.skip(f"docker compose up failed:\n{up.stderr}")

    try:
        yield "localhost:29501"
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", project, "down", "--remove-orphans"],
            capture_output=True,
            text=True,
        )
