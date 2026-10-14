"""Register an inference engine group with a running RL job, so it joins mid-run.

Run it next to the engines (for example in an inference-only SLURM job). It waits until every
engine answers ``/health``, then writes ``<registry>/<group_id>.json``. The orchestrator of the
RL job (``orchestrator.model.client.elastic = true``) picks the group up, gives it the current
weights and routes to it. On SIGTERM or SIGINT the group drains: the record turns to
``draining``, the orchestrator stops routing to the group, waits for its running requests and
deletes the record. Then this process exits, and the engines can be stopped."""

import json
import os
import signal
import socket
import time
from pathlib import Path

import httpx
from pydantic import Field

from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import get_logger, setup_logger


class RegisterConfig(BaseConfig):
    registry: Path
    """The RL job's registry directory, ``<output_dir>/broadcasts/members``."""

    admin_urls: list[str]
    """Admin base URLs of the group's engines (``http://host:port/v1``), in rank order."""

    group_id: str = Field(default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}")
    """Unique id of this group. A relaunched group must use a new id."""

    drain_timeout: int = 900
    """Seconds to wait for the orchestrator to finish draining the group before exiting anyway."""


def write_record(path: Path, admin_urls: list[str], state: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"admin_urls": admin_urls, "state": state}))
    tmp.replace(path)


def main() -> None:
    config = cli(RegisterConfig)
    setup_logger("info")
    logger = get_logger()
    stopping = False

    def stop(*_) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for url in config.admin_urls:
        health = url.rstrip("/").removesuffix("/v1") + "/health"
        while not stopping:
            try:
                if httpx.get(health, timeout=10).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
    if stopping:
        return

    path = config.registry / f"{config.group_id}.json"
    write_record(path, config.admin_urls, "ready")
    logger.info(f"Registered engine group {config.group_id} at {path}")
    try:
        while not stopping and path.exists():
            time.sleep(1)
        if not path.exists():
            logger.info("The orchestrator dropped this engine group")
            return
        write_record(path, config.admin_urls, "draining")
        logger.info("Draining: waiting for the orchestrator to finish the running requests")
        deadline = time.monotonic() + config.drain_timeout
        while path.exists() and time.monotonic() < deadline:
            time.sleep(1)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
