"""Env worker lifecycle: subprocess `ZMQEnvServer` per env config.

Each `EnvWorker` represents one running env-server process and an address
for clients to connect on. When `EnvConfig.address` is None we spawn the
worker via `python -m prime_rl.orchestrator.env_server.env_server`, hand
it a generated TOML config, and pick an unused TCP port. When `address`
is set we don't spawn — we just track the external address.

The worker is a *handle*: Groups start it, await it healthy, attach a
`ZMQEnvClient`, and finally terminate it on stop.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tomli_w
import verifiers as vf
from verifiers.serve import ZMQEnvClient

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.utils.logger import get_logger


def _pick_free_port() -> int:
    """Bind a kernel-assigned port, close immediately, return the number.
    Race-prone in theory; fine in practice for one-shot subprocess spawn."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass
class EnvWorker:
    name: str
    address: str
    process: subprocess.Popen | None  # None when address was external

    async def wait_healthy(self, client: ZMQEnvClient, timeout: float = 600.0) -> None:
        """Poll the ZMQ server until it answers a health probe. Surfaces
        crashed subprocess output rather than timing out silently."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"env worker {self.name!r} subprocess exited (code={self.process.returncode}) before becoming healthy"
                )
            try:
                if await client.health(timeout=1.0):
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"env worker {self.name!r} not healthy after {timeout}s")

    def terminate(self, grace_seconds: float = 10.0) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.0)


def spawn_env_worker(env_config: EnvConfig, *, output_dir: Path, log_level: str = "info") -> EnvWorker:
    """Start (or attach to) the env-server for one env config. Returns an
    `EnvWorker` handle. Caller is responsible for calling
    `wait_healthy()` before issuing rollouts and `terminate()` on shutdown.
    """
    if env_config.address is not None:
        return EnvWorker(name=env_config.resolved_name, address=env_config.address, process=None)

    address = f"tcp://127.0.0.1:{_pick_free_port()}"
    config_payload: dict = {
        "env": {
            "id": env_config.id,
            "name": env_config.resolved_name,
            "args": env_config.args,
            "extra_env_kwargs": env_config.extra_env_kwargs,
            "num_workers": env_config.num_workers,
            "address": address,
        },
        "log": {"level": log_level},
        "output_dir": str(output_dir),
    }

    cfg_file = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".toml", prefix=f"env_server_{env_config.resolved_name}_", delete=False
    )
    tomli_w.dump(config_payload, cfg_file)
    cfg_file.close()

    cmd = [sys.executable, "-m", "prime_rl.orchestrator.env_server.env_server", "@" + cfg_file.name]
    get_logger().info(f"Spawning env worker {env_config.resolved_name!r} on {address} ({cmd!r})")
    proc = subprocess.Popen(cmd)
    return EnvWorker(name=env_config.resolved_name, address=address, process=proc)


async def attach_env_client(env: vf.Environment, address: str) -> ZMQEnvClient:
    """Build a ZMQEnvClient for `address` and attach it to `env` so
    `env.run_rollout(...)` dispatches over ZMQ instead of running locally.
    Returns the client so the caller can close it on stop."""
    client = ZMQEnvClient(address=address)
    await client.ensure_started()
    env.env_client = client
    return client
