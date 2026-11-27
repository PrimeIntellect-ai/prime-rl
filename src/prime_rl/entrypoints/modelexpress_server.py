"""Run-scoped ModelExpress server for the ``rl`` and ``sft`` launchers."""

import socket
import sysconfig
import time
from pathlib import Path
from typing import Any

from prime_rl.configs.shared import WeightBroadcastConfig
from prime_rl.utils.process import ProcessGroup
from prime_rl.utils.utils import get_free_port

MODELEXPRESS_READY_TIMEOUT_S = 120


def wait_for_port(host: str, port: int, processes: ProcessGroup, name: str, timeout: float) -> None:
    """Block until ``host:port`` accepts TCP connections; fail if ``name`` exits first."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            pass
        if processes.stop_events[name].is_set():
            raise RuntimeError(f"{name} exited before listening on {host}:{port}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{name} did not listen on {host}:{port} within {timeout}s")
        time.sleep(0.5)


def start_modelexpress(processes: ProcessGroup, port: int, log_dir: Path, base_env: dict[str, str]) -> None:
    """Start a run-scoped ModelExpress server backed by its own Valkey, both from the modelexpress-server package."""
    bin_dir = Path(sysconfig.get_path("scripts"))
    server, valkey = bin_dir / "modelexpress-server", bin_dir / "valkey-server"
    if not (server.is_file() and valkey.is_file()):
        raise FileNotFoundError(
            f"modelexpress-server and valkey-server are missing in {bin_dir}; install the `inference` extra"
        )
    state_dir = log_dir / "modelexpress"
    (state_dir / "cache").mkdir(parents=True, exist_ok=True)

    valkey_port = get_free_port()
    processes.start(
        "valkey",
        [
            str(valkey),
            "--port",
            str(valkey_port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(state_dir),
        ],
        env=base_env,
        log_path=state_dir / "valkey.log",
    )
    wait_for_port("127.0.0.1", valkey_port, processes, "valkey", MODELEXPRESS_READY_TIMEOUT_S)

    processes.start(
        "modelexpress",
        [
            str(server),
            "--port",
            str(port),
            "--metrics-port",
            "0",
            "--cache-directory",
            str(state_dir / "cache"),
        ],
        env={
            **base_env,
            "MX_METADATA_BACKEND": "redis",
            "REDIS_URL": f"redis://127.0.0.1:{valkey_port}",
            "MX_HEARTBEAT_TIMEOUT_SECS": "31536000",
        },
        log_path=state_dir / "server.log",
    )
    wait_for_port("127.0.0.1", port, processes, "modelexpress", MODELEXPRESS_READY_TIMEOUT_S)


def modelexpress_template_vars(weight_broadcast: WeightBroadcastConfig | None, launch: bool) -> dict[str, Any]:
    """Variables of the multi-node sbatch templates' ModelExpress server (``_launch_modelexpress.sh.j2``).
    ``launch`` starts a job-scoped server; otherwise the job uses the one at ``weight_broadcast.host``."""
    mx = weight_broadcast if weight_broadcast is not None and weight_broadcast.type == "modelexpress" else None
    return {
        "uses_modelexpress_server": mx is not None,
        "launch_modelexpress": mx is not None and launch,
        "modelexpress_host": mx.host if mx is not None else "",
        "modelexpress_port": mx.port if mx is not None else 0,
        "modelexpress_valkey_port": 6380 if mx is not None and mx.port == 6379 else 6379,
    }
