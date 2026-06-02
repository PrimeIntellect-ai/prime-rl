"""Node-local Mooncake store launcher for KV cache offload (standalone-store topology).

Each node runs its own ``mooncake_master`` + one ``mooncake_client``; the node's vLLM
workers connect to that local store as clients (contributing no segment of their own). The
client owns the pool: a DRAM segment sized by the cpu tier, plus a file-backed disk tier
when a disk tier is configured. See ``MooncakeKVCacheOffloadConfig``.

This module both (a) is imported by the local launch path in ``inference/vllm/server.py``
and (b) exposes a small CLI used by the SLURM templates to bring the store up per node.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from prime_rl.configs.inference import MooncakeKVCacheOffloadConfig

DEFAULT_MASTER_RPC_PORT = 50051
DEFAULT_METADATA_HTTP_PORT = 8080
DEFAULT_CLIENT_PORT = 50052
# Per-worker transfer staging buffer in the store client (mirrors Mooncake's default).
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024**3
# Mooncake transfer protocol (RDMA only).
MOONCAKE_PROTOCOL = "rdma"


def _bin(name: str) -> str:
    """Resolve a Mooncake console-script, preferring the one in this interpreter's venv."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    return shutil.which(name) or name


def _local_ip() -> str:
    # Match the address the vLLM worker registers with the store (it uses vllm's get_ip()).
    from vllm.utils.network_utils import get_ip

    return get_ip()


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    """Block until ``host:port`` accepts a TCP connection, or raise on timeout."""
    deadline = time.monotonic() + timeout
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as e:
            last_err = e
            time.sleep(0.2)
    raise TimeoutError(f"Mooncake endpoint {host}:{port} did not come up within {timeout}s: {last_err}")


@dataclass
class MooncakeStore:
    """Handles for a running node-local Mooncake store and the env it needs.

    With one client per node there is exactly one segment in the store, so the master
    routes all puts/gets to it — no ``preferred_segment`` pinning is needed (that hint only
    matters when several client segments coexist in one pool).
    """

    master: subprocess.Popen
    client: subprocess.Popen
    config_path: Path
    env: dict[str, str]

    def apply_env(self) -> None:
        """Export the env vars the vLLM worker process needs (config path, segment, hash seed)."""
        os.environ.update(self.env)

    def shutdown(self) -> None:
        for proc in (self.client, self.master):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _resolve_addresses(cfg: MooncakeKVCacheOffloadConfig) -> tuple[str, str, str, int]:
    """Return (master_server_address, metadata_server, client_host, client_port).

    The metadata server is always the HTTP server hosted by the master (auto-detected to the
    master host); the node-local master hosts it unless an external master is configured.
    """
    ip = _local_ip()
    master_addr = cfg.master_server_address or f"{ip}:{DEFAULT_MASTER_RPC_PORT}"
    master_host = master_addr.split(":")[0]
    metadata = f"http://{master_host}:{DEFAULT_METADATA_HTTP_PORT}/metadata"
    return master_addr, metadata, ip, DEFAULT_CLIENT_PORT


def write_config_file(cfg: MooncakeKVCacheOffloadConfig, master_addr: str, metadata: str, path: Path) -> None:
    """Write the ``MOONCAKE_CONFIG_PATH`` JSON read per-worker by ``MooncakeStoreConfig.from_file``.

    Standalone-store: workers contribute no segment (``global_segment_size == 0``); the
    node-local client owns the pool. ``enable_offload`` is on when a disk tier exists.
    """
    config = {
        "mode": "standalone-store",
        "global_segment_size": 0,
        "local_buffer_size": DEFAULT_LOCAL_BUFFER_SIZE,
        "protocol": MOONCAKE_PROTOCOL,
        "device_name": cfg.device_name,
        "master_server_address": master_addr,
        "metadata_server": metadata,
        "enable_offload": cfg.disk is not None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


def _offload_env(cfg: MooncakeKVCacheOffloadConfig) -> dict[str, str]:
    """Offload (disk-tier) env shared by the master and client when a disk tier is set.

    The master serves the fsdir to the client (the client does ``get_fsdir`` from it), and
    the client writes the offloaded block files there — so both processes need the path.
    """
    if cfg.disk is None:
        return {}
    cfg.disk.path.mkdir(parents=True, exist_ok=True)
    return {
        "MOONCAKE_OFFLOAD_FSDIR": str(cfg.disk.path),
        "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": str(cfg.disk.path),
        "MOONCAKE_OFFLOAD_ENABLE_EVICTION": "true",
    }


def _launch_master(cfg: MooncakeKVCacheOffloadConfig, ip: str, log_dir: Path) -> subprocess.Popen:
    args = [
        _bin("mooncake_master"),
        f"-rpc_port={DEFAULT_MASTER_RPC_PORT}",
        "-enable_http_metadata_server",
        f"-http_metadata_server_host={ip}",
        f"-http_metadata_server_port={DEFAULT_METADATA_HTTP_PORT}",
    ]
    if cfg.disk is not None:
        # The master owns the file-storage backend config the client mounts against.
        args += ["-enable_offload=true", f"-root_fs_dir={cfg.disk.path}"]
    log = (log_dir / "mooncake_master.log").open("w")
    return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **_offload_env(cfg)})


def _launch_client(
    cfg: MooncakeKVCacheOffloadConfig, ip: str, master_addr: str, metadata: str, log_dir: Path
) -> subprocess.Popen:
    assert cfg.cpu is not None  # enforced by config validation
    args = [
        _bin("mooncake_client"),
        f"-host={ip}",
        f"-port={DEFAULT_CLIENT_PORT}",
        f"-master_server_address={master_addr}",
        f"-metadata_server={metadata}",
        f"-protocol={MOONCAKE_PROTOCOL}",
        f"-global_segment_size={int(cfg.cpu.num_bytes)}",
    ]
    if cfg.device_name:
        args.append(f"-device_names={cfg.device_name}")
    if cfg.disk is not None:
        args.append("-enable_offload=true")
    log = (log_dir / "mooncake_client.log").open("w")
    return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **_offload_env(cfg)})


def start_mooncake_store(cfg: MooncakeKVCacheOffloadConfig, output_dir: Path) -> MooncakeStore:
    """Launch the node-local master + client, write the config JSON, and return handles + env.

    The caller must export the returned env (``apply_env``) before the vLLM engine process
    starts (``PYTHONHASHSEED`` must precede the interpreter that hashes KV blocks).
    """
    log_dir = output_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    master_addr, metadata, client_ip, client_port = _resolve_addresses(cfg)

    master = None
    if cfg.master_server_address is None:
        master = _launch_master(cfg, client_ip, log_dir)
        master_host, master_port = master_addr.split(":")
        _wait_for_port(master_host, int(master_port))
        _wait_for_port(client_ip, DEFAULT_METADATA_HTTP_PORT)

    client = _launch_client(cfg, client_ip, master_addr, metadata, log_dir)
    _wait_for_port(client_ip, client_port)

    config_path = output_dir / "mooncake_config.json"
    write_config_file(cfg, master_addr, metadata, config_path)

    env = {
        "MOONCAKE_CONFIG_PATH": str(config_path),
        # Reproducible KV block hashes across the store client and vLLM worker processes.
        "PYTHONHASHSEED": "0",
    }
    # If no separate master was launched (external master), there is no master Popen; use the
    # client handle as a stand-in so shutdown() stays simple.
    return MooncakeStore(
        master=master if master is not None else client,
        client=client,
        config_path=config_path,
        env=env,
    )


def _main() -> None:
    """CLI used by the SLURM templates: launch the node-local store and print env exports.

    Usage: ``python -m prime_rl.inference.mooncake --config <inference.toml> --output-dir <dir>``.
    Launches the node-local master + client (which outlive this process) and prints shell
    ``export`` lines for the template to ``eval`` before starting vLLM on that node.
    """
    import argparse
    import tomllib

    from pydantic import TypeAdapter

    from prime_rl.configs.inference import KVCacheOffloadConfig

    parser = argparse.ArgumentParser(description="Launch a node-local Mooncake store.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the inference TOML.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the store config + logs.")
    args = parser.parse_args()

    raw = tomllib.loads(args.config.read_text())
    offload_raw = raw.get("kv_cache_offload")
    if offload_raw is None:
        raise ValueError("inference config has no kv_cache_offload section")
    offload = TypeAdapter(KVCacheOffloadConfig).validate_python(offload_raw)
    if not isinstance(offload, MooncakeKVCacheOffloadConfig):
        raise ValueError("mooncake CLI requires inference.kv_cache_offload.type == 'mooncake'")

    store = start_mooncake_store(offload, args.output_dir)
    # Write the env as a sourceable file so the SLURM template can `source` it without
    # capturing stdout (which may carry unrelated import logging).
    env_sh = args.output_dir / "env.sh"
    env_sh.write_text("".join(f"export {key}={value}\n" for key, value in store.env.items()))
    print(f"wrote {env_sh}")


if __name__ == "__main__":
    _main()
