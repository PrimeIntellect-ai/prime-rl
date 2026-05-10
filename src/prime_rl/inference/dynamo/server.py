from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.vllm.server import WORKER_EXTENSION_CLS
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable


def _format_cli_value(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return str(value)


def _namespace_to_cli_args(namespace: Namespace) -> list[str]:
    args: list[str] = []
    for key, value in vars(namespace).items():
        if value is None:
            continue

        flag = f"--{key.replace('_', '-')}"
        if value is True:
            args.append(flag)
        elif value is False:
            continue
        else:
            args.extend([flag, _format_cli_value(value)])
    return args


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def _build_worker_namespace(config: InferenceConfig) -> Namespace:
    namespace = config.to_dynamo_vllm()
    namespace.worker_extension_cls = WORKER_EXTENSION_CLS[config.weight_broadcast.type]
    return namespace


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server(config: InferenceConfig) -> None:
    """Launch a local Dynamo frontend with a Dynamo vLLM worker."""
    logger = get_logger()
    disable_nccl_p2p_if_unavailable()

    frontend_namespace = config.to_dynamo_frontend()
    external_port = frontend_namespace.http_port
    frontend_namespace.http_port = _free_port()

    frontend_command = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        *_namespace_to_cli_args(frontend_namespace),
    ]
    worker_command = [
        sys.executable,
        "-m",
        "prime_rl.inference.dynamo.worker",
        *_namespace_to_cli_args(_build_worker_namespace(config)),
    ]
    proxy_command = [
        sys.executable,
        "-m",
        "prime_rl.inference.dynamo.proxy",
        "--host",
        config.server.host or "0.0.0.0",
        "--port",
        str(external_port),
        "--upstream-url",
        f"http://127.0.0.1:{frontend_namespace.http_port}",
        "--worker-url",
        f"http://127.0.0.1:{config.dynamo.system_port}",
        "--model",
        config.model.name,
    ]
    if config.model.trust_remote_code:
        proxy_command.append("--trust-remote-code")

    base_env = os.environ.copy()
    base_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if config.enable_lora:
        base_env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    discovery_dir = None
    if config.dynamo.discovery_backend == "file" and "DYN_FILE_KV" not in base_env:
        discovery_dir = tempfile.TemporaryDirectory(prefix="prime_rl_dynamo_")
        base_env["DYN_FILE_KV"] = discovery_dir.name
    if "DYN_EVENT_PLANE" not in base_env:
        if config.dynamo.event_plane is not None:
            base_env["DYN_EVENT_PLANE"] = config.dynamo.event_plane
        elif config.dynamo.discovery_backend in ("file", "mem"):
            base_env["DYN_EVENT_PLANE"] = "zmq"

    frontend_env = base_env.copy()
    frontend_env.pop("DYN_SYSTEM_PORT", None)

    worker_env = base_env.copy()
    worker_env["DYN_SYSTEM_PORT"] = str(config.dynamo.system_port)

    logger.info(f"Starting Dynamo frontend: {' '.join(frontend_command)}")
    logger.info(f"Starting Dynamo vLLM worker: {' '.join(worker_command)}")
    logger.info(f"Starting prime-rl Dynamo proxy: {' '.join(proxy_command)}")

    processes: list[subprocess.Popen] = []

    def handle_signal(signum, _frame):
        logger.warning(f"Received signal {signum}, terminating Dynamo processes")
        _terminate(processes)
        raise SystemExit(128 + signum)

    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    previous_sigint = signal.signal(signal.SIGINT, handle_signal)

    try:
        processes.append(subprocess.Popen(frontend_command, env=frontend_env))
        processes.append(subprocess.Popen(worker_command, env=worker_env))
        processes.append(subprocess.Popen(proxy_command, env=base_env))

        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    _terminate(processes)
                    raise SystemExit(return_code)
            time.sleep(1)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        _terminate(processes)
        if discovery_dir is not None:
            discovery_dir.cleanup()
