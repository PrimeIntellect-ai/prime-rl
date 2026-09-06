from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from prime_rl.configs.inference import InferenceConfig

_WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "nixl": "prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker",
}

_FRONTEND_INHERITED_ENV = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "VIRTUAL_ENV",
    }
)
_FRONTEND_SHARED_DYNAMO_ENV = frozenset(
    {"DYN_DISCOVERY_BACKEND", "DYN_EVENT_PLANE", "DYN_FILE_KV", "DYN_NAMESPACE", "DYN_REQUEST_PLANE"}
)
_SECRET_ARGUMENT_SUFFIXES = ("api_key", "credentials", "password", "secret", "token")

_WORKER_ARGUMENTS_TO_SKIP = {
    "api_server_count",
    "chat_template",
    "model",
    "tool_call_parser",
}


@dataclass(frozen=True)
class DynamoProcessSpec:
    name: str
    command: tuple[str, ...]
    environment_items: tuple[tuple[str, str], ...]

    @property
    def environment(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.environment_items))


def _environment_items(environment: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(environment.items()))


def _frontend_environment(
    base_environment: Mapping[str, str],
    explicit_environment: Mapping[str, str],
) -> dict[str, str]:
    inherited = _FRONTEND_INHERITED_ENV | _FRONTEND_SHARED_DYNAMO_ENV
    return {
        **{name: value for name, value in base_environment.items() if name in inherited},
        **explicit_environment,
    }


def _vllm_argument(name: str, value: Any) -> tuple[str, ...]:
    flag = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        return (flag,) if value else (f"--no-{flag.removeprefix('--')}",)
    if isinstance(value, (dict, list)):
        return flag, json.dumps(value, separators=(",", ":"))
    return flag, str(value)


def build_dynamo_process_specs(
    config: InferenceConfig,
    *,
    executable: str = sys.executable,
) -> tuple[DynamoProcessSpec, DynamoProcessSpec]:
    if config.deployment.type != "single_node":
        raise ValueError("Managed Dynamo inference currently supports single-node deployments only.")
    if config.vllm.tensor_parallel_size != 1 or config.vllm.data_parallel_size != 1:
        raise ValueError("Managed Dynamo inference currently supports exactly one inference rank.")

    if config.weight_broadcast.type == "filesystem":
        raise ValueError("Managed Dynamo inference currently supports NCCL and NIXL weight transfer.")
    if config.vllm.enable_lora:
        raise ValueError("Managed Dynamo inference does not yet support LoRA weight updates.")
    if config.enable_return_sampling_mask or config.vllm.enable_return_routed_experts:
        raise ValueError("Managed Dynamo inference does not yet support sampling-mask or routed-expert capture.")
    if config.kv_cache_offload is not None:
        raise ValueError("Managed Dynamo inference does not yet support KV-cache offload.")
    if config.vllm.chat_template is not None:
        raise ValueError("Managed Dynamo inference does not yet support a custom chat template.")

    host = config.server.host or "0.0.0.0"
    try:
        discovery_port = int(config.env_vars.get("DYN_RL_PORT", config.server.port + 1))
        system_port = int(config.env_vars.get("DYN_SYSTEM_PORT", config.server.port + 81))
    except ValueError as error:
        raise ValueError("DYN_RL_PORT and DYN_SYSTEM_PORT must be integers.") from error
    if any(not 1 <= port <= 65535 for port in (config.server.port, discovery_port, system_port)):
        raise ValueError("Managed Dynamo ports must be between 1 and 65535.")
    if len({config.server.port, discovery_port, system_port}) != 3:
        raise ValueError("Managed Dynamo frontend, discovery, and worker system ports must be distinct.")

    configured_plugins = (name.strip() for name in config.env_vars.get("VLLM_PLUGINS", "").split(","))
    plugins = ",".join(dict.fromkeys([name for name in configured_plugins if name] + ["prime_rl"]))

    worker_arguments = [executable, "-m", "dynamo.vllm", "--enable-rl", "--model", config.vllm.model]
    namespace = vars(config.to_namespace())
    default_vllm = type(config.vllm)()
    argument_names = (config.vllm.model_fields_set | set(config.vllm.model_extra or {})) - _WORKER_ARGUMENTS_TO_SKIP
    argument_names.update({"additional_config", "hf_overrides"})
    for name in sorted(argument_names):
        if name.lower().endswith(_SECRET_ARGUMENT_SUFFIXES):
            raise ValueError(f"Managed Dynamo credential option {name!r} must be provided through the environment.")
        value = namespace.get(name)
        if value is None or value == {}:
            continue
        if (
            name not in {"additional_config", "hf_overrides"}
            and name in type(config.vllm).model_fields
            and value == getattr(default_vllm, name)
        ):
            continue
        worker_arguments.extend(_vllm_argument(name, value))

    worker_arguments.extend(("--worker-extension-cls", _WORKER_EXTENSION_CLS[config.weight_broadcast.type]))
    frontend = DynamoProcessSpec(
        name="frontend",
        command=(executable, "-m", "dynamo.frontend"),
        environment_items=_environment_items(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "DYN_ENABLE_RL": "true",
                "DYN_HTTP_HOST": host,
                "DYN_HTTP_PORT": str(config.server.port),
                "DYN_RL_PORT": str(discovery_port),
                "DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE": "1",
            }
        ),
    )
    worker = DynamoProcessSpec(
        name="worker",
        command=tuple(worker_arguments),
        environment_items=_environment_items(
            {
                "DYN_ENABLE_RL": "true",
                "DYN_SYSTEM_HOST": "127.0.0.1",
                "DYN_SYSTEM_PORT": str(system_port),
                "DYN_RL_INIT_WEIGHTS_TIMEOUT_S": config.env_vars.get("DYN_RL_INIT_WEIGHTS_TIMEOUT_S", "1200"),
                "VLLM_PLUGINS": plugins,
            }
        ),
    )
    return frontend, worker


def _signal_process_group(pid: int, requested_signal: int) -> bool:
    try:
        os.killpg(pid, requested_signal)
    except ProcessLookupError:
        return False
    return True


def _terminate(process: subprocess.Popen) -> None:
    if not _signal_process_group(process.pid, signal.SIGTERM):
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _signal_process_group(process.pid, signal.SIGKILL)
        process.wait()
    else:
        if _signal_process_group(process.pid, 0):
            _signal_process_group(process.pid, signal.SIGKILL)


def run_dynamo_local(config: InferenceConfig) -> None:
    base_environment = {**os.environ, **config.env_vars}
    base_environment["DYN_DISCOVERY_BACKEND"] = "file"
    base_environment["DYN_REQUEST_PLANE"] = "tcp"
    base_environment["DYN_EVENT_PLANE"] = "zmq"
    base_environment["DYN_NAMESPACE"] = f"prime-rl-{os.environ.get('PRL_RUN_ID', os.getpid())}"

    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    processes: list[subprocess.Popen] = []
    try:
        with tempfile.TemporaryDirectory(prefix="prime-dynamo-") as temporary_dir:
            base_environment["DYN_FILE_KV"] = temporary_dir
            specs = build_dynamo_process_specs(config)
            for spec in specs:
                environment = (
                    _frontend_environment(base_environment, spec.environment)
                    if spec.name == "frontend"
                    else {**base_environment, **spec.environment}
                )
                processes.append(
                    subprocess.Popen(
                        list(spec.command),
                        env=environment,
                        start_new_session=True,
                    )
                )

            while True:
                for spec, process in zip(specs, processes):
                    if (returncode := process.poll()) is not None:
                        raise RuntimeError(f"Dynamo {spec.name} exited with code {returncode or 1}")
                time.sleep(0.2)
    except KeyboardInterrupt:
        return
    finally:
        for process in reversed(processes):
            _terminate(process)
        signal.signal(signal.SIGTERM, previous_sigterm)
