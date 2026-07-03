"""Translate Prime inference config into Dynamo worker processes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from prime_rl.configs.inference import DisaggregatedInferenceDeploymentConfig, InferenceConfig
from prime_rl.utils.pathing import get_config_dir

Role = Literal["agg", "prefill", "decode"]

ENGINE_CONFIG_DIR = "dynamo"
PREFILL_ENGINE_CONFIG = "prefill-engine.json"
DECODE_ENGINE_CONFIG = "decode-engine.json"
AGG_ENGINE_CONFIG = "agg-engine.json"

_ENGINE_CONFIG_EXCLUDED = frozenset(
    {
        "api_server_count",
        "chat_template",
        "enable_auto_tool_choice",
        "host",
        "liveness_timeout_seconds",
        "port",
        "reasoning_parser",
        "tool_call_parser",
    }
)
_RESERVED_ENGINE_KEYS = frozenset(
    {
        "disaggregation_mode",
        "enable_rl",
        "kv_events_config",
        "kv_transfer_config",
        "worker_extension_cls",
    }
)
_WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
}


@dataclass(frozen=True)
class DynamoWorkerSpec:
    name: str
    role: Role
    gpu_ids: tuple[str, ...]
    system_port: int
    nixl_port: int
    kv_events_port: int | None
    engine_config: Path

    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "dynamo.vllm",
            "--engine-config-json",
            str(self.engine_config),
            "--disaggregation-mode",
            self.role,
            "--enable-rl",
        ]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _role_overrides(config: InferenceConfig, role: Role) -> dict[str, Any]:
    if config.deployment.type != "disaggregated":
        return {}
    if role == "prefill":
        return config.deployment.prefill_vllm_overrides
    if role == "decode":
        return config.deployment.decode_vllm_overrides
    return {}


def _validate_overrides(source: str, values: dict[str, Any]) -> None:
    conflicts = sorted(_RESERVED_ENGINE_KEYS & values.keys())
    if conflicts:
        raise ValueError(f"{source} cannot override Dynamo-managed engine keys: {conflicts}")


def build_engine_config(
    config: InferenceConfig,
    role: Role,
    *,
    kv_events_port: int | None = None,
) -> dict[str, Any]:
    """Build one deterministic vLLM ``AsyncEngineArgs`` object."""
    _validate_overrides("vllm_extra", config.vllm_extra)
    overrides = _role_overrides(config, role)
    _validate_overrides(f"{role}_vllm_overrides", overrides)

    values = vars(config.to_vllm()).copy()
    for key in _ENGINE_CONFIG_EXCLUDED:
        values.pop(key, None)
    values.update(config.vllm_extra)
    values.update(overrides)

    if config.deployment.type == "disaggregated":
        # Each generated worker is an independent vLLM server. Preserve local
        # DP within a worker, but never turn the P/D worker count into vLLM DP.
        local_dp = config.deployment.gpus_per_node // config.parallel.tp
        values["data_parallel_size"] = local_dp
        if local_dp == 1:
            values.pop("data_parallel_size_local", None)
            values.pop("data_parallel_rpc_port", None)
        else:
            values["data_parallel_size_local"] = local_dp

    if role in ("prefill", "agg") and kv_events_port is not None:
        values["kv_events_config"] = {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{kv_events_port}",
            "enable_kv_cache_events": True,
        }
    else:
        values.pop("kv_events_config", None)

    values["worker_extension_cls"] = _WORKER_EXTENSION_CLS[config.weight_broadcast.type]
    return {key: value for key, value in values.items() if value is not None}


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n")
    return path


def write_role_engine_configs(config: InferenceConfig, output_dir: Path | None = None) -> dict[Role, Path]:
    """Write canonical role configs used by DGD and dry-run inspection."""
    config_dir = output_dir or (get_config_dir(config.output_dir) / ENGINE_CONFIG_DIR)
    if config.deployment.type == "disaggregated":
        return {
            "prefill": _write_json(
                config_dir / PREFILL_ENGINE_CONFIG,
                build_engine_config(config, "prefill", kv_events_port=20080),
            ),
            "decode": _write_json(config_dir / DECODE_ENGINE_CONFIG, build_engine_config(config, "decode")),
        }
    return {
        "agg": _write_json(
            config_dir / AGG_ENGINE_CONFIG,
            build_engine_config(config, "agg", kv_events_port=20080),
        )
    }


def _visible_gpu_ids() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured:
        return [gpu.strip() for gpu in configured.split(",") if gpu.strip()]
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot discover GPUs; set CUDA_VISIBLE_DEVICES explicitly") from exc
    return [line.strip() for line in output.splitlines() if line.strip()]


def build_local_worker_specs(
    config: InferenceConfig,
    output_dir: Path | None = None,
    gpu_ids: list[str] | None = None,
) -> list[DynamoWorkerSpec]:
    """Allocate local workers and write instance-specific engine configs."""
    config_dir = output_dir or (get_config_dir(config.output_dir) / ENGINE_CONFIG_DIR)
    available = gpu_ids or _visible_gpu_ids()

    if config.deployment.type == "disaggregated":
        deployment: DisaggregatedInferenceDeploymentConfig = config.deployment
        if deployment.num_prefill_nodes != deployment.num_prefill_replicas:
            raise ValueError("Local Dynamo requires one prefill node per prefill replica")
        if deployment.num_decode_nodes != deployment.num_decode_replicas:
            raise ValueError("Local Dynamo requires one decode node per decode replica")
        roles: list[Role] = ["decode"] * deployment.num_decode_replicas + ["prefill"] * deployment.num_prefill_replicas
        gpus_per_worker = deployment.gpus_per_node
    else:
        roles = ["agg"]
        gpus_per_worker = config.parallel.tp * config.parallel.dp

    required = len(roles) * gpus_per_worker
    if len(available) < required:
        raise ValueError(f"Dynamo topology requires {required} GPUs, but only {len(available)} are visible")

    specs: list[DynamoWorkerSpec] = []
    role_indexes: dict[Role, int] = {"agg": 0, "prefill": 0, "decode": 0}
    for worker_index, role in enumerate(roles):
        role_index = role_indexes[role]
        role_indexes[role] += 1
        start = worker_index * gpus_per_worker
        worker_gpus = tuple(available[start : start + gpus_per_worker])
        kv_events_port = 20080 + role_index if role in ("prefill", "agg") else None
        name = f"{role}-{role_index}"
        engine_path = _write_json(
            config_dir / f"{name}-engine.json",
            build_engine_config(config, role, kv_events_port=kv_events_port),
        )
        specs.append(
            DynamoWorkerSpec(
                name=name,
                role=role,
                gpu_ids=worker_gpus,
                system_port=8081 + worker_index,
                nixl_port=20100 + worker_index,
                kv_events_port=kv_events_port,
                engine_config=engine_path,
            )
        )
    return specs


def build_frontend_command(config: InferenceConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--http-host",
        config.server.host or "0.0.0.0",
        "--http-port",
        str(config.server.port),
        "--router-mode",
        "kv",
        "--router-reset-states",
        "--enable-engine-apis",
    ]


def build_worker_environment(
    config: InferenceConfig,
    spec: DynamoWorkerSpec,
    base_environment: dict[str, str],
) -> dict[str, str]:
    environment = base_environment | {
        "CUDA_VISIBLE_DEVICES": ",".join(spec.gpu_ids),
        "DYN_SYSTEM_PORT": str(spec.system_port),
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(spec.nixl_port),
        "VLLM_PLUGINS": "prime_rl",
    }
    if config.deployment.type == "disaggregated":
        role_environment = (
            config.deployment.prefill_env_vars if spec.role == "prefill" else config.deployment.decode_env_vars
        )
        environment.update(role_environment)
    return environment


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_dynamo_local(config: InferenceConfig) -> None:
    """Run a Dynamo frontend and all configured workers until one exits."""
    specs = build_local_worker_specs(config)
    environment = os.environ.copy()
    environment.setdefault("DYN_DISCOVERY_BACKEND", "file")
    environment.setdefault("DYN_EVENT_PLANE", "zmq")
    environment.setdefault("DYN_FILE_KV_TTL_SECS", "1800")
    environment.setdefault("DYN_NAMESPACE", f"prime-rl-{os.getpid()}")
    environment.setdefault("DYN_ENABLE_RL", "1")
    environment.setdefault("DYN_RL_PORT", "8001")
    environment.setdefault("PYTHONHASHSEED", "0")

    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_stop)
    processes: list[subprocess.Popen] = []
    with tempfile.TemporaryDirectory(prefix="prime-dynamo-") as temporary_dir:
        environment.setdefault("DYN_FILE_KV", str(Path(temporary_dir) / "discovery"))
        frontend_env = environment | {"CUDA_VISIBLE_DEVICES": ""}
        frontend_env.pop("DYN_SYSTEM_PORT", None)

        try:
            processes.append(subprocess.Popen(build_frontend_command(config), env=frontend_env, start_new_session=True))
            for spec in specs:
                worker_env = build_worker_environment(config, spec, environment)
                processes.append(subprocess.Popen(spec.command(), env=worker_env, start_new_session=True))

            while all(process.poll() is None for process in processes):
                time.sleep(0.2)
            raise SystemExit(next((process.returncode for process in processes if process.returncode), 1))
        except KeyboardInterrupt:
            return
        finally:
            for process in reversed(processes):
                _terminate(process)
