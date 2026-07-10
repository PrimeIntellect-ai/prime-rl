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
CHAT_TEMPLATE_ASSET = "chat-template.jinja"

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
        "data_parallel_rpc_port",
        "data_parallel_size",
        "data_parallel_size_local",
        "disaggregation_mode",
        "enable_prefix_caching",
        "enable_rl",
        "kv_events_config",
        "kv_transfer_config",
        "pipeline_parallel_size",
        "tensor_parallel_size",
        "worker_extension_cls",
    }
)
_WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
}
_WORKER_COMPONENT = {
    "agg": "backend",
    "prefill": "prefill",
    "decode": "backend",
}


@dataclass(frozen=True)
class DynamoProcessSpec:
    module: str
    arguments: tuple[str, ...]
    environment_items: tuple[tuple[str, str], ...]

    def command(self, executable: str = sys.executable) -> list[str]:
        return [executable, "-m", self.module, *self.arguments]

    def environment(self, base: dict[str, str] | None = None) -> dict[str, str]:
        return (base or {}) | dict(self.environment_items)


@dataclass(frozen=True)
class DynamoWorkerSpec:
    name: str
    role: Role
    gpu_ids: tuple[str, ...]
    system_port: int
    process: DynamoProcessSpec


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
    wrapper_only = sorted(_ENGINE_CONFIG_EXCLUDED & values.keys())
    if wrapper_only:
        raise ValueError(f"{source} keys {wrapper_only} are wrapper/server-only and cannot enter a vLLM engine config")


def _environment_items(values: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(values.items()))


def _role_environment(config: InferenceConfig, role: Role) -> dict[str, str]:
    if config.deployment.type != "disaggregated":
        return {}
    if role == "prefill":
        return config.deployment.prefill_env_vars
    if role == "decode":
        return config.deployment.decode_env_vars
    return {}


def resolve_chat_template_content(config: InferenceConfig) -> str | None:
    """Resolve a configured inline or file-backed chat template to immutable content."""
    template = config.model.chat_template
    if template is None:
        return None
    template_source = Path(os.path.expanduser(template))
    return template_source.read_text(encoding="utf-8") if template_source.is_file() else template


def _materialize_chat_template(
    config: InferenceConfig,
    template_content: str,
    output_dir: Path | None,
) -> Path:
    config_dir = output_dir or (get_config_dir(config.output_dir) / ENGINE_CONFIG_DIR)
    template_path = config_dir / CHAT_TEMPLATE_ASSET
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(template_content, encoding="utf-8")
    return template_path


def _frontend_model_arguments(
    config: InferenceConfig,
    output_dir: Path | None,
    runtime_chat_template_path: Path | None,
) -> tuple[str, ...]:
    template_content = resolve_chat_template_content(config)
    if template_content is None:
        return ()
    template_path = runtime_chat_template_path or _materialize_chat_template(config, template_content, output_dir)
    tool_arguments = (
        ("--enable-auto-tool-choice", "--tool-call-parser", config.model.tool_call_parser)
        if config.model.tool_call_parser is not None
        else ()
    )
    reasoning_arguments = (
        ("--reasoning-parser", config.model.reasoning_parser) if config.model.reasoning_parser is not None else ()
    )
    return (
        *tool_arguments,
        *reasoning_arguments,
        "--dyn-chat-processor",
        "vllm",
        "--chat-template",
        str(template_path),
    )


def build_frontend_process(
    config: InferenceConfig,
    *,
    host: str | None = None,
    port: int | None = None,
    output_dir: Path | None = None,
    runtime_chat_template_path: Path | None = None,
) -> DynamoProcessSpec:
    """Build the canonical Dynamo frontend process contract."""
    environment = {
        **config.env_vars,
        "DYN_ENABLE_RL": "1",
        "DYN_RL_PORT": "8001",
    }
    arguments = (
        "--http-host",
        host or config.server.host or "0.0.0.0",
        "--http-port",
        str(port or config.server.port),
        "--router-mode",
        "kv",
        "--router-reset-states",
        "--enable-engine-apis",
        *_frontend_model_arguments(config, output_dir, runtime_chat_template_path),
    )
    return DynamoProcessSpec(
        module="dynamo.frontend",
        arguments=arguments,
        environment_items=_environment_items(environment),
    )


def _worker_parser_arguments(config: InferenceConfig, role: Role) -> tuple[str, ...]:
    if role == "prefill":
        return ()
    tool_arguments = (
        ("--dyn-tool-call-parser", config.model.tool_call_parser) if config.model.tool_call_parser is not None else ()
    )
    reasoning_arguments = (
        ("--dyn-reasoning-parser", config.model.reasoning_parser) if config.model.reasoning_parser is not None else ()
    )
    return (*tool_arguments, *reasoning_arguments)


def _worker_endpoint_contract(namespace: str | None, component: str) -> tuple[dict[str, str], tuple[str, ...]]:
    if namespace is None:
        return {}, ()
    endpoint = f"dyn://{namespace}.{component}.generate"
    return (
        {"DYN_NAMESPACE": namespace, "DYN_ENDPOINT": endpoint},
        ("--endpoint", endpoint),
    )


def build_worker_process(
    config: InferenceConfig,
    role: Role,
    engine_config: Path,
    *,
    nixl_host: str | None,
    nixl_port: int,
    namespace: str | None = None,
) -> DynamoProcessSpec:
    """Build the canonical Dynamo vLLM worker process contract."""
    resolved_namespace = namespace or config.env_vars.get("DYN_NAMESPACE")
    component = _WORKER_COMPONENT[role]
    endpoint_environment, endpoint_arguments = _worker_endpoint_contract(resolved_namespace, component)
    environment = {
        **config.env_vars,
        **_role_environment(config, role),
        "DYN_ENABLE_RL": "1",
        "DYN_COMPONENT": component,
        **endpoint_environment,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port),
        **({"VLLM_NIXL_SIDE_CHANNEL_HOST": nixl_host} if nixl_host is not None else {}),
        "VLLM_PLUGINS": "prime_rl",
    }
    arguments = (
        "--engine-config-json",
        str(engine_config),
        *endpoint_arguments,
        "--disaggregation-mode",
        role,
        "--enable-rl",
        *_worker_parser_arguments(config, role),
    )
    return DynamoProcessSpec(
        module="dynamo.vllm",
        arguments=arguments,
        environment_items=_environment_items(environment),
    )


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
        local_dp = config.dynamo_local_dp
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
    namespace: str | None = None,
) -> list[DynamoWorkerSpec]:
    """Allocate local workers and write instance-specific engine configs."""
    config_dir = output_dir or (get_config_dir(config.output_dir) / ENGINE_CONFIG_DIR)
    available = gpu_ids if gpu_ids is not None else _visible_gpu_ids()
    resolved_namespace = namespace or config.env_vars.get("DYN_NAMESPACE") or "dynamo"

    if config.deployment.type == "disaggregated":
        deployment: DisaggregatedInferenceDeploymentConfig = config.deployment
        if deployment.num_prefill_nodes != deployment.num_prefill_replicas:
            raise ValueError("Local Dynamo requires one prefill node per prefill replica")
        if deployment.num_decode_nodes != deployment.num_decode_replicas:
            raise ValueError("Local Dynamo requires one decode node per decode replica")
        roles = list(config.dynamo_worker_roles)
        gpus_per_worker = config.dynamo_gpus_per_worker
    else:
        roles = list(config.dynamo_worker_roles)
        gpus_per_worker = config.dynamo_gpus_per_worker

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
                process=build_worker_process(
                    config,
                    role,
                    engine_path,
                    nixl_host="127.0.0.1",
                    nixl_port=20100 + worker_index,
                    namespace=resolved_namespace,
                ),
            )
        )
    return specs


def build_dry_run_worker_specs(
    config: InferenceConfig,
    output_dir: Path | None = None,
) -> list[DynamoWorkerSpec]:
    """Build local specs without consulting host GPU hardware."""
    if config.deployment.type == "disaggregated":
        gpu_count = len(config.dynamo_worker_roles) * config.dynamo_gpus_per_worker
    else:
        gpu_count = config.dynamo_gpus_per_worker
    return build_local_worker_specs(
        config,
        output_dir=output_dir,
        gpu_ids=[f"<gpu:{index}>" for index in range(gpu_count)],
        namespace=config.env_vars.get("DYN_NAMESPACE") or "dynamo",
    )


def build_worker_environment(
    spec: DynamoWorkerSpec,
    base_environment: dict[str, str],
) -> dict[str, str]:
    return spec.process.environment(base_environment) | {
        "CUDA_VISIBLE_DEVICES": ",".join(spec.gpu_ids),
        "DYN_SYSTEM_PORT": str(spec.system_port),
    }


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
    environment = os.environ.copy()
    environment.setdefault("DYN_DISCOVERY_BACKEND", "file")
    environment.setdefault("DYN_EVENT_PLANE", "zmq")
    environment.setdefault("DYN_FILE_KV_TTL_SECS", "1800")
    namespace = config.env_vars.get("DYN_NAMESPACE") or environment.get("DYN_NAMESPACE") or f"prime-rl-{os.getpid()}"
    environment["DYN_NAMESPACE"] = namespace
    environment.setdefault("PYTHONHASHSEED", "0")
    specs = build_local_worker_specs(config, namespace=namespace)

    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_stop)
    processes: list[subprocess.Popen] = []
    with tempfile.TemporaryDirectory(prefix="prime-dynamo-") as temporary_dir:
        environment.setdefault("DYN_FILE_KV", str(Path(temporary_dir) / "discovery"))
        frontend = build_frontend_process(config)
        frontend_env = frontend.environment(environment) | {"CUDA_VISIBLE_DEVICES": ""}
        frontend_env.pop("DYN_SYSTEM_PORT", None)

        try:
            processes.append(subprocess.Popen(frontend.command(), env=frontend_env, start_new_session=True))
            for spec in specs:
                worker_env = build_worker_environment(spec, environment)
                processes.append(subprocess.Popen(spec.process.command(), env=worker_env, start_new_session=True))

            exited_process = next((process for process in processes if process.poll() is not None), None)
            while exited_process is None:
                time.sleep(0.2)
                exited_process = next((process for process in processes if process.poll() is not None), None)
            returncode = exited_process.returncode
            if returncode is None:
                raise RuntimeError("Dynamo child exit was observed without a return code")
            # A clean child exit is still a service failure while its siblings are supervised.
            raise SystemExit(returncode if returncode != 0 else 1)
        except KeyboardInterrupt:
            return
        finally:
            for process in reversed(processes):
                _terminate(process)
