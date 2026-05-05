from __future__ import annotations

import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from prime_rl.baselines.metrics import DEFAULT_KS

LaunchMode = Literal["external", "local", "srun", "srun_multinode"]


@dataclass
class LaunchConfig:
    mode: LaunchMode = "external"
    port: int = 8000
    host: str = "127.0.0.1"
    tp: int = 1
    dp: int = 1
    api_server_count: int | None = None
    gpus: str | None = None
    nodes: int = 1
    gpus_per_node: int | None = None
    srun_job_id: str | None = None
    router_port: int | None = None
    backend_port: int | None = None
    data_parallel_size_local: int | None = None
    data_parallel_rpc_port: int = 13345
    enable_expert_parallel: bool = False
    use_deep_gemm: bool = False
    srun_network: str | None = None
    srun_cpus_per_task: int | None = None
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = None
    enforce_eager: bool = False
    chat_template: str | None = None
    launch_prefix: list[str] = field(default_factory=list)
    wait_timeout_s: float = 900.0
    server_start_retries: int = 1
    extra_args: list[str] = field(default_factory=list)
    vllm_extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineConfig:
    env_id: str
    model: str
    output_dir: Path
    protocol: str = "single_shot"
    dataset: str | None = None
    run_id: str | None = None
    seed: int = 0
    num_examples: int = 8
    record_ids: list[str] = field(default_factory=list)
    rollouts_per_example: int = 1
    max_concurrency: int = 32
    score_max_concurrency: int | None = None
    max_retries: int = 0
    fail_on_error: bool = True
    success_threshold: float = 1.0
    ks: tuple[int, ...] = DEFAULT_KS
    env_args: dict[str, Any] = field(default_factory=dict)
    sampling_args: dict[str, Any] = field(default_factory=dict)
    base_url: str | None = None
    api_key_var: str = "VLLM_API_KEY"
    client_type: str = "openai_chat_completions"
    api_profile: str | None = None
    verifiers_path: Path | None = None
    env_paths: list[Path] = field(default_factory=list)
    launch: LaunchConfig = field(default_factory=LaunchConfig)

    @property
    def dataset_label(self) -> str:
        if self.dataset:
            return self.dataset
        if "dataset_name" in self.env_args:
            return str(self.env_args["dataset_name"])
        if "subset" in self.env_args:
            return str(self.env_args["subset"])
        return self.env_id


def _coerce_launch(raw: dict[str, Any]) -> LaunchConfig:
    prefix = raw.get("launch_prefix", [])
    if isinstance(prefix, str):
        prefix = shlex.split(prefix)
    extra_args = raw.get("extra_args", [])
    if isinstance(extra_args, str):
        extra_args = shlex.split(extra_args)
    return LaunchConfig(
        mode=raw.get("mode", "external"),
        port=int(raw.get("port", 8000)),
        host=str(raw.get("host", "127.0.0.1")),
        tp=int(raw.get("tp", 1)),
        dp=int(raw.get("dp", 1)),
        api_server_count=raw.get("api_server_count"),
        gpus=raw.get("gpus"),
        nodes=int(raw.get("nodes", 1)),
        gpus_per_node=(int(raw["gpus_per_node"]) if raw.get("gpus_per_node") is not None else None),
        srun_job_id=raw.get("srun_job_id"),
        router_port=(int(raw["router_port"]) if raw.get("router_port") is not None else None),
        backend_port=(int(raw["backend_port"]) if raw.get("backend_port") is not None else None),
        data_parallel_size_local=(
            int(raw["data_parallel_size_local"]) if raw.get("data_parallel_size_local") is not None else None
        ),
        data_parallel_rpc_port=int(raw.get("data_parallel_rpc_port", 13345)),
        enable_expert_parallel=bool(raw.get("enable_expert_parallel", False)),
        use_deep_gemm=bool(raw.get("use_deep_gemm", False)),
        srun_network=raw.get("srun_network"),
        srun_cpus_per_task=(
            int(raw["srun_cpus_per_task"]) if raw.get("srun_cpus_per_task") is not None else None
        ),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.85)),
        max_model_len=raw.get("max_model_len"),
        enforce_eager=bool(raw.get("enforce_eager", False)),
        chat_template=raw.get("chat_template"),
        launch_prefix=list(prefix),
        wait_timeout_s=float(raw.get("wait_timeout_s", 900.0)),
        server_start_retries=int(raw.get("server_start_retries", 1)),
        extra_args=list(extra_args),
        vllm_extra=dict(raw.get("vllm_extra", {})),
    )


def _coerce_record_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item) for item in value]


def load_config(path: Path) -> BaselineConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    env_paths = [Path(p) for p in raw.get("env_paths", [])]
    if raw.get("env_path"):
        env_paths.append(Path(raw["env_path"]))

    return BaselineConfig(
        env_id=raw["env_id"],
        model=raw["model"],
        output_dir=Path(raw["output_dir"]),
        protocol=raw.get("protocol", "single_shot"),
        dataset=raw.get("dataset"),
        run_id=raw.get("run_id"),
        seed=int(raw.get("seed", 0)),
        num_examples=int(raw.get("num_examples", 8)),
        record_ids=_coerce_record_ids(raw.get("record_ids")),
        rollouts_per_example=int(raw.get("rollouts_per_example", 1)),
        max_concurrency=int(raw.get("max_concurrency", 32)),
        score_max_concurrency=(
            int(raw["score_max_concurrency"])
            if raw.get("score_max_concurrency") is not None
            else None
        ),
        max_retries=int(raw.get("max_retries", 0)),
        fail_on_error=bool(raw.get("fail_on_error", True)),
        success_threshold=float(raw.get("success_threshold", 1.0)),
        ks=tuple(int(k) for k in raw.get("ks", DEFAULT_KS)),
        env_args=dict(raw.get("env_args", {})),
        sampling_args=dict(raw.get("sampling", raw.get("sampling_args", {}))),
        base_url=raw.get("base_url"),
        api_key_var=raw.get("api_key_var", "VLLM_API_KEY"),
        client_type=raw.get("client_type", "openai_chat_completions"),
        api_profile=raw.get("api_profile"),
        verifiers_path=Path(raw["verifiers_path"]) if raw.get("verifiers_path") else None,
        env_paths=env_paths,
        launch=_coerce_launch(dict(raw.get("launch", {}))),
    )
