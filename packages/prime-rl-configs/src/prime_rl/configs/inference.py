from argparse import Namespace
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_config import BaseConfig

from prime_rl.configs.shared import BaseModelConfig, SlurmConfig
from prime_rl.utils.config import find_package_resource, rgetattr, rsetattr

# TODO: Set thinking/ solution budget

InferenceBackend: TypeAlias = Literal["vllm", "dynamo"]
DynamoRouterMode: TypeAlias = Literal[
    "round-robin",
    "random",
    "kv",
    "direct",
    "power-of-two",
    "least-loaded",
    "device-aware-weighted",
]


class ServerConfig(BaseConfig):
    """Configures the inference server."""

    host: Annotated[str | None, Field(description="The host to bind to.")] = None
    port: Annotated[int, Field(description="The port to bind to.")] = 8000
    liveness_timeout_seconds: Annotated[
        float,
        Field(
            gt=0,
            description=(
                "Timeout in seconds for the /liveness endpoint's internal vLLM worker RPC. "
                "If Kubernetes liveness probes are enabled, keep the probe timeoutSeconds at least this high."
            ),
        ),
    ] = 30.0


class ParallelConfig(BaseConfig):
    """Configures multi-node and multi-GPU setups through different types of parallelism (TP, DP, PP)."""

    tp: Annotated[
        int,
        Field(
            description="The tensor parallel size. It is passed to vLLM as `--tensor-parallel-size`",
        ),
    ] = 1

    dp: Annotated[
        int,
        Field(
            ge=1,
            description="The data parallel size. It is passed to vLLM as `--data-parallel-size`",
        ),
    ] = 1

    def __str__(self) -> str:
        return f"tp={self.tp} dp={self.dp}"


class ModelConfig(BaseModelConfig):
    """Configures the inference model. Most arguments are passed directly to the vLLM LLM class (https://docs.vllm.ai/en/latest/api/vllm.LLM.html)."""

    dtype: Annotated[
        Literal["auto", "float16", "bfloat16", "float32"],
        Field(
            description="Data type for model weights and activations. If 'auto' will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. Passed to vLLM as `--dtype`",
        ),
    ] = "auto"

    max_model_len: Annotated[
        int | None,
        Field(
            description="Maximum model context length. If None, will use the maximum context length from model config. Passed to vLLM as `--max-model-len`",
        ),
    ] = None

    enforce_eager: Annotated[
        bool,
        Field(
            description="Whether to enforce eager mode. If False, will use PyTorch eager and cuda graphs in hybrid for maximal performance. Passed to vLLM as `--enforce-eager`",
        ),
    ] = False

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code. Passed to vLLM engine init",
        ),
    ] = False

    chat_template: Annotated[
        str | None,
        Field(
            description="Chat template to use. Can be a Jinja2 template string or a path to a template file. "
            "Passed to vLLM as `--chat-template`. If None, uses the model's default.",
        ),
    ] = None

    tool_call_parser: Annotated[
        str | None,
        Field(
            description="The tool call parser to use. Passed to vLLM as `--tool-call-parser`. "
            'Set to "auto" to infer from the model name.',
        ),
    ] = "auto"

    reasoning_parser: Annotated[
        str | None,
        Field(
            description="Parser for extracting reasoning content from model outputs. Passed to vLLM as `--reasoning-parser`. Setting this enables reasoning mode.",
        ),
    ] = None

    rope_scaling: Annotated[
        dict[str, Any] | str | None,
        Field(
            description='RoPE scaling configuration as a dict. For YaRN, use: {rope_type="yarn", factor=4.0, original_max_position_embeddings=32768} or. Passed to vLLM as `--rope-scaling`.',
        ),
    ] = None


class WeightBroadcastConfig(BaseConfig):
    """Configures weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )


class DynamoConfig(BaseConfig):
    """Configures the experimental Dynamo backend."""

    deploy_router: Annotated[
        bool,
        Field(
            description=(
                "Launch Dynamo's frontend/router from the inference entrypoint. Disable only when an "
                "external Dynamo frontend/router is already deployed and orchestrator.client.base_url points at it."
            ),
        ),
    ] = True

    namespace: Annotated[
        str,
        Field(description="Dynamo namespace used by the frontend and worker for service discovery."),
    ] = "dynamo"

    discovery_backend: Annotated[
        Literal["file", "etcd", "kubernetes", "mem"],
        Field(description="Dynamo discovery backend. Use 'file' for local single-node runs without etcd/NATS."),
    ] = "file"

    request_plane: Annotated[
        Literal["tcp", "nats", "http"],
        Field(description="Dynamo request plane used between the frontend and worker."),
    ] = "tcp"

    router_mode: Annotated[
        DynamoRouterMode,
        Field(description="Dynamo frontend router mode used for OpenAI chat-completions traffic."),
    ] = "round-robin"

    min_initial_workers: Annotated[
        int | None,
        Field(
            ge=0,
            description="Optional Dynamo frontend minimum worker count before the router starts serving requests.",
        ),
    ] = None

    event_plane: Annotated[
        Literal["nats", "zmq"] | None,
        Field(description="Dynamo event plane. If None, Dynamo derives it from discovery_backend."),
    ] = None

    system_port: Annotated[
        int,
        Field(ge=1, le=65535, description="Dynamo worker system-server port for /health and /engine routes."),
    ] = 8081

    use_vllm_tokenizer: Annotated[
        bool,
        Field(
            description=(
                "Use Dynamo's text-in/text-out vLLM worker path. prime-rl defaults this off so the "
                "Dynamo frontend tokenizes OpenAI requests before dispatching them to the worker."
            ),
        ),
    ] = False

    frontend_extra: Annotated[
        dict[str, Any],
        Field(description="Extra CLI arguments for python -m dynamo.frontend."),
    ] = {}

    worker_extra: Annotated[
        dict[str, Any],
        Field(description="Extra CLI arguments for the Dynamo vLLM worker."),
    ] = {}


class KVCacheOffloadConfig(BaseModel):
    """CPU KV cache offloading for vLLM inference workers."""

    model_config = ConfigDict(extra="forbid")

    cpu_bytes: Annotated[int, Field(gt=0, description="CPU bytes available for KV cache offloading per worker.")] = (
        1_000_000_000
    )


# Valid vLLM max_lora_rank values (from vllm/config/lora.py)
# TODO: on newer vLLM, can import via `get_args(vllm.config.lora.MaxLoRARanks)`
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)

# vLLM all2all backend options for expert-parallel deployments.
All2AllBackend = Literal[
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "flashinfer_nvlink_one_sided",
    "flashinfer_nvlink_two_sided",
]


class BaseInferenceDeploymentConfig(BaseModel):
    """Base deployment config for inference."""

    model_config = ConfigDict(extra="forbid")

    gpus_per_node: Annotated[int, Field(description="Number of GPUs per node.")] = 8


class SingleNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    """Configures a single-node inference deployment."""

    type: Literal["single_node"] = "single_node"


class MultiNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    """Configures a multi-node inference deployment. Each node runs an independent vLLM replica."""

    type: Literal["multi_node"] = "multi_node"

    num_nodes: Annotated[int, Field(ge=1, description="Number of inference nodes.")] = 2

    router_port: Annotated[int, Field(description="Port for the vllm-router.")] = 8000
    backend_port: Annotated[int, Field(description="Port for vLLM backend instances.")] = 8100
    router_policy: Annotated[
        str, Field(description="Routing policy for the vllm-router (e.g. 'consistent_hash', 'round_robin').")
    ] = "consistent_hash"


class DisaggregatedInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    """Configures a disaggregated prefill/decode inference deployment.

    Each inference replica is split into separate prefill and decode node groups.
    Requires NIXL for KV transfer and a vllm-router for request routing.

    Multi-replica support: set ``num_prefill_replicas`` / ``num_decode_replicas``
    to run multiple independent vLLM instances within the prefill / decode node
    groups.  For example, ``num_prefill_nodes=4, num_prefill_replicas=2`` creates
    two prefill vLLM instances each spanning 2 nodes (EP16 with 8 GPUs/node).
    """

    type: Literal["disaggregated"] = "disaggregated"

    num_prefill_nodes: Annotated[int, Field(ge=1, description="Total number of prefill nodes.")] = 1
    num_decode_nodes: Annotated[int, Field(ge=1, description="Total number of decode nodes.")] = 1

    num_prefill_replicas: Annotated[
        int,
        Field(
            ge=1,
            description="Number of independent prefill vLLM instances. Must evenly divide num_prefill_nodes.",
        ),
    ] = 1
    num_decode_replicas: Annotated[
        int,
        Field(
            ge=1,
            description="Number of independent decode vLLM instances. Must evenly divide num_decode_nodes.",
        ),
    ] = 1

    router_port: Annotated[int, Field(description="Port for the vllm-router on each replica.")] = 8000
    prefill_port: Annotated[int, Field(description="Port for prefill vLLM instances.")] = 8100
    decode_port: Annotated[int, Field(description="Port for decode vLLM instances.")] = 8200
    router_policy: Annotated[
        str, Field(description="Routing policy for the vllm-router (e.g. 'consistent_hash', 'round_robin').")
    ] = "consistent_hash"

    prefill_env_overrides: Annotated[
        dict[str, str],
        Field(description="Extra environment variables exported only on prefill nodes."),
    ] = {}
    decode_env_overrides: Annotated[
        dict[str, str],
        Field(description="Extra environment variables exported only on decode nodes."),
    ] = {}

    @property
    def num_nodes(self) -> int:
        return self.num_prefill_nodes + self.num_decode_nodes

    @model_validator(mode="after")
    def validate_replicas_divide_nodes(self):
        if self.num_prefill_nodes % self.num_prefill_replicas != 0:
            raise ValueError(
                f"num_prefill_replicas ({self.num_prefill_replicas}) must evenly divide "
                f"num_prefill_nodes ({self.num_prefill_nodes})"
            )
        if self.num_decode_nodes % self.num_decode_replicas != 0:
            raise ValueError(
                f"num_decode_replicas ({self.num_decode_replicas}) must evenly divide "
                f"num_decode_nodes ({self.num_decode_nodes})"
            )
        return self


InferenceDeploymentConfig: TypeAlias = Annotated[
    SingleNodeInferenceDeploymentConfig | MultiNodeInferenceDeploymentConfig | DisaggregatedInferenceDeploymentConfig,
    Field(discriminator="type"),
]


class InferenceExperimentalConfig(BaseConfig):
    """Experimental features for inference."""


class InferenceConfig(BaseConfig):
    """Configures inference."""

    backend: Annotated[
        InferenceBackend,
        Field(
            description=(
                "Inference server backend to launch. vLLM is the default; Dynamo is an experimental "
                "backend for OpenAI-compatible rollouts with prime-rl weight updates."
            )
        ),
    ] = "vllm"

    # The server configuration
    server: ServerConfig = ServerConfig()

    # The model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)

    # The parallel configuration
    parallel: ParallelConfig = ParallelConfig()

    enable_lora: Annotated[
        bool,
        Field(
            description="Whether to enable LORA. Passed to vLLM as `--enable-lora`",
        ),
    ] = False

    max_loras: Annotated[
        int,
        Field(
            description="The maximum number of LoRAs to use. Passed to vLLM as `--max-loras`",
        ),
    ] = 8

    # TODO: The default value is very high because our areal impl for lora isn't ideal
    # We add a lora with the same name instead of changing weights inplace
    # Because we dont cancel requests that are past max_async, these requests could be using a LoRA that gets unloaded which will crash the inference server
    max_cpu_loras: Annotated[
        int,
        Field(
            description="The maximum number of LoRAs to use on CPU. Passed to vLLM as `--max-cpu-loras`",
        ),
    ] = 100

    max_lora_rank: Annotated[
        int | None,
        Field(
            description="The maximum LoRA rank to use. Passed to vLLM as `--max-lora-rank`",
        ),
    ] = None

    lora_target_modules: Annotated[
        list[str] | None,
        Field(
            description="The target modules for LoRA. Passed to vLLM as `--lora-target-modules`.",
        ),
    ] = None

    enable_prefix_caching: Annotated[
        bool | None,
        Field(
            description="Whether to enable prefix caching. Passed to vLLM as `--enable-prefix-caching`",
        ),
    ] = None

    gpu_memory_utilization: Annotated[
        float,
        Field(
            description="The GPU memory utilization to use. Passed to vLLM as `--gpu-memory-utilization`",
        ),
    ] = 0.9

    api_server_count: Annotated[
        int,
        Field(
            ge=0,
            description="The number of API servers to use. Passed to vLLM as `--api-server-count`. Set to 0 for headless mode.",
        ),
    ] = 1

    data_parallel_size_local: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of data parallel replicas to run on this node. Passed to vLLM as `--data-parallel-size-local`.",
        ),
    ] = None

    data_parallel_rpc_port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            description="RPC port for data parallel communication. Passed to vLLM as `--data-parallel-rpc-port`.",
        ),
    ] = 13345

    seed: Annotated[
        int,
        Field(
            description="Seed the inference components. Passed to vLLM as `--seed`",
        ),
    ] = 0

    enable_expert_parallel: Annotated[
        bool,
        Field(
            description="Enable expert parallelism for MoE models. Passed to vLLM as `--enable-expert-parallel`.",
        ),
    ] = False

    all2all_backend: Annotated[
        All2AllBackend,
        Field(
            description="All-to-all backend for expert parallel communication. Passed to vLLM as `--all2all-backend`.",
        ),
    ] = "allgather_reducescatter"

    enable_eplb: Annotated[
        bool,
        Field(
            description="Enable expert parallel load balancer (EPLB). Passed to vLLM as `--enable-eplb`.",
        ),
    ] = False

    enable_dbo: Annotated[
        bool,
        Field(
            description="Enable dual batch overlap (DBO). Passed to vLLM as `--enable-dbo`.",
        ),
    ] = False

    use_deep_gemm: Annotated[
        bool,
        Field(
            description="Force DeepGEMM FP8 kernels via VLLM_USE_DEEP_GEMM=1. Only works with per-tensor FP8 quantization (e.g. GLM-5-FP8).",
        ),
    ] = False

    weight_broadcast: Annotated[WeightBroadcastConfig, Field(description="The weight broadcast config.")] = (
        WeightBroadcastConfig()
    )

    dynamo: Annotated[
        DynamoConfig,
        Field(description="Experimental Dynamo-specific settings used when backend='dynamo'."),
    ] = DynamoConfig()

    kv_cache_offload: Annotated[
        KVCacheOffloadConfig | None,
        Field(
            description=(
                "CPU KV cache offload config for inference workers. Standard inference uses vLLM's "
                "OffloadingConnector. Disaggregated P/D deployments combine it with NIXL through "
                "MultiConnector in the SLURM launcher."
            ),
        ),
    ] = None

    enable_return_routed_experts: Annotated[
        bool,
        Field(
            description="Whether to enable return routed experts. Passed to vLLM as `--enable-return-routed-experts`",
        ),
    ] = False

    enable_fp32_lm_head: Annotated[
        bool,
        Field(
            description="Run the lm_head projection in fp32 via a native bf16xbf16 -> fp32 GEMM (`torch.mm` with `out_dtype=torch.float32`). Stabilizes logprob precision under FP8/bf16 inference, matching SGLang's `--enable-fp32-lm-head`. Implemented as a monkey-patch over vLLM's LogitsProcessor, activated by setting `additional_config[\"fp32_lm_head\"] = True` on the vLLM config.",
        ),
    ] = False

    vllm_extra: Annotated[
        dict[str, Any],
        Field(
            description="Extra arguments to pass to vLLM. These are applied as attributes on the vLLM namespace after config translation.",
        ),
    ] = {}

    # Launcher-only fields

    deployment: Annotated[
        InferenceDeploymentConfig,
        Field(
            description="Deployment configuration for inference.",
        ),
    ] = SingleNodeInferenceDeploymentConfig()

    slurm: Annotated[
        SlurmConfig | None,
        Field(
            description="SLURM configuration. If set, the run will be submitted as a SLURM job instead of running locally.",
        ),
    ] = None

    output_dir: Annotated[Path, Field(description="Directory for SLURM logs and generated scripts.")] = Path("outputs")

    dry_run: Annotated[bool, Field(description="Only validate and dump resolved configs and exit early.")] = False

    experimental: Annotated[
        InferenceExperimentalConfig,
        Field(description="Experimental features for inference."),
    ] = InferenceExperimentalConfig()

    @model_validator(mode="after")
    def validate_multi_node_requires_slurm(self):
        if self.deployment.type == "multi_node" and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node deployment.")
        return self

    @model_validator(mode="after")
    def validate_backend_support(self):
        if self.backend != "dynamo":
            return self

        if self.deployment.type != "single_node":
            raise ValueError("inference.backend='dynamo' currently supports only single_node deployment.")
        if self.kv_cache_offload is not None:
            raise ValueError("inference.backend='dynamo' does not support prime-rl kv_cache_offload plumbing yet.")
        return self

    @model_validator(mode="after")
    def auto_setup_kv_cache_offload(self):
        if self.kv_cache_offload is not None:
            if self.enable_prefix_caching is False:
                raise ValueError("KV cache offloading requires inference.enable_prefix_caching to be true.")
            if "enable_prefix_caching" not in self.model_fields_set:
                self.enable_prefix_caching = True

        return self

    @model_validator(mode="after")
    def auto_setup_disaggregated(self):
        """Auto-configure inference for disaggregated P/D: enable EP and compute DP."""
        if self.deployment.type == "disaggregated":
            if "enable_expert_parallel" not in self.model_fields_set:
                self.enable_expert_parallel = True
            if "enable_eplb" not in self.model_fields_set:
                self.enable_eplb = False
            gpus_per_node = self.deployment.gpus_per_node
            tp = self.parallel.tp
            dp_per_node = gpus_per_node // tp
            if self.data_parallel_size_local is None:
                self.data_parallel_size_local = dp_per_node
            if self.parallel.dp == 1:
                self.parallel.dp = dp_per_node
            if self.api_server_count == 1:
                self.api_server_count = dp_per_node
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.template_path is None:
            templates_dir = find_package_resource("templates")
            if templates_dir is not None:
                self.slurm.template_path = templates_dir / "inference.sbatch.j2"
        return self

    @model_validator(mode="after")
    def auto_setup_max_lora_rank(self):
        """Auto-setup max_lora_rank by rounding up to the nearest valid vLLM value.

        vLLM only accepts specific values for max_lora_rank: (1, 8, 16, 32, 64, 128, 256, 320, 512).
        This validator ensures that any configured rank is rounded up to the minimum valid value
        that can serve adapters of the requested rank.
        """
        if self.max_lora_rank is not None:
            original_rank = self.max_lora_rank
            for valid_rank in VALID_VLLM_LORA_RANKS:
                if valid_rank >= self.max_lora_rank:
                    self.max_lora_rank = valid_rank
                    break
            else:
                raise ValueError(f"max_lora_rank={original_rank} exceeds vLLM maximum of {VALID_VLLM_LORA_RANKS[-1]}")
        return self

    @model_validator(mode="after")
    def auto_setup_api_server_count(self):
        """
        Ensures that we have at least as many API servers as data parallel
        size. Unless LoRA is enabled, in which case only one API server is
        supported (vLLM limitation).
        """
        if self.vllm_extra.get("headless", False):
            self.api_server_count = 0
            return self

        if "api_server_count" not in self.model_fields_set:
            min_api_server_count = self.data_parallel_size_local or self.parallel.dp
            if self.api_server_count < min_api_server_count:
                self.api_server_count = min_api_server_count

        if self.enable_lora:
            self.api_server_count = 1  # LoRA requires only one API server
        return self

    def to_vllm(self) -> Namespace:
        """Convert InferenceConfig to vLLM-compatible Namespace."""
        namespace = Namespace()
        to_vllm = {
            "server.host": "host",
            "server.port": "port",
            "server.liveness_timeout_seconds": "liveness_timeout_seconds",
            "model.name": "model",
            "model.dtype": "dtype",
            "model.max_model_len": "max_model_len",
            "model.enforce_eager": "enforce_eager",
            "model.trust_remote_code": "trust_remote_code",
            "model.chat_template": "chat_template",
            "model.tool_call_parser": "tool_call_parser",
            "model.reasoning_parser": "reasoning_parser",
            "model.rope_scaling": "rope_scaling",
            "parallel.tp": "tensor_parallel_size",
            "parallel.dp": "data_parallel_size",
            "data_parallel_size_local": "data_parallel_size_local",
            "data_parallel_rpc_port": "data_parallel_rpc_port",
            "enable_lora": "enable_lora",
            "enable_prefix_caching": "enable_prefix_caching",
            "max_loras": "max_loras",
            "max_cpu_loras": "max_cpu_loras",
            "max_lora_rank": "max_lora_rank",
            "lora_target_modules": "lora_target_modules",
            "gpu_memory_utilization": "gpu_memory_utilization",
            "api_server_count": "api_server_count",
            "enable_return_routed_experts": "enable_return_routed_experts",
            "enable_expert_parallel": "enable_expert_parallel",
            "all2all_backend": "all2all_backend",
            "enable_eplb": "enable_eplb",
            "enable_dbo": "enable_dbo",
            "seed": "seed",
        }

        for config_key, vllm_key in to_vllm.items():
            value = rgetattr(self, config_key.replace("-", "_"))
            rsetattr(namespace, vllm_key, value)

        # Set `logprobs_mode` to `processed_logprobs` by default
        rsetattr(namespace, "logprobs_mode", "processed_logprobs")

        if self.kv_cache_offload is not None:
            rsetattr(
                namespace,
                "kv_transfer_config",
                {
                    "kv_connector": "OffloadingConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use": int(self.kv_cache_offload.cpu_bytes),
                    },
                },
            )

        # Pass prime-rl-specific flags through vLLM's additional_config dict;
        # workers read these via get_current_vllm_config().additional_config.
        if self.enable_fp32_lm_head:
            existing = getattr(namespace, "additional_config", None) or {}
            existing["fp32_lm_head"] = True
            rsetattr(namespace, "additional_config", existing)

        # Remove chat_template if not set (vLLM doesn't accept None)
        if namespace.chat_template is None:
            delattr(namespace, "chat_template")

        # Remove reasoning_parser if not set (vLLM doesn't accept None)
        if namespace.reasoning_parser is None:
            delattr(namespace, "reasoning_parser")

        # Remove lora_target_modules if not set (vLLM doesn't accept None)
        if hasattr(namespace, "lora_target_modules") and namespace.lora_target_modules is None:
            delattr(namespace, "lora_target_modules")

        # Remove rope_scaling if not set (vLLM doesn't accept None)
        if hasattr(namespace, "rope_scaling"):
            if namespace.rope_scaling is None:
                delattr(namespace, "rope_scaling")

        return namespace

    def to_dynamo_frontend(self) -> Namespace:
        """Convert InferenceConfig to Dynamo frontend CLI namespace."""
        namespace = Namespace()
        to_frontend = {
            "server.host": "http_host",
            "server.port": "http_port",
            "dynamo.namespace": "namespace",
            "dynamo.discovery_backend": "discovery_backend",
            "dynamo.request_plane": "request_plane",
            "dynamo.router_mode": "router_mode",
            "dynamo.min_initial_workers": "min_initial_workers",
            "dynamo.event_plane": "event_plane",
            "model.name": "model_name",
        }

        for config_key, frontend_key in to_frontend.items():
            value = rgetattr(self, config_key.replace("-", "_"))
            rsetattr(namespace, frontend_key, value)

        namespace.dyn_chat_processor = "vllm"

        for key, value in self.dynamo.frontend_extra.items():
            setattr(namespace, key, value)

        for optional_key in ("http_host", "event_plane", "model_name", "min_initial_workers"):
            value = getattr(namespace, optional_key, None)
            if value is None:
                delattr(namespace, optional_key)

        return namespace

    def to_dynamo_vllm(self) -> Namespace:
        """Convert InferenceConfig to Dynamo vLLM worker CLI namespace."""
        namespace = Namespace()
        to_dynamo_vllm = {
            "dynamo.namespace": "namespace",
            "dynamo.discovery_backend": "discovery_backend",
            "dynamo.request_plane": "request_plane",
            "dynamo.event_plane": "event_plane",
            "dynamo.use_vllm_tokenizer": "use_vllm_tokenizer",
            "model.name": "model",
            "model.dtype": "dtype",
            "model.max_model_len": "max_model_len",
            "model.enforce_eager": "enforce_eager",
            "model.trust_remote_code": "trust_remote_code",
            "model.rope_scaling": "rope_scaling",
            "parallel.tp": "tensor_parallel_size",
            "parallel.dp": "data_parallel_size",
            "data_parallel_size_local": "data_parallel_size_local",
            "data_parallel_rpc_port": "data_parallel_rpc_port",
            "enable_lora": "enable_lora",
            "enable_prefix_caching": "enable_prefix_caching",
            "max_loras": "max_loras",
            "max_cpu_loras": "max_cpu_loras",
            "max_lora_rank": "max_lora_rank",
            "lora_target_modules": "lora_target_modules",
            "gpu_memory_utilization": "gpu_memory_utilization",
            "enable_return_routed_experts": "enable_return_routed_experts",
            "enable_expert_parallel": "enable_expert_parallel",
            "all2all_backend": "all2all_backend",
            "enable_eplb": "enable_eplb",
            "enable_dbo": "enable_dbo",
            "seed": "seed",
        }

        for config_key, dynamo_key in to_dynamo_vllm.items():
            value = rgetattr(self, config_key.replace("-", "_"))
            rsetattr(namespace, dynamo_key, value)

        namespace.served_model_name = self.model.name

        if self.model.tool_call_parser not in (None, "auto"):
            namespace.dyn_tool_call_parser = self.model.tool_call_parser
        if self.model.reasoning_parser is not None:
            namespace.dyn_reasoning_parser = self.model.reasoning_parser

        namespace.logprobs_mode = "processed_logprobs"

        if self.enable_fp32_lm_head:
            existing = getattr(namespace, "additional_config", None) or {}
            existing["fp32_lm_head"] = True
            rsetattr(namespace, "additional_config", existing)

        for key, value in self.vllm_extra.items():
            setattr(namespace, key, value)
        for key, value in self.dynamo.worker_extra.items():
            setattr(namespace, key, value)

        for optional_key in (
            "event_plane",
            "rope_scaling",
            "data_parallel_size_local",
            "data_parallel_rpc_port",
            "enable_prefix_caching",
            "max_lora_rank",
            "lora_target_modules",
            "additional_config",
        ):
            if not hasattr(namespace, optional_key):
                continue

            value = getattr(namespace, optional_key)
            if value is None:
                delattr(namespace, optional_key)

        return namespace
