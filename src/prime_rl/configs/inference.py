from argparse import Namespace
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_config import BaseConfig

from prime_rl.configs.shared import BaseModelConfig, SlurmConfig
from prime_rl.utils.utils import rgetattr, rsetattr

# TODO: Set thinking/ solution budget


class ServerConfig(BaseConfig):
    """Configures the inference server."""

    host: Annotated[str | None, Field(description="The host to bind to.")] = None
    port: Annotated[int, Field(description="The port to bind to.")] = 8000


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

    tool_call_parser: Annotated[
        str | None,
        Field(
            description="The tool call parser to use. Passed to vLLM as `--tool-call-parser`. "
            'Set to "auto" to infer from the model name.',
        ),
    ] = None

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


# Valid vLLM max_lora_rank values (from vllm/config/lora.py)
# TODO: on newer vLLM, can import via `get_args(vllm.config.lora.MaxLoRARanks)`
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)

# vLLM all2all backend options for expert-parallel deployments.
All2AllBackend = Literal[
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "flashinfer_all2allv",
    "naive",
    "pplx",
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


class InferenceConfig(BaseConfig):
    """Configures inference."""

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

    use_deep_gemm: Annotated[
        bool,
        Field(
            description="Force DeepGEMM FP8 kernels via VLLM_USE_DEEP_GEMM=1. Only works with per-tensor FP8 quantization (e.g. GLM-5-FP8).",
        ),
    ] = False

    weight_broadcast: Annotated[WeightBroadcastConfig, Field(description="The weight broadcast config.")] = (
        WeightBroadcastConfig()
    )

    enable_return_routed_experts: Annotated[
        bool,
        Field(
            description="Whether to enable return routed experts. Passed to vLLM as `--enable-return-routed-experts`",
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

    @model_validator(mode="after")
    def validate_multi_node_requires_slurm(self):
        if self.deployment.type == "multi_node" and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node deployment.")
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
            import prime_rl

            templates_dir = Path(prime_rl.__file__).parent / "templates"
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
            "model.name": "model",
            "model.dtype": "dtype",
            "model.max_model_len": "max_model_len",
            "model.enforce_eager": "enforce_eager",
            "model.trust_remote_code": "trust_remote_code",
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
            "gpu_memory_utilization": "gpu_memory_utilization",
            "api_server_count": "api_server_count",
            "enable_return_routed_experts": "enable_return_routed_experts",
            "enable_expert_parallel": "enable_expert_parallel",
            "all2all_backend": "all2all_backend",
            "enable_eplb": "enable_eplb",
            "seed": "seed",
        }

        for config_key, vllm_key in to_vllm.items():
            value = rgetattr(self, config_key.replace("-", "_"))
            rsetattr(namespace, vllm_key, value)

        # Set `logprobs_mode` to `processed_logprobs` by default
        rsetattr(namespace, "logprobs_mode", "processed_logprobs")

        # Remove reasoning_parser if not set (vLLM doesn't accept None)
        if namespace.reasoning_parser is None:
            delattr(namespace, "reasoning_parser")

        # Remove rope_scaling if not set (vLLM doesn't accept None)
        if hasattr(namespace, "rope_scaling"):
            if namespace.rope_scaling is None:
                delattr(namespace, "rope_scaling")

        return namespace
