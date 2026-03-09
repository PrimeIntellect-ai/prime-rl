from argparse import Namespace
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_config import BaseConfig

from prime_rl.configs.shared import SlurmConfig
from prime_rl.utils.parsers import resolve_reasoning_parser, resolve_tool_call_parser

# Valid vLLM max_lora_rank values (from vllm/config/lora.py)
# TODO: on newer vLLM, can import via `get_args(vllm.config.lora.MaxLoRARanks)`
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)

WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
}


# vLLM all2all backend options for expert-parallel deployments.
All2AllBackend = Literal[
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "flashinfer_all2allv",
    "naive",
    "pplx",
]

DEFAULT_MAX_LORAS = 8
# TODO: The default value is very high because our PipelineRL implementation for
# LoRA isn't ideal We add a lora with the same name instead of changing weights
# inplace Because we dont cancel requests that are past max_async, these
# requests could be using a LoRA that gets unloaded which will crash the
# inference server
DEFAULT_MAX_CPU_LORAS = 100


class VLLMConfig(BaseConfig):
    """Configures vLLM server arguments.

    All fields use vLLM's native arg names. If a value is None, we use the vLLM
    default. We allow arbitrary vLLM args (alo those not defined in this config)
    only via TOML but not CLI.
    """

    model_config = ConfigDict(extra="allow")

    model: Annotated[str, Field(description="Name or path of the HF model to use.")] = "Qwen/Qwen3-0.6B"

    host: Annotated[str | None, Field(description="The host to bind to.")] = None

    port: Annotated[int | None, Field(description="The port to bind to.")] = None

    max_model_len: Annotated[
        int | None,
        Field(description="Maximum model context length."),
    ] = None

    enforce_eager: Annotated[
        bool | None,
        Field(
            description="Whether to enforce eager mode. If False, will use PyTorch eager and cuda graphs in hybrid for maximal performance."
        ),
    ] = None

    trust_remote_code: Annotated[bool | None, Field(description="Whether to trust remote code.")] = None

    tool_call_parser: Annotated[
        str | None,
        Field(
            description='The tool call parser to use. Set to "auto" to infer from the model name.',
        ),
    ] = None

    reasoning_parser: Annotated[
        str | None,
        Field(
            description="Parser for extracting reasoning content from model outputs. Setting this enables reasoning mode."
        ),
    ] = None

    rope_scaling: Annotated[
        dict[str, Any] | str | None,
        Field(
            description='RoPE scaling configuration as a dict. For YaRN, use: {rope_type="yarn", factor=4.0, original_max_position_embeddings=32768}.'
        ),
    ] = None

    tensor_parallel_size: Annotated[
        int,
        Field(description="The tensor parallel size."),
    ] = 1

    data_parallel_size: Annotated[
        int,
        Field(description="The data parallel size."),
    ] = 1

    data_parallel_size_local: Annotated[
        int | None,
        Field(description="Number of data parallel replicas to run on this node."),
    ] = None

    data_parallel_rpc_port: Annotated[
        int | None,
        Field(description="RPC port for data parallel communication."),
    ] = None

    enable_lora: Annotated[
        bool | None,
        Field(description="Whether to enable LoRA."),
    ] = None

    max_loras: Annotated[
        int | None,
        Field(description="The maximum number of LoRAs to use."),
    ] = None

    max_cpu_loras: Annotated[
        int | None,
        Field(description="The maximum number of LoRAs to use on CPU."),
    ] = None

    max_lora_rank: Annotated[
        int | None,
        Field(description="The maximum LoRA rank to use."),
    ] = None

    enable_prefix_caching: Annotated[
        bool | None,
        Field(description="Whether to enable prefix caching."),
    ] = None

    gpu_memory_utilization: Annotated[
        float,
        Field(description="The GPU memory utilization to use."),
    ] = 0.9

    api_server_count: Annotated[
        int,
        Field(description="The number of API servers to use."),
    ] = 1

    seed: Annotated[
        int | None,
        Field(description="Seed the inference components."),
    ] = None

    enable_expert_parallel: Annotated[
        bool,
        Field(description="Enable expert parallelism for MoE models."),
    ] = False

    all2all_backend: Annotated[
        All2AllBackend,
        Field(description="All-to-all backend for expert parallel communication."),
    ] = "allgather_reducescatter"

    enable_eplb: Annotated[
        bool,
        Field(description="Enable expert parallel load balancer (EPLB)."),
    ] = False

    enable_return_routed_experts: Annotated[
        bool,
        Field(description="Whether to enable return routed experts."),
    ] = False

    @model_validator(mode="after")
    def auto_resolve_parsers(self):
        """Resolve tool_call_parser="auto" and reasoning_parser="auto" to the correct parser for the model."""
        self.tool_call_parser = resolve_tool_call_parser(self.model, self.tool_call_parser)
        self.reasoning_parser = resolve_reasoning_parser(self.model, self.reasoning_parser)
        return self

    @model_validator(mode="after")
    def auto_setup_max_lora(self):
        """Auto-setup LoRA settings:
        - Max. 8 LoRAs
        - Max. 100 LoRAs on CPU
        - Rounding up max_lora_rank to the nearest valid vLLM value.
        """
        if self.enable_lora:
            # setup max_loras
            if self.max_loras is None:
                self.max_loras = DEFAULT_MAX_LORAS

            # setup max_lora_rank
            if self.max_lora_rank is not None:
                original_rank = self.max_lora_rank
                for valid_rank in VALID_VLLM_LORA_RANKS:
                    if valid_rank >= self.max_lora_rank:
                        self.max_lora_rank = valid_rank
                        break
                else:
                    raise ValueError(
                        f"max_lora_rank={original_rank} exceeds vLLM maximum of {VALID_VLLM_LORA_RANKS[-1]}"
                    )

            # setup max_cpu_loras
            if self.max_cpu_loras is None:
                self.max_cpu_loras = DEFAULT_MAX_CPU_LORAS

        return self

    @model_validator(mode="after")
    def auto_setup_api_server_count(self):
        """Ensures that we have at least as many API servers as data parallel
        size. Unless LoRA is enabled, in which case only one API server is
        supported (vLLM limitation).
        """
        if "api_server_count" not in self.model_fields_set and self.api_server_count is not None:
            min_api_server_count = self.data_parallel_size_local or self.data_parallel_size
            if self.api_server_count < min_api_server_count:
                self.api_server_count = min_api_server_count

        if self.enable_lora:
            self.api_server_count = 1  # LoRA requires only one API server
        return self

    def to_namespace(self) -> Namespace:
        """Convert VLLMConfig to vLLM-compatible Namespace."""
        data = self.model_dump(exclude_none=True)

        # ALWAYS set logprobs_mode="processed_logprobs"
        data["logprobs_mode"] = "processed_logprobs"

        # Enable auto tool choice when a tool_call_parser is configured
        data["enable_auto_tool_choice"] = self.tool_call_parser is not None

        return Namespace(**data)


class WeightBroadcastConfig(BaseConfig):
    """Configures weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )


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


InferenceDeploymentConfig: TypeAlias = Annotated[
    SingleNodeInferenceDeploymentConfig | MultiNodeInferenceDeploymentConfig, Field(discriminator="type")
]


class InferenceConfig(BaseConfig):
    """Configures inference."""

    vllm: VLLMConfig = VLLMConfig()

    weight_broadcast: Annotated[WeightBroadcastConfig, Field(description="The weight broadcast config.")] = (
        WeightBroadcastConfig()
    )

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

    def to_namespace(self) -> Namespace:
        """Convert to vLLM-compatible Namespace, including worker extension class."""
        namespace = self.vllm.to_namespace()
        namespace.worker_extension_cls = WORKER_EXTENSION_CLS[self.weight_broadcast.type]
        return namespace

    @model_validator(mode="after")
    def validate_multi_node_requires_slurm(self):
        if self.deployment.type == "multi_node" and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node deployment.")
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.template_path is None:
            import prime_rl

            templates_dir = Path(prime_rl.__file__).parent / "templates"
            self.slurm.template_path = templates_dir / "inference.sbatch.j2"
        return self
