from argparse import Namespace
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings, get_all_fields
from prime_rl.utils.utils import rgetattr, rsetattr

MODEL_TOOL_CALL_PARSER: dict[str, str] = {
    # GLM-4.5
    "zai-org/GLM-4.5": "glm45",
    "zai-org/GLM-4.5-FP8": "glm45",
    "zai-org/GLM-4.5-Base": "glm45",
    "zai-org/GLM-4.5-Air": "glm45",
    "zai-org/GLM-4.5-Air-FP8": "glm45",
    "zai-org/GLM-4.5-Air-Base": "glm45",
    "zai-org/GLM-4.5V": "glm45",
    "zai-org/GLM-4.5V-FP8": "glm45",
    # GLM-4.7
    "zai-org/GLM-4.7": "glm47",
    "zai-org/GLM-4.7-FP8": "glm47",
    "zai-org/GLM-4.7-Flash": "glm47",
    # MiniMax M2
    "MiniMaxAI/MiniMax-M2": "minimax_m2",
    "MiniMaxAI/MiniMax-M2.1": "minimax_m2",
    "MiniMaxAI/MiniMax-M2.5": "minimax_m2",
    # INTELLECT-3
    "PrimeIntellect/INTELLECT-3": "hermes",
    "PrimeIntellect/INTELLECT-3-FP8": "hermes",
    "PrimeIntellect/INTELLECT-3.1": "hermes",
    # Qwen3 dense
    "Qwen/Qwen3-0.6B": "hermes",
    "Qwen/Qwen3-0.6B-Base": "hermes",
    "Qwen/Qwen3-0.6B-FP8": "hermes",
    "Qwen/Qwen3-1.7B": "hermes",
    "Qwen/Qwen3-1.7B-Base": "hermes",
    "Qwen/Qwen3-1.7B-FP8": "hermes",
    "Qwen/Qwen3-4B": "hermes",
    "Qwen/Qwen3-4B-Base": "hermes",
    "Qwen/Qwen3-4B-FP8": "hermes",
    "Qwen/Qwen3-8B": "hermes",
    "Qwen/Qwen3-8B-Base": "hermes",
    "Qwen/Qwen3-8B-FP8": "hermes",
    "Qwen/Qwen3-14B": "hermes",
    "Qwen/Qwen3-14B-Base": "hermes",
    "Qwen/Qwen3-14B-FP8": "hermes",
    "Qwen/Qwen3-32B": "hermes",
    "Qwen/Qwen3-32B-FP8": "hermes",
    # Qwen3 MoE
    "Qwen/Qwen3-30B-A3B": "hermes",
    "Qwen/Qwen3-30B-A3B-Base": "hermes",
    "Qwen/Qwen3-30B-A3B-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B": "hermes",
    "Qwen/Qwen3-235B-A22B-FP8": "hermes",
    # Qwen3 2507
    "Qwen/Qwen3-4B-Instruct-2507": "hermes",
    "Qwen/Qwen3-4B-Thinking-2507": "hermes",
    "Qwen/Qwen3-4B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-4B-Thinking-2507-FP8": "hermes",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "hermes",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "hermes",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "hermes",
    "Qwen/Qwen3-235B-A22B-Thinking-2507": "hermes",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": "hermes",
    # Qwen3-Next
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Thinking": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8": "hermes",
    # Qwen3-Coder
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": "hermes",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": "hermes",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "hermes",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": "hermes",
    # Qwen3-Coder-Next
    "Qwen/Qwen3-Coder-Next": "hermes",
    "Qwen/Qwen3-Coder-Next-Base": "hermes",
    "Qwen/Qwen3-Coder-Next-FP8": "hermes",
    # Qwen3.5
    "Qwen/Qwen3.5-397B-A17B": "hermes",
    "Qwen/Qwen3.5-397B-A17B-FP8": "hermes",
}

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


class ModelConfig(BaseConfig):
    """Configures the inference model. Most arguments are passed directly to the vLLM LLM class (https://docs.vllm.ai/en/latest/api/vllm.LLM.html)."""

    name: Annotated[
        str,
        Field(
            description="Name or path of the HF model to use.",
        ),
    ] = "Qwen/Qwen3-0.6B"

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

    enable_auto_tool_choice: Annotated[
        bool,
        Field(
            description="Whether to enable auto tool choice. Passed to vLLM as `--enable-auto-tool-choice`. "
            "Automatically set to True when tool_call_parser is configured.",
        ),
    ] = False

    tool_call_parser: Annotated[
        str | None,
        Field(
            description="The tool call parser to use. Passed to vLLM as `--tool-call-parser`. "
            "If not set, automatically inferred from the model name.",
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

    @model_validator(mode="after")
    def resolve_tool_call_parser(self):
        if self.tool_call_parser is None:
            parser = MODEL_TOOL_CALL_PARSER.get(self.name)
            if parser is not None:
                self.tool_call_parser = parser

        if self.tool_call_parser is not None:
            self.enable_auto_tool_choice = True

        return self


class WeightBroadcastConfig(BaseSettings):
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


class InferenceConfig(BaseSettings):
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
            ge=1,
            description="The number of API servers to use. Passed to vLLM as `--api-server-count`",
        ),
    ] = 1

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

    weight_broadcast: Annotated[WeightBroadcastConfig, Field(description="The weight broadcast config.")] = (
        WeightBroadcastConfig()
    )

    @model_validator(mode="after")
    def round_up_max_lora_rank(self):
        """Round up max_lora_rank to the nearest valid vLLM value.

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
        if self.api_server_count < self.parallel.dp:
            self.api_server_count = self.parallel.dp

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
            "model.enable_auto_tool_choice": "enable_auto_tool_choice",
            "model.tool_call_parser": "tool_call_parser",
            "model.reasoning_parser": "reasoning_parser",
            "model.rope_scaling": "rope_scaling",
            "parallel.tp": "tensor_parallel_size",
            "parallel.dp": "data_parallel_size",
            "enable_lora": "enable_lora",
            "enable_prefix_caching": "enable_prefix_caching",
            "max_loras": "max_loras",
            "max_cpu_loras": "max_cpu_loras",
            "max_lora_rank": "max_lora_rank",
            "gpu_memory_utilization": "gpu_memory_utilization",
            "api_server_count": "api_server_count",
            "enable_expert_parallel": "enable_expert_parallel",
            "all2all_backend": "all2all_backend",
            "enable_eplb": "enable_eplb",
        }

        for key in get_all_fields(self):
            value = rgetattr(self, key.replace("-", "_"))
            rsetattr(namespace, to_vllm.get(key, key), value)

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
