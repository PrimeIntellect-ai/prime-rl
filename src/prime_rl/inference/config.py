import os
from argparse import Namespace
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings, get_all_fields
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
            description="Whether to enable auto tool choice. Passed to vLLM as `--enable-auto-tool-choice`",
        ),
    ] = False

    tool_call_parser: Annotated[
        str,
        Field(
            description="The tool call parser to use. Passed to vLLM as `--tool-call-parser`",
        ),
    ] = "hermes"

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


class WeightBroadcastConfig(BaseSettings):
    """Configures weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )


# Valid vLLM max_lora_rank values (from vllm/config/lora.py)
# TODO: on newer vLLM, can import via `get_args(vllm.config.lora.MaxLoRARanks)`
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


class InferenceConfig(BaseSettings):
    """Configures inference."""

    # The server configuration
    server: ServerConfig = ServerConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

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

    weight_broadcast: Annotated[WeightBroadcastConfig, Field(description="The weight broadcast config.")] = (
        WeightBroadcastConfig()
    )

    @model_validator(mode="after")
    def nccl_and_dp(self):
        if self.weight_broadcast.type == "nccl" and self.parallel.dp != 1:
            raise ValueError("NCCL broadcast backend requires data parallel size to be 1")
        return self

    @model_validator(mode="after")
    def auto_setup_dynamic_lora_updating(self):
        if self.enable_lora:
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        return self

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
            "max_loras": "max_loras",
            "max_cpu_loras": "max_cpu_loras",
            "max_lora_rank": "max_lora_rank",
            "gpu_memory_utilization": "gpu_memory_utilization",
            "api_server_count": "api_server_count",
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
