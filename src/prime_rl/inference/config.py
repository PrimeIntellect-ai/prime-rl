import os
from argparse import Namespace
from typing import Annotated, Literal

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


class WeightBroadcastConfig(BaseSettings):
    """Configures weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )


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

    gpu_memory_utilization: Annotated[
        float,
        Field(
            description="The GPU memory utilization to use. Passed to vLLM as `--gpu-memory-utilization`",
        ),
    ] = 0.9

    seed: Annotated[
        int | None,
        Field(
            description="Seed the inference components. If None, no seeding is used. Passed to vLLM as `--seed`",
        ),
    ] = None

    weight_broadcast: Annotated[WeightBroadcastConfig, Field(description="The weight broadcast config.")] = (
        WeightBroadcastConfig()
    )

    @model_validator(mode="after")
    def nccl_and_dp(self):
        if self.weight_broadcast.type == "nccl" and self.parallel.dp != 1:
            raise ValueError("NCCL broadcast backend requires data parallel size to be 1")
        return self

    @model_validator(mode="after")
    def set_env_var_for_lora(self):
        if self.enable_lora:
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
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
            "parallel.tp": "tensor_parallel_size",
            "parallel.dp": "data_parallel_size",
            "enable_lora": "enable_lora",
            "gpu_memory_utilization": "gpu_memory_utilization",
        }

        for key in get_all_fields(self):
            value = rgetattr(self, key.replace("-", "_"))
            rsetattr(namespace, to_vllm.get(key, key), value)

        # Set `logprobs_mode` to `processed_logprobs` by default
        rsetattr(namespace, "logprobs_mode", "processed_logprobs")

        return namespace
