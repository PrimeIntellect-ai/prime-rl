from argparse import Namespace
from typing import Annotated, Literal

from pydantic import Field

from zeroband.utils.pydantic_config import BaseConfig, BaseSettings, get_all_fields
from zeroband.utils.utils import rgetattr, rsetattr

# TODO: Setting a thinking/ solution budget makes more sense to be part of the inference config, but is currently handled by the orchestrator because vLLM doesn't support it yet.


class ServerConfig(BaseConfig):
    """Configures the inference server."""

    host: Annotated[str | None, Field(default=None, description="The host to bind to.")]
    port: Annotated[int, Field(default=8000, description="The port to bind to.")]


class ParallelConfig(BaseConfig):
    """Configures multi-node and multi-GPU setups through different types of parallelism (TP, DP, PP)."""

    tp: Annotated[
        int,
        Field(
            default=1,
            description="The tensor parallel size. It is passed to vLLM as `--tensor-parallel-size`",
        ),
    ]

    dp: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="The data parallel size. It is passed to vLLM as `--data-parallel-size`",
        ),
    ]

    def __str__(self) -> str:
        return f"tp={self.tp} dp={self.dp}"


class ModelConfig(BaseConfig):
    """Configures the inference model. Most arguments are passed directly to the vLLM LLM class (https://docs.vllm.ai/en/latest/api/vllm.LLM.html)."""

    name: Annotated[
        str,
        Field(
            default="Qwen/Qwen3-0.6B",
            description="Name or path of the HF model to use.",
        ),
    ]

    dtype: Annotated[
        Literal["auto", "float16", "bfloat16", "float32"],
        Field(
            default="auto",
            description="Data type for model weights and activations. If 'auto' will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. Passed to vLLM as `--dtype`",
        ),
    ]

    max_model_len: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum model context length. If None, will use the maximum context length from model config. Passed to vLLM as `--max-model-len`",
        ),
    ]

    enforce_eager: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to enforce eager mode. If False, will use PyTorch eager and cuda graphs in hybrid for maximal performance. Passed to vLLM as `--enforce-eager`",
        ),
    ]


class InferenceConfig(BaseSettings):
    """Configures inference."""

    # The server configuration
    server: Annotated[ServerConfig, Field(default=ServerConfig())]

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The parallel configuration
    parallel: Annotated[ParallelConfig, Field(default=ParallelConfig())]

    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Seed the inference components. If None, no seeding is used. Passed to vLLM as `--seed`",
        ),
    ]

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
            "parallel.tp": "tensor_parallel_size",
            "parallel.dp": "data_parallel_size",
        }

        for key in get_all_fields(self):
            value = rgetattr(self, key.replace("-", "_"))
            rsetattr(namespace, to_vllm.get(key, key), value)

        return namespace

    def num_gpus(self) -> int:
        return self.parallel.tp * self.parallel.dp
