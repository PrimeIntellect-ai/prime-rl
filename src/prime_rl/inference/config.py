from argparse import Namespace
from typing import Annotated, Literal

from pydantic import Field

from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings, get_all_fields
from prime_rl.utils.utils import rgetattr, rsetattr

ServerType = Literal["vllm", "sglang"]

# TODO: Set thinking/ solution budget


class ServerConfig(BaseConfig):
    """Configures the inference server."""

    host: Annotated[str | None, Field(description="The host to bind to.")] = None
    port: Annotated[int, Field(description="The port to bind to.")] = 8000
    server_type: Annotated[ServerType, Field(description="Backend type.")] = "vllm"


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

    pp: Annotated[
        int,
        Field(description="The pipeline parallel size. It is passed to vLLM as `--pipeline-parallel-size`"),
    ] = 1

    def __str__(self) -> str:
        return f"tp={self.tp} dp={self.dp} pp={self.pp}"


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
        str | None,
        Field(
            description="The tool call parser to use. For vLLM, defaults to 'hermes' if unset. For sglang, only pass values supported by sglang.",
        ),
    ] = None

    quantization: Annotated[
        str | None,
        Field(description="Quantization to use. Passed to vLLM as `--quantization`"),
    ] = None


class InferenceConfig(BaseSettings):
    """Configures inference."""

    # The server configuration
    server: ServerConfig = ServerConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The parallel configuration
    parallel: ParallelConfig = ParallelConfig()

    seed: Annotated[
        int | None,
        Field(
            description="Seed the inference components. If None, no seeding is used. Passed to vLLM as `--seed`",
        ),
    ] = None

    mem_fraction_static: Annotated[
        float | None,
        Field(description="Static GPU memory fraction. Passed as `--mem-fraction-static`"),
    ] = None

    logprob_start_len: Annotated[
        int | None,
        Field(description="Start length for logprob calculation. Passed as `--logprob-start-len`"),
    ] = None

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
            "model.enable_auto_tool_choice": "enable_auto_tool_choice",  # requires underscores (unlike on CLI)
            "model.tool_call_parser": "tool_call_parser",  # requires underscores (unlike on CLI)
            "parallel.tp": "tensor_parallel_size",
            "parallel.dp": "data_parallel_size",
            "parallel.pp": "pipeline_parallel_size",
            "model.quantization": "quantization",
            "mem_fraction_static": "mem_fraction_static",
            "logprob_start_len": "logprob_start_len",
        }

        for key in get_all_fields(self):
            value = rgetattr(self, key.replace("-", "_"))
            rsetattr(namespace, to_vllm.get(key, key), value)

        # Set defaults specific to vLLM
        # Set `logprobs_mode` to `processed_logprobs` by default
        rsetattr(namespace, "logprobs_mode", "processed_logprobs")

        # Default tool_call_parser to 'hermes' for vLLM if not specified
        if getattr(namespace, "tool_call_parser", None) is None:
            rsetattr(namespace, "tool_call_parser", "hermes")

        return namespace

    def to_sglang(self) -> list[str]:
        args: list[str] = ["--model-path", self.model.name, "--port", str(self.server.port)]
        if self.server.host:
            args += ["--host", self.server.host]
        if self.model.dtype:
            args += ["--dtype", self.model.dtype]
        if self.model.max_model_len is not None:
            args += ["--context-length", str(self.model.max_model_len)]
        if self.model.trust_remote_code:
            args.append("--trust-remote-code")
        args += [
            "--tp-size",
            str(self.parallel.tp),
            "--dp-size",
            str(self.parallel.dp),
            "--pp-size",
            str(self.parallel.pp),
        ]
        if self.model.quantization:
            args += ["--quantization", self.model.quantization]
        if self.mem_fraction_static is not None:
            args += ["--mem-fraction-static", str(self.mem_fraction_static)]
        if self.seed is not None:
            args += ["--random-seed", str(self.seed)]
        if self.logprob_start_len is not None:
            args += ["--logprob-start-len", str(self.logprob_start_len)]
        if self.model.tool_call_parser:
            args += ["--tool-call-parser", self.model.tool_call_parser]
        return args
