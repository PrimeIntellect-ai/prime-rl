from argparse import Namespace
from typing import Annotated, Literal

from pydantic import Field

from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings, get_all_fields
from prime_rl.utils.utils import rgetattr, rsetattr

ServerType = Literal["vllm", "sglang"]


class ServerConfig(BaseConfig):
    """Configures the inference server."""

    host: Annotated[str | None, Field(description="The host to bind to.")] = None
    port: Annotated[int, Field(description="The port to bind to.")] = 8000
    type: Annotated[ServerType, Field(description="Backend type.")] = "vllm"


class ParallelConfig(BaseConfig):
    """Configures multi-node and multi-GPU setups through different types of parallelism (TP, DP)."""

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
    """Configures the inference model. Most arguments are passed directly to the vLLM LLM class."""

    name: Annotated[
        str,
        Field(
            description="Name or path of the HF model to use.",
        ),
    ] = "Qwen/Qwen3-0.6B"

    dtype: Annotated[
        Literal["auto", "float16", "bfloat16", "float32"],
        Field(
            description="Data type for model weights and activations. Passed to vLLM as `--dtype`",
        ),
    ] = "auto"

    max_model_len: Annotated[
        int | None,
        Field(
            description="Maximum model context length. If None, will use the maximum context length from the model config.",
        ),
    ] = None

    enforce_eager: Annotated[
        bool,
        Field(
            description="Whether to enforce eager mode. Passed to vLLM as `--enforce-eager`",
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
            description="The tool call parser to use. Passed to vLLM as `--tool-call-parser`.",
        ),
    ] = "hermes"


class InferenceConfig(BaseSettings):
    """Configures inference."""

    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    parallel: ParallelConfig = ParallelConfig()

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
            "gpu_memory_utilization": "gpu_memory_utilization",
        }

        for key in get_all_fields(self):
            value = rgetattr(self, key.replace("-", "_"))
            rsetattr(namespace, to_vllm.get(key, key), value)

        # Ensure processed logprobs so the orchestrator sees transformed logprobs by default.
        rsetattr(namespace, "logprobs_mode", "processed_logprobs")

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
            "1",
        ]
        parser = self.model.tool_call_parser
        if parser == "hermes":
            parser = None
        if parser:
            args += ["--tool-call-parser", parser]
        if self.seed is not None:
            args += ["--random-seed", str(self.seed)]
        return args
