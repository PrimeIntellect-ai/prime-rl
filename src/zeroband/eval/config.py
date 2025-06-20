from typing import Annotated, Literal

from pydantic import Field, model_validator

from zeroband.utils.config import MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class SamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model. Largely follows the vLLM sampling parameters (https://docs.vllm.ai/en/latest/api/vllm.sampling_params.html)."""

    n: Annotated[int, Field(default=16, ge=1, description="Number of output sequences to return for the given prompt.")]

    presence_penalty: Annotated[
        float,
        Field(
            default=0,
            description="Penalizes new tokens based on whether they appear in the generated text so far. Values >0 => penalize, Values <0 => reward repeated tokens",
        ),
    ]

    frequency_penalty: Annotated[
        float,
        Field(
            default=0,
            description="Penalizes new tokens based on their frequency in the generated text so far. Values <0 => penalize repetition, Values >0 => reward repetition",
        ),
    ]

    temperature: Annotated[
        float,
        Field(
            default=1.0,
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily.",
        ),
    ]

    top_p: Annotated[
        float,
        Field(
            default=1,
            gt=0,
            le=1,
            description="Cumulative probability of the top tokens to consider. If 1, all tokens are considered.",
        ),
    ]

    top_k: Annotated[
        int,
        Field(default=-1, ge=-1, description="Number of top tokens to consider. If -1, all tokens are considered."),
    ]

    min_p: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Minimum probability for a token to be considered, relative to the probability of the most likely token. If 0, all tokens are considered.",
        ),
    ]

    logprobs: Annotated[
        int | None,
        Field(
            default=0,
            description="Number of tokens to return log probabilities for. If None, no probability is returned. For all other values, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens (e.g. 0 returns only the logprob of the chosen token)",
        ),
    ]

    max_tokens: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum number of output tokens to generate per sequence. If None, will generate until maximum context length or EOS token is hit.",
        ),
    ]
    min_tokens: Annotated[int, Field(default=0, ge=0, description="Minimum number of output tokens to generate per sequence.")]

    @model_validator(mode="after")
    def convert_negative_logprobs_to_none(self):
        """Convert negative logprobs values to None to disable logprobs calculation."""
        if self.logprobs is not None and self.logprobs < 0:
            self.logprobs = None
        return self


class ParallelConfig(BaseConfig):
    """Configures multi-node and multi-GPU setups through different types of parallelism (TP, DP, PP)."""

    tp: Annotated[
        int | Literal["auto"],
        Field(
            default=1,
            description="Number of local GPUs to use for tensor parallelism. It is directly passed to vLLM. If 'auto', will be set to all available local GPUs.",
        ),
    ]


class ModelConfig(BaseConfig):
    """Configures the inference model. Most arguments are passed directly to the vLLM LLM class (https://docs.vllm.ai/en/latest/api/vllm.LLM.html)."""

    name: Annotated[str, Field(default="Qwen/Qwen3-0.6B", description="Name or path of the HF model to use.")]

    dtype: Annotated[
        Literal["auto", "float16", "bfloat16", "float32"],
        Field(
            default="auto",
            description="Data type for model weights and activations. If 'auto' will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.",
        ),
    ]

    kv_cache_dtype: Annotated[
        Literal["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        Field(default="auto", description="Data type for the KV cache. If 'auto' will use the model data type."),
    ]

    max_model_len: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum model context length. If None, will use the maximum context length from model config.",
        ),
    ]

    quantization: Annotated[
        Literal["awq", "gguf", "gptq", "bitsandbytes", "fp8"] | None,
        Field(
            default=None,
            description="Method used to quantize the weights. If None, will apply the default quantization (if any) from model config.",
        ),
    ]

    enforce_eager: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to enforce eager mode. If False, will use PyTorch eager and cuda graphs in hybrid for maximal performance.",
        ),
    ]

    device: Annotated[Literal["auto", "cuda", "cpu"], Field(default="auto", description="Device to use for inference.")]

    enable_thinking: Annotated[
        bool,
        Field(default=True, description="Whether to enable thinking. Used by the `format_prompts` function to prepend a thinking prompt."),
    ]


class DataConfig(BaseConfig):
    """Configures the data to be used for inference."""

    name: Annotated[
        str,
        Field(
            default="PrimeIntellect/INTELLECT-2-RL-Dataset",
            description="Name of the HF dataset to use.",
        ),
    ]

    split: Annotated[str, Field(default="train", description="Split of the dataset to use.")]


class LogConfig(BaseConfig):
    """Configures the logger."""

    level: Annotated[
        Literal["debug", "info"],
        Field(default="info", description="Logging level for the inference run. Will determine the logging verbosity and format."),
    ]

    all_ranks: Annotated[
        bool, Field(default=False, description="Whether to log from all DP ranks. If False, will only log from the main rank (DP rank 0).")
    ]

    utc: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to use UTC time in the logger. If False, it will default to the local time. If the local time is wrong, you can set it by setting the `TZ` environment variable. For example, `TZ=America/Los_Angeles` will set the local time to SF time.",
        ),
    ]


class Config(BaseSettings):
    """Configures evaluation."""

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The sampling configuration
    sampling: Annotated[SamplingConfig, Field(default=SamplingConfig())]

    # The data configuration
    data: Annotated[DataConfig, Field(default=DataConfig())]

    # The parallel configuration
    parallel: Annotated[ParallelConfig, Field(default=ParallelConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    # The logging configuration
    log: Annotated[LogConfig, Field(default=LogConfig())]

    max_batch_size: Annotated[
        int | Literal["auto"],
        Field(
            default="auto",
            description="Maximum number of of sequences to decode in parallel. If 'auto', the maximum batch size is automatically computed.",
        ),
    ]

    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed used across inference components. If None, no seeding is used.",
        ),
    ]
