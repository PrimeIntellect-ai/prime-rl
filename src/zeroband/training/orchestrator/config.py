from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from zeroband.eval.registry import Benchmark
from zeroband.utils.config import ModelConfig, MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class ClientConfig(BaseConfig):
    """Configures the client to be used for inference."""

    base_url: Annotated[
        str,
        Field(
            default="http://localhost:8000/v1",
            description="Base URL of the OpenAI API. By default, it is set to a local inference server.",
        ),
    ]
    api_key: Annotated[
        str,
        Field(
            default="insecure",
            description="API key to use for the OpenAI API. An arbitrary string can be passed if the inference server is not protected by an API key.",
        ),
    ]


class SamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model. Largely follows the vLLM sampling parameters (https://docs.vllm.ai/en/latest/api/vllm.sampling_params.html)."""

    n: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Number of output sequences to return for the given prompt.",
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
        Field(
            default=-1,
            ge=-1,
            description="Number of top tokens to consider. If -1, all tokens are considered.",
        ),
    ]

    min_p: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Minimum probability for a token to be considered, relative to the probability of the most likely token. If 0, all tokens are considered.",
        ),
    ]

    max_tokens: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum number of output tokens to generate per sequence. If None, will generate until maximum context length or EOS token is hit.",
        ),
    ]

    min_tokens: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Minimum number of output tokens to generate per sequence.",
        ),
    ]

    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed to use for sampling. If None, no seeding is used.",
        ),
    ]


# TODO(Mika, Will): Change data config to verifiers environment
# TODO(Mika): Find an elegant way to enable online/ offline difficulty filtering
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
        Field(
            default="info",
            description="Logging level for the inference run. Will determine the logging verbosity and format.",
        ),
    ]

    all_ranks: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to log from all DP ranks. If False, will only log from the main rank (DP rank 0).",
        ),
    ]

    utc: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to use UTC time in the logger. If False, it will default to the local time. If the local time is wrong, you can set it by setting the `TZ` environment variable. For example, `TZ=America/Los_Angeles` will set the local time to SF time.",
        ),
    ]


class PathConfig(BaseConfig):
    """Configures a path used for input/ output operations"""

    path: Annotated[Path, Field(description="Path to write to.")]

    clean: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to clean the path at the beginning of the run. If True, will delete the entire directory.",
        ),
    ]


class OnlineEvalConfig(BaseConfig):
    """Configures online evaluation."""

    ckpt_path: Annotated[
        Path,
        Field(
            default=Path("checkpoints"),
            description="Path to read checkpoints from when doing online evaluation. Expects subdirectories named 'step_x' within the directory.",
        ),
    ]

    interval: Annotated[
        int,
        Field(
            default=100,
            ge=0,
            description="Interval at which to evaluate the model.",
        ),
    ]

    max_steps: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum number of steps to run online evaluation for. If None, will run indefinitely.",
        ),
    ]


class EvalConfig(BaseConfig):
    """Configures evaluation."""

    benchmarks: Annotated[
        list[Benchmark],
        Field(
            default=["math500"],
            description="Benchmarks to evaluate on. By default, it will evaluate only on the MATH-500 benchmark.",
        ),
    ]

    online: Annotated[OnlineEvalConfig | None, Field(default=None)]


class OrchestratorConfig(BaseSettings):
    """Configures the orchestrator for RL training."""

    # The OAI client configuration
    client: Annotated[ClientConfig, Field(default=ClientConfig())]

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The sampling configuration
    sampling: Annotated[SamplingConfig, Field(default=SamplingConfig())]

    # The data configuration
    data: Annotated[DataConfig, Field(default=DataConfig())]

    # The evaluation configuration
    eval: Annotated[EvalConfig | None, Field(default=None)]

    # The logging configuration
    log: Annotated[LogConfig, Field(default=LogConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    collate_mode: Annotated[Literal["packing", "padding"], Field(default="padding")]

    batch_size: Annotated[int, Field(default=128, ge=1, description="Number of samples to train on per step.")]

    micro_batch_size: Annotated[
        int,
        Field(
            default=128,
            ge=1,
            description="Number of samples to train on per micro batch. This value should be tuned based on the hardware available. Usually, to the largest value divisble by the training batch size.",
        ),
    ]

    seq_len: Annotated[
        int,
        Field(
            default=1024,
            description="Sequence length to use for training. If a sample is shorter than this, it will be padded. If a sequence is longer than this, it will be truncated.",
        ),
    ]

    # TODO(Mika): This should be automatic from the number of ZMQ connections
    num_train_workers: Annotated[
        int,
        Field(default=1, ge=1, description="Number of training workers to use for training."),
    ]

    max_steps: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum number of training steps to run. If None, will run indefinitely.",
        ),
    ]

    async_level: Annotated[
        int,
        Field(
            default=2,
            ge=0,
            description="Maximum number of async levels to use. If 0, will do synchronous RL. Else, it will allow to go `async_level` steps ahead of training.",
        ),
    ]

    rollout: Annotated[
        PathConfig,
        Field(
            default=PathConfig(path=Path("rollouts"), clean=True),
            description="Path to write inference outputs to. Will be populated by the orchestrator with responses from inference pool.",
        ),
    ]

    weights: Annotated[
        PathConfig,
        Field(
            default=PathConfig(path=Path("weights"), clean=True),
            description="Path to read updated model weights from. Will be populated by the trainer.",
        ),
    ]

    seed: Annotated[int | None, Field(default=None, description="Random seed for the orchestrator.")]

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.sampling.n != 0:
            raise ValueError("Batch size must be divisible by the number of samples per problem")
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self
