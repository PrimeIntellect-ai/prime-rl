from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from zeroband.eval.registry import Benchmark
from zeroband.utils.config import LogConfig, ModelConfig, MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class ClientConfig(BaseConfig):
    """Configures the client to be used for inference."""

    base_url: Annotated[
        str,
        Field(
            description="Base URL of the OpenAI API. By default, it is set to a local inference server.",
        ),
    ] = "http://localhost:8000/v1"

    api_key: Annotated[
        str,
        Field(
            description="API key to use for the OpenAI API. An arbitrary string can be passed if the inference server is not protected by an API key.",
        ),
    ] = "insecure"


class SamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model. Largely follows the vLLM sampling parameters (https://docs.vllm.ai/en/latest/api/vllm.sampling_params.html)."""

    n: Annotated[
        int,
        Field(
            ge=1,
            description="Number of output sequences to return for the given prompt.",
        ),
    ] = 1

    temperature: Annotated[
        float,
        Field(
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily.",
        ),
    ] = 1.0

    top_p: Annotated[
        float,
        Field(
            gt=0,
            le=1,
            description="Cumulative probability of the top tokens to consider. If 1, all tokens are considered.",
        ),
    ] = 1

    top_k: Annotated[
        int,
        Field(
            ge=-1,
            description="Number of top tokens to consider. If -1, all tokens are considered.",
        ),
    ] = -1

    min_p: Annotated[
        float,
        Field(
            ge=0,
            description="Minimum probability for a token to be considered, relative to the probability of the most likely token. If 0, all tokens are considered.",
        ),
    ] = 0.0

    max_seq_len: Annotated[
        int | None,
        Field(
            description="Maximum number of input and output tokens allowed before aborting a generation. If set, it will dynamically override the `max_tokens` sampling arg based on the number of input tokens of the particular request. The argument is comparable to the `max_model_len` server parameter in vLLM, but moved to the client to allow for dynamic model contexts.",
        ),
    ] = None

    max_tokens: Annotated[
        int | None,
        Field(
            description="Maximum number of output tokens to generate per sequence. If None, will generate until maximum context length or EOS token is hit.",
        ),
    ] = None

    min_tokens: Annotated[
        int,
        Field(
            ge=0,
            description="Minimum number of output tokens to generate per sequence.",
        ),
    ] = 0

    seed: Annotated[
        int | None,
        Field(
            description="Random seed to use for sampling. If None, no seeding is used.",
        ),
    ] = None


class DifficultyFilteringConfig(BaseConfig):
    """Configures difficulty filtering for the dataset."""

    min_solve_rate: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Minimum solve rate for samples to be included.",
        ),
    ] = 0.0

    max_solve_rate: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Maximum solve rate for samples to be included.",
        ),
    ] = 1.0

    solve_rate_field: Annotated[
        str,
        Field(
            description="Field name in the dataset that contains the solve rate.",
        ),
    ] = "solve_rate"


class DataConfig(BaseConfig):
    """Configures the data to be used for inference."""

    name: Annotated[
        str,
        Field(
            description="Name of the HF dataset to use.",
        ),
    ] = "PrimeIntellect/INTELLECT-2-RL-Dataset"

    split: Annotated[str, Field(description="Split of the dataset to use.")] = "train"

    difficulty_filtering: Annotated[
        DifficultyFilteringConfig | None,
        Field(
            description="Optional difficulty filtering configuration. If None, no filtering is applied.",
        ),
    ] = None


class PathConfig(BaseConfig):
    """Configures a path used for input/ output operations"""

    path: Annotated[Path, Field(description="Path to write to.")]

    clean: Annotated[
        bool,
        Field(
            description="Whether to clean the path at the beginning of the run. If True, will delete the entire directory.",
        ),
    ] = False


class EvalConfig(BaseConfig):
    """Configures evaluation."""

    benchmarks: Annotated[
        list[Benchmark],
        Field(
            description="Benchmarks to evaluate on. By default, it will evaluate only on the MATH-500 benchmark.",
        ),
    ] = ["math500"]

    interval: Annotated[
        int,
        Field(
            ge=0,
            description="Interval at which to evaluate the model.",
        ),
    ] = 100

    eval_base_model: Annotated[
        bool,
        Field(
            description="Whether to evaluate the base model we are training on.",
        ),
    ] = False


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the orchestrator."""

    path: Annotated[Path, Field(description="Directory to write checkpoints to.")] = Path("checkpoints")

    interval: Annotated[int, Field(ge=1, description="Interval at which to save the checkpoint.")] = 50

    resume_step: Annotated[
        int | None,
        Field(
            ge=1,
            description="Step to resume orchestrator from. If None, will start from scratch.",
        ),
    ] = None


class OrchestratorConfig(BaseSettings):
    """Configures the orchestrator for RL training."""

    # The OAI client configuration
    client: ClientConfig = ClientConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The sampling configuration
    sampling: SamplingConfig = SamplingConfig()

    # The data configuration
    data: DataConfig = DataConfig()

    # The evaluation configuration
    eval: EvalConfig | None = None

    # The logging configuration
    log: LogConfig = LogConfig(path=Path("logs/orchestrator"))

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    collate_mode: Annotated[Literal["packing", "padding"], Field(description="Collate mode to use.")] = "padding"

    batch_size: Annotated[int, Field(ge=1, description="Number of samples to train on per step.")] = 128

    micro_batch_size: Annotated[
        int,
        Field(
            ge=1,
            description="Number of samples to train on per micro batch. This value should be tuned based on the hardware available. Usually, to the largest value divisble by the training batch size.",
        ),
    ] = 128

    seq_len: Annotated[
        int,
        Field(
            description="Sequence length to use for training. If a sample is shorter than this, it will be padded. If a sequence is longer than this, it will be truncated.",
        ),
    ] = 2048

    # TODO(Mika): This should be automatic from the number of ZMQ connections
    num_train_workers: Annotated[
        int,
        Field(default=1, ge=1, description="Number of training workers to use for training."),
    ] = 1

    max_steps: Annotated[
        int | None,
        Field(
            description="Maximum number of training steps to run. If None, will run indefinitely.",
        ),
    ] = None

    async_level: Annotated[
        int,
        Field(
            ge=0,
            description="Maximum number of async levels to use. If 0, will do synchronous RL. Else, it will allow to go `async_level` steps ahead of training.",
        ),
    ] = 2

    rollout_path: Annotated[
        Path,
        Field(
            description="Path to write inference outputs to. Will be populated by the orchestrator with responses from inference pool.",
        ),
    ] = Path("rollouts")

    weights_path: Annotated[
        Path,
        Field(
            description="Path to read updated model weights from. Will be populated by the trainer.",
        ),
    ] = Path("weights")

    clean: Annotated[
        bool,
        Field(
            description="Whether to clean the rollouts, checkpoint, checkpoint weights and logs directories at the beginning of the run. If True, will forceably, and irreversibly, delete all directories.",
        ),
    ] = True

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5, max async level to ~infinity and disable W&B.",
        ),
    ] = False

    seed: Annotated[int | None, Field(description="Random seed for the orchestrator.")] = None

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.sampling.n != 0:
            raise ValueError("Batch size must be divisible by the number of samples per problem")
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self

    @model_validator(mode="after")
    def validate_bench(self):
        if self.bench:
            self.max_steps = 6  # Run for 1 warmup step + 5 evaluation steps
            self.async_level = 1e9  # Never wait for RL weight checkpoints

        return self
