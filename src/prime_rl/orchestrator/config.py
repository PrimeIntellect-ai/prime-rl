from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.eval.registry import Benchmark
from prime_rl.orchestrator.advantage import AdvantageType
from prime_rl.utils.config import LogConfig, ModelConfig, MultiMonitorConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class ClientConfig(BaseConfig):
    """Configures the client to be used for inference."""

    host: Annotated[
        str,
        Field(
            description="Host to use for the OpenAI API. By default, it is set to a local inference server.",
        ),
    ] = "localhost"

    port: Annotated[
        int,
        Field(
            description="Port to use for the OpenAI API. By default, it is set to a local inference server.",
        ),
    ] = 8000

    api_key: Annotated[
        str,
        Field(
            description="API key to use for the OpenAI API. An arbitrary string can be passed if the inference server is not protected by an API key.",
        ),
    ] = "insecure"


class SamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model. Largely follows the vLLM sampling parameters (https://docs.vllm.ai/en/latest/api/vllm.sampling_params.html)."""

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

    max_tokens: Annotated[
        int | None,
        Field(
            description="Maximum number of output tokens to generate per turn. If None, will generate until maximum context length or EOS token is hit.",
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


class EnvironmentConfig(BaseConfig):
    """Configures the environment to be used for inference."""

    id: Annotated[str, Field(description="ID of the environment to use.")] = "vf-reverse-text"
    args: Annotated[dict, Field(description="Arguments to pass to the environment.")] = {}


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

    rollouts_per_prompt: Annotated[
        list[int],
        Field(
            description="Number of samples to generate for each benchmark.",
        ),
    ] = [1]

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
    ] = True

    @model_validator(mode="after")
    def validate_rollouts_per_prompt(self):
        if len(self.rollouts_per_prompt) != len(self.benchmarks):
            raise ValueError("Number of rollouts per prompt must be the same as the number of benchmarks")
        return self


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


class SimpleBufferConfig(BaseModel):
    type: Literal["simple"] = "simple"


class DifficultyPoolBufferConfig(BaseModel):
    type: Literal["difficulty-pool"] = "difficulty-pool"

    difficulty_field: Annotated[
        str | None,
        Field(
            description="Field name in the dataset that contains difficulty information. Should only contain `easy`, `normal` and `hard`. If None, all samples are treated as `normal` initially.",
        ),
    ] = None

    easy_border: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="If a problem has more than `easy_border` average reward across rollouts, it will be moved to the easy pool.",
        ),
    ] = 0.8

    hard_border: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="If a problem has less than `hard_border` average reward across rollouts, it will be moved to the hard pool.",
        ),
    ] = 0.2

    # TODO: Maybe make this float | int to allow for specific numbers of easy/hard samples?
    easy_fraction: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Fraction of the batch that should consist of easy samples.",
        ),
    ] = 0.1

    hard_fraction: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Fraction of the batch that should consist of hard samples.",
        ),
    ] = 0.1


class OnlineDifficultyBufferConfig(BaseModel):
    type: Literal["online-difficulty"] = "online-difficulty"

    min_reward: Annotated[
        float | None,
        Field(
            ge=0,
            le=1,
            description="Minimum reward to include the sample in a batch.",
        ),
    ] = 0.01

    max_reward: Annotated[
        float | None,
        Field(
            ge=0,
            le=1,
            description="Maximum reward to include the sample in a batch.",
        ),
    ] = 0.99

    oversampling_factor: Annotated[
        float,
        Field(
            gt=0,
            description="Factor by which to oversample during filtering to ensure sufficient samples.",
        ),
    ] = 1.0


DataBufferConfig: TypeAlias = SimpleBufferConfig | DifficultyPoolBufferConfig | OnlineDifficultyBufferConfig


class OrchestratorConfig(BaseSettings):
    """Configures the orchestrator for RL training."""

    # The OAI client configuration
    client: ClientConfig = ClientConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The sampling configuration
    sampling: SamplingConfig = SamplingConfig()

    # The environment configuration
    environment: EnvironmentConfig = EnvironmentConfig()

    # The evaluation configuration
    eval: EvalConfig | None = None

    # Data buffer configuration
    buffer: DataBufferConfig = Field(discriminator="type", default=SimpleBufferConfig())

    # The logging configuration
    log: LogConfig = LogConfig()

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    collate_mode: Annotated[Literal["packing", "padding"], Field(description="Collate mode to use.")] = "packing"

    batch_size: Annotated[int, Field(ge=1, description="Number of samples to train on per step.")] = 128

    micro_batch_size: Annotated[
        int,
        Field(
            ge=1,
            description="Number of samples to train on per micro batch. This value should be tuned based on the hardware available. Usually, to the largest value divisble by the training batch size.",
        ),
    ] = 128

    rollouts_per_prompt: Annotated[
        int,
        Field(
            ge=1,
            description="Number of output sequences to return for the given prompt.",
        ),
    ] = 1

    advantage_type: Annotated[
        AdvantageType,
        Field(
            description="Type of advantage computation to use. For details on the variants please refer directly to their docstrings."
        ),
    ] = "drgrpo"

    seq_len: Annotated[
        int,
        Field(
            description="Sequence length to use for training. If a sample is shorter than this, it will be padded. If a sequence is longer than this, it will be truncated.",
        ),
    ] = 2048

    mask_env_responses: Annotated[
        bool,
        Field(
            description="Whether to mask environment responses from the loss.",
        ),
    ] = True

    mask_truncated_completions: Annotated[
        bool,
        Field(
            description="Whether to mask truncated completions from the loss.",
        ),
    ] = False

    zero_truncated_completions: Annotated[
        bool,
        Field(
            description="Whether to override reward scores with 0 for truncated completions.",
        ),
    ] = True

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

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5, max async level to ~infinity and disable W&B.",
        ),
    ] = False

    seed: Annotated[int | None, Field(description="Random seed for the orchestrator.")] = None

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.rollouts_per_prompt != 0:
            raise ValueError("Batch size must be divisible by the number of samples per problem")
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.max_steps = 4  # Run for 1 warmup step + 3 evaluation steps
            self.async_level = int(1e9)  # Never wait for RL weight checkpoints

            # Disable evaluation
            self.eval = None
            if self.monitor.wandb:
                self.monitor.wandb.log_extras = None

        return self
