from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.orchestrator.advantage import AdvantageType
from prime_rl.utils.config import LogConfig, ModelConfig, MultiMonitorConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class ClientConfig(BaseConfig):
    """Configures the client to be used for inference."""

    timeout: Annotated[
        int,
        Field(
            description="Timeout in seconds for the OpenAI API. By default, it is set to 1200 seconds.",
        ),
    ] = 1200

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
    """Configures how tokens are sampled from the model for training. Largely follows the vLLM sampling parameters."""

    temperature: Annotated[
        float,
        Field(
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily.",
        ),
    ] = 1.0

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


class EvalSamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model for evaluation. Largely follows the vLLM sampling parameters."""

    temperature: Annotated[
        float | None,
        Field(
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    top_p: Annotated[
        float | None,
        Field(
            description="Cumulative probability of the top tokens to consider. If 1, all tokens are considered. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    top_k: Annotated[
        int | None,
        Field(
            description="Number of top tokens to consider. If -1, all tokens are considered. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    min_p: Annotated[
        float | None,
        Field(
            description="Minimum probability for a token to be considered, relative to the probability of the most likely token. If 0, all tokens are considered. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    max_tokens: Annotated[
        int | None,
        Field(
            description="Maximum number of output tokens to generate per turn. If None, will generate until maximum context length or EOS token is hit.",
        ),
    ] = None

    min_tokens: Annotated[
        int | None,
        Field(
            description="Minimum number of output tokens to generate per sequence. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    seed: Annotated[
        int | None,
        Field(
            description="Random seed to use for sampling. If None, no seeding is used. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None


class EnvironmentConfig(BaseConfig):
    """Configures the environment to be used for inference."""

    id: Annotated[str, Field(description="ID of the environment to use.")] = "reverse-text"
    args: Annotated[dict, Field(description="Arguments to pass to the environment.")] = {}


class EvalConfig(BaseConfig):
    """Configures evaluation using verifiers environments."""

    environment_ids: Annotated[
        list[str],
        Field(
            description="List of verifiers environment IDs to evaluate on. Each ID also serves as the metric prefix."
        ),
    ] = []

    environment_args: Annotated[
        dict[str, dict],
        Field(
            description="Per-environment overrides keyed by ID; forwarded as kwargs to verifiers.load_environment(id, **args)."
        ),
    ] = {}

    num_examples: Annotated[
        list[int],
        Field(
            description="Number of examples to evaluate per environment. Set all or none; if None, defaults to -1 for every ID."
        ),
    ] = []

    rollouts_per_example: Annotated[
        list[int],
        Field(
            description="Number of samples to generate per example for each environment (length must match eval.environment_ids)."
        ),
    ] = []

    sampling: EvalSamplingConfig = Field(
        default_factory=EvalSamplingConfig,
        description="Shared sampling configuration for evals; can differ from training sampling.",
    )

    save: Annotated[
        bool,
        Field(
            description="Whether to save the evaluation artifacts to the outputs directory.",
        ),
    ] = True

    @model_validator(mode="after")
    def _validate_and_fill_eval_lists(self):
        # If rollouts_per_example is empty, default to 1 for all ids
        if len(self.rollouts_per_example) == 0:
            self.rollouts_per_example = [1 for _ in self.environment_ids]
        elif len(self.rollouts_per_example) != len(self.environment_ids):
            raise ValueError("Number of rollouts_per_example entries must match number of ids")

        # num_examples: if empty/unspecified, default to -1 for all; else length must match ids
        if len(self.num_examples) == 0:
            self.num_examples = [-1 for _ in self.environment_ids]
        elif len(self.num_examples) != len(self.environment_ids):
            raise ValueError("Number of num_examples entries must match number of ids")

        return self


class OnlineEvalConfig(EvalConfig):
    """Configures online evaluation."""

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


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the orchestrator."""

    interval: Annotated[int | None, Field(ge=1, description="Interval at which to save the checkpoint.")] = None

    resume_step: Annotated[
        int | None,
        Field(
            ge=1,
            description="Step to resume orchestrator from. If None, will start from scratch.",
        ),
    ] = None

    keep: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints.",
        ),
    ] = None


class BufferConfig(BaseModel):
    """Base config for all buffer types."""

    from_scratch: Annotated[
        bool,
        Field(
            description="Whether to initialize the metadata and rollout buffer from scratch. Defaults to True, which means we will initialize empty metadata and rollout buffers. If False, we expect columns `metadata` and `rollouts` to be present in the environment dataset to initialize the buffer from.",
        ),
    ] = True


class SimpleBufferConfig(BufferConfig):
    type: Literal["simple"] = "simple"


class DifficultyPoolBufferConfig(BufferConfig):
    type: Literal["difficulty-pool"] = "difficulty-pool"

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


class OnlineDifficultyBufferConfig(BufferConfig):
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


DataBufferConfigType: TypeAlias = SimpleBufferConfig | DifficultyPoolBufferConfig | OnlineDifficultyBufferConfig


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
    eval: OnlineEvalConfig | None = None

    # Data buffer configuration
    buffer: Annotated[DataBufferConfigType, Field(discriminator="type")] = SimpleBufferConfig()

    # The logging configuration
    log: LogConfig = LogConfig()

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    batch_size: Annotated[int, Field(ge=1, description="Number of samples to train on per step.")] = 128

    micro_batch_size: Annotated[
        int,
        Field(
            ge=1,
            description="Number of samples to train on per micro batch. This value should be tuned based on the hardware available. Usually, to the largest value divisble by the training batch size.",
        ),
    ] = 128

    rollouts_per_example: Annotated[
        int,
        Field(
            ge=1,
            description="Number of output sequences to return per example during training.",
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
    ] = False

    length_bonus: Annotated[
        float | None,
        Field(
            description="Add an extra reward to the shortest correct answer in fully correct rollout groups.",
        ),
    ] = 0.0

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

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5, max async level to ~infinity and disable W&B.",
        ),
    ] = False

    seed: Annotated[int | None, Field(description="Random seed for the orchestrator.")] = 42

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.rollouts_per_example != 0:
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
