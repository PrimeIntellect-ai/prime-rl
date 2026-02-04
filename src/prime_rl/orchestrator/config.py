from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AliasChoices, BaseModel, Field, model_validator

from prime_rl.transport.config import FileSystemTransportConfig, TransportConfigType
from prime_rl.utils.config import (
    ClientConfig,
    HeartbeatConfig,
    LogConfig,
    PrimeMonitorConfig,
    WandbWithExtrasConfig,
)
from prime_rl.utils.config import (
    ModelConfig as BaseModelConfig,
)
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class OptimizerConfig(BaseConfig):
    """Per-run optimizer configuration for multi-run training."""

    lr: Annotated[
        float,
        Field(
            ge=0,
            description="Learning rate for this run.",
        ),
    ] = 1e-4


class LoRAConfig(BaseConfig):
    """Per-run LoRA configuration for multi-run training."""

    name: Annotated[
        str | None,
        Field(
            description="Name of the LoRA adapter. If None, auto-generated from rank and alpha.",
        ),
    ] = None

    rank: Annotated[
        int | None,
        Field(
            ge=1,
            description="LoRA rank for this run. Must be <= trainer's max rank. If None, uses trainer's rank.",
        ),
    ] = None

    alpha: Annotated[
        float,
        Field(
            ge=0,
            description="LoRA alpha for this run.",
        ),
    ] = 32.0


class ModelConfig(BaseModelConfig):
    """Extended model configuration with per-run LoRA settings."""

    lora: Annotated[
        LoRAConfig | None,
        Field(
            description="LoRA configuration. If None, LoRA is not used.",
        ),
    ] = None


class TemperatureScheduleConfig(BaseConfig):
    """Configures temperature scheduling over training steps. Use this OR sampling.temperature, not both."""

    type: Annotated[
        Literal["linear", "cosine"],
        Field(
            description="Schedule shape. Linear interpolates linearly; cosine uses smooth, monotonic curve.",
        ),
    ] = "linear"

    start_temperature: Annotated[
        float,
        Field(
            ge=0,
            description="Temperature at step 0.",
        ),
    ]

    end_temperature: Annotated[
        float,
        Field(
            ge=0,
            description="Temperature at final step.",
        ),
    ]

    total_steps: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of steps to reach end_temperature. Defaults to orchestrator max_steps if None.",
        ),
    ] = None


class SamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model for training. Largely follows the vLLM sampling parameters."""

    temperature: Annotated[
        float | None,
        Field(
            ge=0,
            description="Constant temperature for sampling. Defaults to 1.0 if neither this nor temp_scheduler is set. Cannot be set together with temp_scheduler.",
        ),
    ] = None

    temp_scheduler: Annotated[
        TemperatureScheduleConfig | None,
        Field(
            description="Temperature schedule over training steps. Set this OR temperature, not both.",
        ),
    ] = None

    repetition_penalty: Annotated[
        float,
        Field(
            ge=0,
            description="Penalty for repeating tokens. Values > 1.0 discourage repetition, values < 1.0 encourage repetition, and 1.0 means no penalty.",
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

    # Strictly speaking, extra_body is not a sampling parameter, but it is the
    # easiest way to pass arbitrary extra parameters to the server via verifiers
    extra_body: Annotated[
        dict[str, Any],
        Field(
            description="Extra body to pass with each request to the inference server. By default, it is set to an empty dictionary.",
        ),
    ] = {}


class EvalSamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model for evaluation. Largely follows the vLLM sampling parameters."""

    temperature: Annotated[
        float | None,
        Field(
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    repetition_penalty: Annotated[
        float | None,
        Field(
            ge=0,
            description="Penalty for repeating tokens. Values > 1.0 discourage repetition, values < 1.0 encourage repetition, and 1.0 means no penalty. Defaults to None, which means we fall back to the inference server's default value.",
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

    reasoning_effort: Annotated[
        Literal["minimal", "low", "medium", "high"] | None,
        Field(
            description="Constrains effort on reasoning for reasoning models. Currently supported values are minimal, low, medium, and high. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    seed: Annotated[
        int | None,
        Field(
            description="Random seed to use for sampling. If None, no seeding is used. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    # Strictly speaking, extra_body is not a sampling parameter, but it is the
    # easiest way to pass arbitrary extra parameters to the server via verifiers
    extra_body: Annotated[
        dict[str, Any],
        Field(
            description="Extra body to use for the OpenAI API. By default, it is set to an empty dictionary.",
        ),
    ] = {}


class EvalSaveDiskConfig(BaseConfig):
    """Configures how to save the eval results to disk."""

    path: Annotated[
        Path | None,
        Field(
            description="The path to save the eval results to. If None, will default to <output_dir>/evals/<step_path>/<env_id> for online evals and the verifiers default for offline evals."
        ),
    ] = None


class EvalSaveHFConfig(BaseConfig):
    """Configures how to save the eval results to HF."""

    dataset_name: Annotated[
        str | None,
        Field(
            description="The name of the HF dataset to save the eval results to. If None, will auto-generate a name."
        ),
    ] = None

    dataset_subset: Annotated[
        str | None,
        Field(
            description="The subset name of the HF dataset to save the evaluation results. If None, will default to the environment ID.",
        ),
    ] = None

    dataset_split: Annotated[
        str | None,
        Field(
            description="The split name of the HF dataset to save the evaluation results. If None, will default to 'evals'.",
        ),
    ] = None

    private: Annotated[
        bool,
        Field(description="Whether to save the eval results to a private HF dataset."),
    ] = False


class EvalSaveConfig(BaseConfig):
    disk: EvalSaveDiskConfig | None = None
    hf: EvalSaveHFConfig | None = None
    env_hub: Annotated[
        bool,
        Field(
            description="Whether to push eval results to Prime Environment Hub. Automatically pushes all evaluated environments. Requires PRIME_API_KEY and authorization for the environments."
        ),
    ] = False
    stream: Annotated[
        bool,
        Field(
            description="Whether to save results incrementally as rollouts complete.",
        ),
    ] = False


class RetryConfig(BaseConfig):
    """Configures retry behavior for rollout generation."""

    max_attempts: Annotated[
        int,
        Field(
            ge=1,
            description="Maximum number of retry attempts.",
        ),
    ] = 10

    wait_multiplier: Annotated[
        float,
        Field(
            ge=0,
            description="Multiplier for exponential backoff wait time.",
        ),
    ] = 1.0

    wait_min: Annotated[
        float,
        Field(
            ge=0,
            description="Minimum wait time in seconds between retries.",
        ),
    ] = 1.0

    wait_max: Annotated[
        float,
        Field(
            ge=0,
            description="Maximum wait time in seconds between retries.",
        ),
    ] = 60.0

    reraise: Annotated[
        bool,
        Field(
            description="Whether to reraise the original exception after all retries are exhausted. If False, raises tenacity.RetryError instead.",
        ),
    ] = False


class EnvConfig(BaseConfig):
    """Configures an environment for training."""

    id: Annotated[str, Field(description="ID of the environment to use.")] = "reverse-text"
    args: Annotated[dict, Field(description="Arguments to pass to the environment.")] = {}
    name: Annotated[str | None, Field(description="Name of the environment to use.")] = None


class EvalEnvConfig(EnvConfig):
    """Configures an environment for evaluation."""

    num_examples: Annotated[
        int | None,
        Field(
            description="Number of examples to evaluate per environment. If not set, will use 'num_examples' from main config."
        ),
    ] = None
    rollouts_per_example: Annotated[
        int | None,
        Field(
            description="Number of samples to generate per example for each environment. If not set, will use 'rollouts_per_example' from main config."
        ),
    ] = None

    skip_first: Annotated[
        int,
        Field(
            description="Number of examples to skip from the beginning of the dataset.",
        ),
    ] = 0


class ValConfig(BaseConfig):
    """Configures the validation of the model."""

    num_examples: Annotated[
        int, Field(ge=1, description="Number of examples to use for validation. If -1, will use all examples.")
    ] = 16
    rollouts_per_example: Annotated[
        int, Field(ge=1, description="Number of samples to generate per example for validation.")
    ] = 1
    interval: Annotated[int, Field(description="Interval at which to validate the model.")] = 10


class EvalConfig(BaseConfig):
    """Configures evaluation using verifiers environments."""

    env: list[EvalEnvConfig] = [EvalEnvConfig()]
    sampling: EvalSamplingConfig = Field(
        default_factory=EvalSamplingConfig,
        description="Shared sampling configuration for evals; can differ from training sampling.",
    )
    save: EvalSaveConfig = Field(
        default_factory=EvalSaveConfig,
        description="Configures how to save the eval results.",
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Configures retry behavior for rollout generation.",
    )
    num_examples: Annotated[int, Field(description="Number of examples to evaluate per environment.")] = -1
    rollouts_per_example: Annotated[
        int, Field(ge=1, description="Number of samples to generate per example for each environment.")
    ] = 1
    reasoning_field: Annotated[
        str,
        Field(
            description="The field in the raw model response that contains the reasoning content. Defaults to 'reasoning_content', which is the default for vLLM when serving a model with a reasoning parser. Other APIs (e.g. DeepSeek, GLM, etc.) may use different field names.",
        ),
    ] = "reasoning_content"
    per_rollout: Annotated[
        bool,
        Field(
            description="Schedule rollouts individually instead of as groups. Enables live progress updates and per-rollout resume, but incompatible with group-based rubrics.",
        ),
    ] = False


class OnlineEvalConfig(EvalConfig):
    """Configures online evaluation."""

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Interval at which to evaluate the model.",
        ),
    ] = 100

    eval_base_model: Annotated[
        bool,
        Field(
            description="Whether to evaluate the base model we are training on.",
        ),
    ] = True

    skip_eval_on_resume: Annotated[
        bool,
        Field(
            validation_alias=AliasChoices("skip_eval_on_resume", "skip_eval_on_restart"),
            description=(
                "If True and resuming the orchestrator from a checkpoint, skip the (potentially redundant) "
                "online eval that would otherwise run immediately at the resumed checkpoint step."
            ),
        ),
    ] = True


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the orchestrator."""

    interval: Annotated[int | None, Field(ge=1, description="Interval at which to save the checkpoint.")] = None

    resume_step: Annotated[
        int | None,
        Field(
            ge=-1,
            description="Step to resume orchestrator from. If None, will start from scratch. If -1, will restart from latest checkpoint available.",
        ),
    ] = None

    wait_for_weights_timeout: Annotated[
        int | None,
        Field(
            ge=1,
            description="When resuming, wait up to this many seconds for the weight directory to appear. Useful when the orchestrator restarts while the trainer is still saving weights. If None (default), fail immediately if weights are not found.",
        ),
    ] = None

    keep_last: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency.",
        ),
    ] = None

    keep_interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep checkpoints at every N steps permanently (e.g., keep_interval=100 keeps step 100, 200, ...). If None, no interval-based keeping.",
        ),
    ] = None

    skip_progress: Annotated[
        bool,
        Field(
            description="Whether to skip loading the progress from checkpoint.",
        ),
    ] = False

    skip_buffer: Annotated[
        bool,
        Field(
            description="Whether to skip loading the buffer from checkpoint.",
        ),
    ] = False


class BufferConfig(BaseConfig):
    """Configures the buffer for the orchestrator."""

    seed: Annotated[
        int | None,
        Field(
            description="Random seed to use for the buffer. If set, the sampling from the buffer will be deterministic.",
        ),
    ] = None

    env_ratios: Annotated[
        list[float] | None,
        Field(
            description=(
                "Ratios for sampling from each environment. "
                "If None, samples uniformly across all available problems (not environments)."
            ),
        ),
    ] = None

    easy_threshold: Annotated[
        float | None,
        Field(
            description="Threshold for easy difficulty classification. If average reward >= this threshold, mark as easy.",
        ),
    ] = None

    hard_threshold: Annotated[
        float | None,
        Field(
            description="Threshold for hard difficulty classification. If average reward <= this threshold, mark as hard.",
        ),
    ] = None

    easy_fraction: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Fraction of easy problems to convert to normal when resuming or starting training. Only problems with difficulty 'normal' are sampled.",
        ),
    ] = 0.0

    hard_fraction: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Fraction of hard problems to convert to normal when resuming or starting training. Only problems with difficulty 'normal' are sampled.",
        ),
    ] = 0.0

    online_difficulty_filtering: Annotated[
        bool,
        Field(
            description="Whether to filter rollouts based on difficulty. If True, rollouts with average reward 0.0 or 1.0 are not added to the buffer.",
        ),
    ] = False

    hash_keys: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Keys to use for computing example hashes. Will be used to match examples from buffer checkpoints and determine buffer resume behavior.",
        ),
    ] = ["task", "prompt"]

    skip_verification: Annotated[
        bool,
        Field(
            description=(
                "Whether to skip verification of rollouts using the environment's rubric. "
                "If True, rewards are always set to 0, online_difficulty_filtering is disabled, "
                "and easy/hard thresholds are not used."
            ),
        ),
    ] = False

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.easy_threshold is not None and self.hard_threshold is not None:
            assert self.easy_threshold > self.hard_threshold, "easy_threshold must be greater than hard_threshold."
        return self

    @model_validator(mode="after")
    def validate_env_ratios(self):
        if self.env_ratios is not None:
            assert all(ratio > 0 for ratio in self.env_ratios), "All env_ratios must be positive."
        return self

    @model_validator(mode="after")
    def validate_skip_verification(self):
        """Validate that skip_verification is not used with reward-dependent features."""
        if self.skip_verification:
            if self.online_difficulty_filtering:
                raise ValueError(
                    "skip_verification cannot be True when online_difficulty_filtering is True. "
                    "These features depend on rewards which are disabled when skip_verification=True."
                )
            if self.easy_threshold is not None:
                raise ValueError(
                    "skip_verification cannot be True when easy_threshold is set. "
                    "Easy threshold depends on rewards which are disabled when skip_verification=True."
                )
            if self.hard_threshold is not None:
                raise ValueError(
                    "skip_verification cannot be True when hard_threshold is set. "
                    "Hard threshold depends on rewards which are disabled when skip_verification=True."
                )
        return self


class AdvantageConfig(BaseConfig):
    length_weighted_mean: bool = False


class FileSystemWeightBroadcastConfig(BaseModel):
    """Configures the filesystem weight broadcast."""

    type: Literal["filesystem"] = "filesystem"


class NCCLWeightBroadcastConfig(BaseModel):
    """Configures the NCCL weight broadcast."""

    type: Literal["nccl"] = "nccl"

    host: Annotated[str, Field(description="The host to use for the NCCL broadcast.")] = "localhost"
    port: Annotated[int, Field(description="The port to use for the NCCL broadcast.")] = 29501
    timeout: Annotated[int, Field(description="The timeout in seconds to use for the NCCL broadcast.")] = 1200


WeightBroadcastConfigType: TypeAlias = FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig


class TeacherModelConfig(BaseConfig):
    """Configures the teacher model for computing teacher logprobs (e.g. for distillation)."""

    client: Annotated[
        ClientConfig,
        Field(description="The OAI client configuration for the teacher model."),
    ] = ClientConfig()

    model: Annotated[
        ModelConfig,
        Field(description="The model configuration for the teacher model."),
    ] = ModelConfig()


class OrchestratorConfig(BaseSettings):
    """Configures the orchestrator for RL training."""

    # The OAI client configuration
    client: ClientConfig = ClientConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The optimizer configuration (per-run LR for multi-run training)
    optim: OptimizerConfig = OptimizerConfig()

    # The teacher model configuration (optional)
    teacher_model: Annotated[
        TeacherModelConfig | None,
        Field(
            description="The teacher model configuration for computing teacher logprobs (e.g. for distillation). "
            "If provided, teacher logprobs will be computed using the specified model. "
            "If None, no teacher model will be used."
        ),
    ] = None

    # The sampling configuration
    sampling: SamplingConfig = SamplingConfig()

    # The environment configuration
    env: list[EnvConfig] = [EnvConfig()]

    # The evaluation configuration
    eval: OnlineEvalConfig | None = None

    # Data buffer configuration
    buffer: BufferConfig = BufferConfig()

    # The advantage configuration
    advantage: AdvantageConfig | None = AdvantageConfig()

    # The logging configuration
    log: LogConfig = LogConfig()

    # The wandb configuration
    wandb: WandbWithExtrasConfig | None = None

    # The prime monitor configuration
    prime_monitor: PrimeMonitorConfig | None = None

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    # Whether to reset inference weights to base model when starting from scratch
    reload_weights_on_start: Annotated[
        bool,
        Field(description="Whether to reset inference weights to the base model when starting from scratch."),
    ] = True

    # The validation configuration
    val: ValConfig | None = None

    weight_broadcast: Annotated[WeightBroadcastConfigType, Field(discriminator="type")] = (
        FileSystemWeightBroadcastConfig()
    )

    rollout_transport: Annotated[TransportConfigType, Field(discriminator="type")] = FileSystemTransportConfig()

    trajectory_strategy: Annotated[
        Literal["interleaved", "branching"],
        Field(
            description="Strategy to use for building training examples from multi-turn rollouts. If interleaved, will try to concatenate consecutive trajectory steps into a single training example. If branching, will create a separate training example for each trajectory step."
        ),
    ] = "interleaved"

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs/run_default")

    max_concurrent: Annotated[
        int | None,
        Field(
            description="Maximum number of concurrent rollouts to generate and score per-environment. If None, will not limit concurrency.",
        ),
    ] = None

    tasks_per_minute: Annotated[
        int | None,
        Field(
            ge=1,
            description="Rate limit for tasks per environment worker, in tasks per minute. Recommended for sandbox-backed environments to prevent sandbox-not-ready errors during autoscaling. When set to None, no rate limiting is applied. Note: with multiple workers, the effective total rate equals workers × this value.",
        ),
    ] = None

    workers_per_env: Annotated[
        int,
        Field(
            ge=1,
            description="Number of worker subprocesses to spawn per environment. Multiple workers enable isolation of event loop lag - if one worker slows down, others continue at full speed. Uses least-pending routing to distribute load.",
        ),
    ] = 1

    max_env_worker_restarts: Annotated[
        int,
        Field(
            ge=-1,
            description="Maximum number of automatic restarts for an environment worker that dies unexpectedly. After this limit, the orchestrator will crash. Set to -1 for unlimited restarts.",
        ),
    ] = 5

    batch_size: Annotated[int, Field(ge=1, description="Number of samples to train on per step.")] = 128

    oversampling_factor: Annotated[
        float,
        Field(
            ge=1,
            description="Factor by which to oversample the batch. Will lead to more in-flight group rollout requests at the same time.",
        ),
    ] = 1.0

    rollouts_per_example: Annotated[
        int,
        Field(
            ge=1,
            description="Number of output sequences to return per example during training.",
        ),
    ] = 1

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

    max_off_policy_steps: Annotated[
        int,
        Field(
            ge=0,
            description="Maximum number of policies that are allowed to generate a single rollout. Rollouts that are generated from more than `max_off_policy_steps` steps ahead of training will be discarded. Higher values yield better throughput, but lead to more off-policyness in training.",
        ),
    ] = 8

    max_async_level: Annotated[
        int,
        Field(
            ge=0,
            description="Maximum number of steps the inference can be ahead of training. If 0, will degenerate to synchronous on-policy RL. If >=1, training and inference will be overlapped.",
        ),
    ] = 1

    strict_async_level: Annotated[
        bool,
        Field(
            description="Whether to strictly enforce the max async level. If True, will always ensure that the policy used for generating rollouts is exactly `max_async_level` steps ahead of training. If False, any policy that is at most `max_async_level` steps ahead of training is allowed, i.e. we always use the latest available policy.",
        ),
    ] = False

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5, max async level to ~infinity and disable W&B.",
        ),
    ] = False

    seed: Annotated[int | None, Field(description="Random seed for the orchestrator.")] = 42

    heartbeat: Annotated[
        HeartbeatConfig | None, Field(description="The heartbeat config for monitoring training progress.")
    ] = None

    @model_validator(mode="after")
    def validate_max_concurrent(self):
        min_concurrent = self.rollouts_per_example * (self.workers_per_env or 1)
        if self.max_concurrent is not None and self.max_concurrent < min_concurrent:
            raise ValueError(
                f"max_concurrent must be at least rollouts_per_example * workers_per_env ({min_concurrent})"
            )
        return self

    @model_validator(mode="after")
    def nccl_max_async_level(self):
        if self.weight_broadcast.type == "nccl":
            if not self.max_async_level == 1:
                raise ValueError("max_async_level must be 1 for NCCL broadcast")
        return self

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.rollouts_per_example != 0:
            raise ValueError("Batch size must be divisible by the number of samples per problem")
        return self

    @model_validator(mode="after")
    def validate_env_ratios(self):
        if self.buffer.env_ratios is not None:
            assert len(self.buffer.env_ratios) == len(self.env), "env_ratios length must match number of environments"
        return self

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.max_steps = 4  # Run for 1 warmup step + 3 evaluation steps
            self.max_async_level = int(1e9)  # Never wait for RL weight checkpoints

            # Disable evaluation
            self.eval = None
            if self.wandb:
                self.wandb.log_extras = None
            if self.prime_monitor:
                self.prime_monitor.log_extras = None

        return self

    @model_validator(mode="after")
    def validate_temperature_config(self):
        has_temp = self.sampling.temperature is not None
        has_scheduler = self.sampling.temp_scheduler is not None

        if has_temp and has_scheduler:
            raise ValueError("Set either sampling.temperature OR sampling.temp_scheduler, not both")

        # Default to temperature=1.0 if neither is set
        if not has_temp and not has_scheduler:
            self.sampling.temperature = 1.0

        if has_scheduler:
            schedule = self.sampling.temp_scheduler
            if schedule.total_steps is None and self.max_steps is None:
                raise ValueError("temp_scheduler.total_steps must be set when max_steps is None")

        return self
