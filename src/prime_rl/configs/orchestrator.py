import math
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from prime_rl.configs.shared import (
    BaseModelConfig,
    ClientConfig,
    FileSystemTransportConfig,
    HeartbeatConfig,
    LogConfig,
    PrimeMonitorConfig,
    TransportConfig,
    WandbWithExtrasConfig,
)
from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger


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
        float | None,
        Field(
            ge=0,
            description="LoRA alpha for this run. If None, uses trainer's alpha.",
        ),
    ] = None


class ModelConfig(BaseModelConfig):
    """Extended model configuration with per-run LoRA settings."""

    lora: Annotated[
        LoRAConfig | None,
        Field(
            description="LoRA configuration. If None, LoRA is not used.",
        ),
    ] = None


class TrainSamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model for training. Largely follows the vLLM sampling parameters."""

    temperature: Annotated[
        float,
        Field(
            ge=0,
            description="Temperature for sampling.",
        ),
    ] = 1.0

    repetition_penalty: Annotated[
        float,
        Field(
            ge=0,
            description="Penalty for repeating tokens. Values > 1.0 discourage repetition, values < 1.0 encourage repetition, and 1.0 means no penalty.",
        ),
    ] = 1.0

    max_completion_tokens: Annotated[
        int | None,
        Field(
            validation_alias=AliasChoices("max_completion_tokens", "max_tokens"),
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

    def to_sampling_args(self) -> dict[str, Any]:
        """Convert to OAI-compatible sampling args dict, omitting None values."""
        # Top-level OAI params
        args: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": 1.0,
            "logprobs": True,
        }
        if self.max_completion_tokens is not None:
            args["max_completion_tokens"] = self.max_completion_tokens
        if self.seed is not None:
            args["seed"] = self.seed

        # vLLM extra_body params
        extra_body = dict(self.extra_body)
        if self.min_tokens > 0:
            extra_body["min_tokens"] = self.min_tokens
        if self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if extra_body:
            args["extra_body"] = extra_body

        return args

    @model_validator(mode="before")
    @classmethod
    def _deprecate_max_tokens(cls, data: Any) -> Any:
        if isinstance(data, dict) and "max_tokens" in data and "max_completion_tokens" not in data:
            get_logger().warning(
                "'max_tokens' is deprecated, use 'max_completion_tokens' instead. "
                "Auto-translating for now, but this will be removed in a future release."
            )
        return data


class EvalSamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model for evaluation.

    All sampling fields default to None, meaning the inference server's own
    default is used. Only explicitly set fields are forwarded.
    """

    temperature: Annotated[
        float | None,
        Field(ge=0, description="Sampling temperature. None defers to the inference server default."),
    ] = None

    repetition_penalty: Annotated[
        float | None,
        Field(ge=0, description="Repetition penalty. None defers to the inference server default."),
    ] = None

    top_p: Annotated[
        float | None,
        Field(description="Nucleus sampling threshold. None defers to the inference server default."),
    ] = None

    top_k: Annotated[
        int | None,
        Field(description="Top-k sampling. None defers to the inference server default."),
    ] = None

    min_p: Annotated[
        float | None,
        Field(ge=0, description="Min-p sampling threshold. None defers to the inference server default."),
    ] = None

    max_completion_tokens: Annotated[
        int | None,
        Field(
            validation_alias=AliasChoices("max_completion_tokens", "max_tokens"),
            description="Maximum output tokens per turn. None defers to the inference server default.",
        ),
    ] = None

    min_tokens: Annotated[
        int | None,
        Field(ge=0, description="Minimum output tokens per sequence. None defers to the inference server default."),
    ] = None

    reasoning_effort: Annotated[
        Literal["minimal", "low", "medium", "high"] | None,
        Field(description="Reasoning effort constraint for reasoning models."),
    ] = None

    seed: Annotated[
        int | None,
        Field(description="Random seed for sampling. None means no seeding."),
    ] = None

    extra_body: Annotated[
        dict[str, Any],
        Field(description="Extra body parameters forwarded to the inference server."),
    ] = {}

    def to_sampling_args(self) -> dict[str, Any]:
        """Convert to OAI-compatible sampling args dict. Only includes non-None fields."""
        args: dict[str, Any] = {}
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.top_p is not None:
            args["top_p"] = self.top_p
        if self.max_completion_tokens is not None:
            args["max_completion_tokens"] = self.max_completion_tokens
        if self.reasoning_effort is not None:
            args["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            args["seed"] = self.seed

        extra_body = dict(self.extra_body)
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.min_tokens is not None:
            extra_body["min_tokens"] = self.min_tokens
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if extra_body:
            args["extra_body"] = extra_body

        return args

    @model_validator(mode="before")
    @classmethod
    def _deprecate_max_tokens(cls, data: Any) -> Any:
        if isinstance(data, dict) and "max_tokens" in data and "max_completion_tokens" not in data:
            get_logger().warning(
                "'max_tokens' is deprecated, use 'max_completion_tokens' instead. "
                "Auto-translating for now, but this will be removed in a future release."
            )
        return data


class EnvConfig(BaseConfig):
    """Base environment configuration."""

    id: Annotated[
        str,
        Field(
            description="Registered verifiers environment ID (e.g. 'math-env', 'primeintellect/math-env'). May include an @version suffix for installation."
        ),
    ] = "reverse-text"

    name: Annotated[
        str | None,
        Field(
            description="Display name for this environment in logs, metrics, and buffer keys. Defaults to the id (without @version). Must be unique across all envs in the same group."
        ),
    ] = None

    args: Annotated[
        dict,
        Field(
            description="Keyword arguments forwarded to vf.load_environment. See the environment's docstring for accepted args."
        ),
    ] = {}

    extra_env_kwargs: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Extra kwargs passed to an env (e.g. seq_len, max_total_completion_tokens). This field is auto-populated on the orchestrator for all envs. It is generally NOT recommended for this field to be overriden by the user. It's main use case is to match the extra_env_kwargs when running an env in an isolated environment server."
            ),
        ),
    ] = {}

    address: Annotated[
        str | None,
        Field(
            description="ZMQ address of an external env server (e.g. 'tcp://host:5000'). When set, the orchestrator connects to this server instead of spawning one. When None (default), a subprocess env server is spawned automatically."
        ),
    ] = None

    num_workers: Annotated[
        int | Literal["auto"],
        Field(
            description="Number of worker processes for the spawned env server. 'auto' scales to 1 worker per 256 concurrent rollouts. Ignored when address is set (external server)."
        ),
    ] = "auto"

    ratio: Annotated[
        float | None,
        Field(
            gt=0,
            description="Sampling weight for this environment in the buffer. When None for all envs, samples uniformly across all available problems. When set, must be set on all envs — values are relative weights normalized to probabilities (e.g. [1, 1] and [0.5, 0.5] are equivalent).",
        ),
    ] = None

    max_retries: Annotated[
        int,
        Field(ge=0, description="Number of times the env server retries a failed rollout before returning an error."),
    ] = 0

    max_total_completion_tokens: Annotated[
        int,
        Field(
            description=(
                "Maximum total completion tokens across all turns in a multi-turn rollout. "
                "Set to -1 (default) to disable. Auto-populated into extra_env_kwargs."
            ),
        ),
    ] = -1

    @property
    def stripped_id(self) -> str:
        """Environment ID without the @version suffix."""
        return self.id.split("@")[0]

    @property
    def resolved_name(self) -> str:
        return self.name or self.stripped_id

    @model_validator(mode="after")
    def validate_env_name(self):
        if self.resolved_name == "all":
            raise ValueError(
                'Environment name "all" is reserved for global metric aggregation. Use a different name or id.'
            )
        return self

    @model_validator(mode="after")
    def resolve_max_total_completion_tokens(self):
        self.extra_env_kwargs["max_total_completion_tokens"] = self.max_total_completion_tokens
        return self


class TrainEnvConfig(EnvConfig):
    """Configures a training environment."""

    sampling: Annotated[
        TrainSamplingConfig,
        Field(
            description="Per-env sampling overrides. Unset fields inherit from the group-level train sampling config.",
        ),
    ] = TrainSamplingConfig()


class EvalEnvConfig(EnvConfig):
    """Configures an evaluation environment."""

    sampling: Annotated[
        EvalSamplingConfig,
        Field(
            description="Per-env sampling overrides. Unset fields inherit from the group-level eval sampling config.",
        ),
    ] = EvalSamplingConfig()

    num_examples: Annotated[
        int,
        Field(
            description="Number of eval examples to sample from the dataset. Set to -1 to use all available examples."
        ),
    ] = -1

    rollouts_per_example: Annotated[
        int,
        Field(
            ge=1,
            description="Number of rollouts generated per example. Used for pass@k estimation (e.g. rollouts_per_example=8 enables pass@1 through pass@8).",
        ),
    ] = 1

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Per-env eval interval. If unset, inherits from the group-level eval interval.",
        ),
    ] = 100


class TrainConfig(BaseConfig):
    """Configures training environments and their shared sampling settings."""

    env: list[TrainEnvConfig] = [TrainEnvConfig()]

    sampling: TrainSamplingConfig = TrainSamplingConfig()

    num_workers: Annotated[
        int | Literal["auto"],
        Field(
            description="Default number of worker processes for env servers. Can be overridden per env.",
        ),
    ] = "auto"

    max_retries: Annotated[
        int,
        Field(ge=0, description="Default number of retries for failed rollouts. Can be overridden per env."),
    ] = 0

    @model_validator(mode="after")
    def resolve_env_defaults(self):
        """Resolve per-env overrides: inherit group-level sampling, num_workers, and max_retries."""
        group_sampling = self.sampling.model_dump()
        for env in self.env:
            if "sampling" not in env.model_fields_set:
                env.sampling = TrainSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | env.sampling.model_dump(exclude_unset=True)
                env.sampling = TrainSamplingConfig(**merged)
            if "num_workers" not in env.model_fields_set:
                env.num_workers = self.num_workers
            if "max_retries" not in env.model_fields_set:
                env.max_retries = self.max_retries
        return self

    @model_validator(mode="after")
    def validate_unique_env_names(self):
        env_names = [env.resolved_name for env in self.env]
        duplicates = [n for n in env_names if env_names.count(n) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate training environment names: {set(duplicates)}. Each env must have a unique name."
            )
        return self

    @model_validator(mode="after")
    def validate_env_ratios(self):
        ratios = [env.ratio for env in self.env]
        if all(r is None for r in ratios):
            return self
        if any(r is None for r in ratios):
            raise ValueError("Either all envs must have a ratio or none of them. Got a mix of set and unset ratios.")
        return self


class EvalConfig(BaseConfig):
    """Configures evaluation using verifiers environments."""

    env: list[EvalEnvConfig] = [EvalEnvConfig()]

    sampling: EvalSamplingConfig = Field(
        default_factory=EvalSamplingConfig,
        description="Shared sampling configuration for evals; can differ from training sampling.",
    )

    num_examples: Annotated[
        int,
        Field(
            description="Default number of eval examples per environment. Set to -1 to use all. Can be overridden per env."
        ),
    ] = -1

    rollouts_per_example: Annotated[
        int,
        Field(ge=1, description="Default number of rollouts per example. Can be overridden per env."),
    ] = 1

    num_workers: Annotated[
        int | Literal["auto"],
        Field(
            description="Default number of worker processes for env servers. Can be overridden per env.",
        ),
    ] = "auto"

    max_retries: Annotated[
        int,
        Field(ge=0, description="Default number of retries for failed rollouts. Can be overridden per env."),
    ] = 0

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Interval at which to evaluate the model.",
        ),
    ] = 100

    @model_validator(mode="after")
    def resolve_env_defaults(self):
        """Resolve per-env overrides: inherit group-level sampling, num_workers, max_retries, num_examples, rollouts_per_example, and interval. Then resolve auto num_workers."""
        group_sampling = self.sampling.model_dump()
        for env in self.env:
            if "sampling" not in env.model_fields_set:
                env.sampling = EvalSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | env.sampling.model_dump(exclude_unset=True)
                env.sampling = EvalSamplingConfig(**merged)
            if "num_examples" not in env.model_fields_set:
                env.num_examples = self.num_examples
            if "rollouts_per_example" not in env.model_fields_set:
                env.rollouts_per_example = self.rollouts_per_example
            if "interval" not in env.model_fields_set:
                env.interval = self.interval
            if "num_workers" not in env.model_fields_set:
                env.num_workers = self.num_workers
            if "max_retries" not in env.model_fields_set:
                env.max_retries = self.max_retries
            # Resolve auto num_workers now that num_examples and rollouts_per_example are set
            if env.num_workers == "auto":
                if env.num_examples == -1:
                    env.num_workers = 4
                else:
                    max_concurrent = env.num_examples * env.rollouts_per_example
                    env.num_workers = max(1, math.ceil(max_concurrent / 256))
        return self

    @model_validator(mode="after")
    def validate_unique_env_names(self):
        env_names = [env.resolved_name for env in self.env]
        duplicates = [n for n in env_names if env_names.count(n) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate evaluation environment names: {set(duplicates)}. Each env must have a unique name."
            )
        return self

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

    cancel_inflight_rollouts_on_eval: Annotated[
        bool,
        Field(
            description="Whether to cancel in-flight training rollouts before starting online evals. This is useful to avoid congestion (e.g. do not have training + eval rollouts happening at the same time) but leads to slower training steps as rollouts get cancelled and the pipeline has to fill up after each eval",
        ),
    ] = False


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
    ] = ["env_name", "prompt"]

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.easy_threshold is not None and self.hard_threshold is not None:
            assert self.easy_threshold > self.hard_threshold, "easy_threshold must be greater than hard_threshold."
        return self


class DefaultAdvantageConfig(BaseModel):
    """Config for the default advantage."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["default"] = "default"
    length_shaping: Annotated[
        bool,
        Field(
            description=(
                "Enable correctness-gated length shaping. In mixed groups, shorter correct rollouts get "
                "amplified advantage (up to 2x), longer correct rollouts are unchanged, incorrect untouched. "
                "In all-correct groups, below-average-length rollouts get advantage in [0, 1], others get 0."
            )
        ),
    ] = False


class CustomAdvantageConfig(BaseModel):
    """Config for a custom external advantage function."""

    type: Literal["custom"] = "custom"
    import_path: Annotated[
        str, Field(description="Import path to the advantage function (e.g., 'my_module.my_advantage')")
    ]
    kwargs: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Kwargs to pass to the advantage function")
    ]


AdvantageConfig: TypeAlias = Annotated[
    DefaultAdvantageConfig | CustomAdvantageConfig,
    Field(discriminator="type"),
]


class GibberishFilterConfig(BaseModel):
    """Flags rare tokens generated at high entropy (Section 5.2, https://arxiv.org/abs/2510.02387)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["gibberish"] = "gibberish"
    enforce: Annotated[
        bool,
        Field(
            description="If True, skip detected rollouts entirely so they are not sent to the trainer. If False, only track detection metrics."
        ),
    ] = False
    token_id_threshold: Annotated[
        int,
        Field(description="Token IDs above this are candidates for gibberish. BPE tokens are sorted by merge order."),
    ] = 100_000
    logprob_offset: Annotated[
        float,
        Field(description="Offset from uniform distribution logprob. Threshold = -log(vocab_size) - logprob_offset."),
    ] = 2.0


class RepetitionFilterConfig(BaseModel):
    """Flags rollouts where the model gets stuck in a repetition loop, emitting high-confidence tokens
    for an extended stretch. A rollout is flagged when `window` consecutive tokens are each sampled
    with probability above `prob_threshold`. (Section 3.2, https://arxiv.org/abs/2506.13585)"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["repetition"] = "repetition"
    enforce: Annotated[
        bool,
        Field(
            description="If True, skip detected rollouts entirely so they are not sent to the trainer. If False, only track detection metrics."
        ),
    ] = False
    window: Annotated[
        int,
        Field(ge=1, description="Number of consecutive high-probability steps before flagging."),
    ] = 3_000
    prob_threshold: Annotated[
        float,
        Field(
            gt=0,
            le=1,
            description="Tokens sampled with probability above this are considered repetitive. Consecutive such tokens count toward the window.",
        ),
    ] = 0.99


class ZeroAdvantageFilterConfig(BaseModel):
    """Flags rollouts with zero advantage."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["zero_advantage"] = "zero_advantage"
    enforce: Annotated[
        bool,
        Field(
            description="If True, skip detected rollouts entirely so they are not sent to the trainer. If False, only track detection metrics."
        ),
    ] = True


FilterConfig: TypeAlias = Annotated[
    GibberishFilterConfig | RepetitionFilterConfig | ZeroAdvantageFilterConfig,
    Field(discriminator="type"),
]


class FileSystemWeightBroadcastConfig(BaseModel):
    """Configures the filesystem weight broadcast."""

    type: Literal["filesystem"] = "filesystem"


class NCCLWeightBroadcastConfig(BaseModel):
    """Configures the NCCL weight broadcast."""

    type: Literal["nccl"] = "nccl"

    host: Annotated[str, Field(description="The host to use for the NCCL broadcast.")] = "localhost"
    port: Annotated[int, Field(description="The port to use for the NCCL broadcast.")] = 29501
    timeout: Annotated[int, Field(description="The timeout in seconds to use for the NCCL broadcast.")] = 1200
    quantize_in_weight_transfer: Annotated[
        bool,
        Field(description="Use kernel-format FP8 quantized NCCL transfer for weight updates."),
    ] = False

    inference_world_size: Annotated[
        int,
        Field(
            ge=1,
            description="Total number of inference GPUs across all servers. Used by init_nccl_broadcast to compute per-server rank offsets.",
        ),
    ] = 1


WeightBroadcastConfig: TypeAlias = Annotated[
    FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig, Field(discriminator="type")
]


class OrchestratorExperimentalConfig(BaseConfig):
    """Experimental features for the orchestrator."""


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


class TeacherRolloutModelConfig(BaseConfig):
    """Configures an external teacher model used to generate rollout text."""

    client: Annotated[
        ClientConfig,
        Field(description="The OAI client configuration for rollout generation."),
    ] = ClientConfig()

    model: Annotated[
        ModelConfig,
        Field(description="The model configuration for rollout generation."),
    ] = ModelConfig()


class OrchestratorConfig(BaseConfig):
    """Configures the orchestrator for RL training."""

    # Training environments and sampling
    train: TrainConfig = TrainConfig()

    # The OAI client configuration
    client: ClientConfig = ClientConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The tokenizer configuration
    tokenizer: TokenizerConfig = TokenizerConfig()

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

    # External teacher rollout model configuration (optional)
    teacher_rollout_model: Annotated[
        TeacherRolloutModelConfig | None,
        Field(
            description=(
                "Optional external teacher model used for rollout generation. "
                "When set, rollouts are generated from this endpoint/model instead of the student inference server."
            ),
        ),
    ] = None

    # The evaluation configuration
    eval: EvalConfig | None = None

    # Data buffer configuration
    buffer: BufferConfig = BufferConfig()

    # The advantage configuration
    advantage: AdvantageConfig | None = DefaultAdvantageConfig()

    # Rollout filters (monitor by default, enforce optionally)
    filters: list[FilterConfig] = [GibberishFilterConfig(), RepetitionFilterConfig(), ZeroAdvantageFilterConfig()]

    # The logging configuration
    log: LogConfig = LogConfig()

    # The wandb configuration
    wandb: WandbWithExtrasConfig | None = None

    # The prime monitor configuration
    prime_monitor: PrimeMonitorConfig | None = None

    # Whether to collect inference server metrics (requires wandb)
    collect_inference_metrics: bool = True

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()

    rollout_transport: TransportConfig = FileSystemTransportConfig()

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs/run_default")

    tasks_per_minute: Annotated[
        int | None,
        Field(
            ge=1,
            description="Rate limit for tasks per environment worker, in tasks per minute. Recommended for sandbox-backed environments to prevent sandbox-not-ready errors during autoscaling. When set to None, no rate limiting is applied. Note: with multiple workers, the effective total rate equals workers × this value.",
        ),
    ] = None

    batch_size: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of samples to train on per step (rollout-based batching). Set this OR token_batch_size.",
        ),
    ] = None

    token_batch_size: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of tokens to train on per step (token-based batching). Set this OR batch_size.",
        ),
    ] = None

    oversampling_factor: Annotated[
        float | None,
        Field(
            ge=1,
            description=(
                "Rollout-mode batching only. Multiplier used to derive max_inflight_rollouts from batch_size "
                "when max_inflight_rollouts is unset."
            ),
        ),
    ] = None

    max_inflight_rollouts: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Maximum number of rollouts to keep in-flight. Required for token-based batching. "
                "If batch_size is set and this is unset, defaults to batch_size * oversampling_factor "
                "(or batch_size when oversampling_factor is unset)."
            ),
        ),
    ] = None

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

    use_token_client: Annotated[
        bool,
        Field(
            description="Whether to use the token-in-token-out (TITO) client for training across all environments. WARNING: Only use this if your environment has a linear history and the chat template has the extension property (i.e. no tokens are ever removed or inserted by the chat template)"
        ),
    ] = True

    env_install_prerelease: Annotated[
        bool,
        Field(
            description="Allow pre-release versions when installing environments (e.g. verifiers>=0.1.12.dev5). Passes --prerelease to prime env install."
        ),
    ] = False

    experimental: Annotated[
        OrchestratorExperimentalConfig,
        Field(description="Experimental features for the orchestrator."),
    ] = OrchestratorExperimentalConfig()

    @model_validator(mode="before")
    @classmethod
    def _env_to_train(cls, data: Any) -> Any:
        """Allow [[env]] and [sampling] as shorthand for [train] with [[train.env]] and [train.sampling]."""
        if not isinstance(data, dict):
            return data
        if "env" in data or "sampling" in data:
            train = data.setdefault("train", {})
            if isinstance(train, dict):
                if "env" in data:
                    get_logger().warning(
                        "'[[orchestrator.env]]' is deprecated, use '[[orchestrator.train.env]]' instead. "
                        "Auto-translating for now, but this will be removed in a future release."
                    )
                    train.setdefault("env", data.pop("env"))
                if "sampling" in data:
                    get_logger().warning(
                        "'[orchestrator.sampling]' is deprecated, use '[orchestrator.train.sampling]' instead. "
                        "Auto-translating for now, but this will be removed in a future release."
                    )
                    train.setdefault("sampling", data.pop("sampling"))
        return data

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def validate_unique_filter_types(self):
        types = [f.type for f in self.filters]
        if len(types) != len(set(types)):
            raise ValueError(f"Duplicate filter types: {types}. Each filter type may only appear once.")
        return self

    @model_validator(mode="after")
    def nccl_max_async_level(self):
        if self.weight_broadcast.type == "nccl":
            if not self.max_async_level == 1:
                raise ValueError("max_async_level must be 1 for NCCL broadcast")
        return self

    @model_validator(mode="after")
    def resolve_batching(self):
        has_rollout_batch = self.batch_size is not None
        has_token_batch = self.token_batch_size is not None

        if has_rollout_batch and has_token_batch:
            raise ValueError("Set exactly one of batch_size or token_batch_size")

        if not has_rollout_batch and not has_token_batch:
            self.batch_size = 128

        if has_token_batch:
            if self.oversampling_factor is not None:
                raise ValueError("oversampling_factor can only be set when batch_size is set")
            if self.max_inflight_rollouts is None:
                raise ValueError("max_inflight_rollouts must be set when token_batch_size is set")
        else:
            assert self.batch_size is not None
            if self.batch_size % self.rollouts_per_example != 0:
                raise ValueError("Batch size must be divisible by the number of samples per problem")
            if self.max_inflight_rollouts is not None and self.oversampling_factor is not None:
                expected_max_inflight_rollouts = int(self.batch_size * self.oversampling_factor)
                if self.max_inflight_rollouts != expected_max_inflight_rollouts:
                    raise ValueError("max_inflight_rollouts conflicts with oversampling_factor * batch_size")
            if self.max_inflight_rollouts is None:
                oversampling_factor = self.oversampling_factor if self.oversampling_factor is not None else 1.0
                self.max_inflight_rollouts = int(self.batch_size * oversampling_factor)

        if self.max_inflight_rollouts is not None and self.max_inflight_rollouts < self.rollouts_per_example:
            raise ValueError("max_inflight_rollouts must be at least the number of rollouts per example")

        # Resolve train env num_workers from max_inflight_rollouts
        for env_cfg in self.train.env:
            if env_cfg.num_workers == "auto":
                assert self.max_inflight_rollouts is not None
                env_cfg.num_workers = max(1, math.ceil(self.max_inflight_rollouts / 256))

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
    def resolve_env_config(self):
        """Populate extra_env_kwargs and vLLM sampling defaults from top-level fields."""
        is_vllm = self.teacher_rollout_model is None
        for env in self.train.env:
            env.extra_env_kwargs.update(max_seq_len=self.seq_len)
            if is_vllm:
                env.sampling.extra_body.setdefault("top_k", -1)
                env.sampling.extra_body.setdefault("min_p", 0.0)
                env.sampling.extra_body.setdefault("return_token_ids", True)
        return self
