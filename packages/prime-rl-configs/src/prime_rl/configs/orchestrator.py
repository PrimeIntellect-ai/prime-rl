import math
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AliasChoices, Field, model_validator

from prime_rl.configs.shared import (
    BaseModelConfig,
    ClientConfig,
    FileSystemTransportConfig,
    HeartbeatConfig,
    LogConfig,
    PrimeMonitorConfig,
    RendererConfig,
    TransportConfig,
    WandbWithExtrasConfig,
)
from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.utils.config import BaseConfig


class OptimizerConfig(BaseConfig):
    lr: float = Field(1e-4, ge=0)
    """Learning rate for this run (per-run override for multi-run training)."""


class LoRAConfig(BaseConfig):
    name: str | None = None
    """LoRA adapter name. If None, auto-generated from rank and alpha."""

    rank: int | None = Field(None, ge=1)
    """LoRA rank for this run. Must be ≤ trainer's max rank. If None, uses the trainer's rank."""

    alpha: float | None = Field(None, ge=0)
    """LoRA alpha for this run. If None, uses the trainer's alpha."""


class ModelConfig(BaseModelConfig):
    lora: LoRAConfig | None = None
    """Per-run LoRA configuration. If None, LoRA is disabled."""


class TrainSamplingConfig(BaseConfig):
    temperature: float = Field(1.0, ge=0)
    """Sampling temperature."""

    repetition_penalty: float = Field(1.0, ge=0)
    """Repetition penalty. Values > 1.0 discourage repetition, < 1.0 encourage it, 1.0 disables."""

    max_completion_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_completion_tokens", "max_tokens")
    )
    """Maximum output tokens per turn. If None, generates until max context length or EOS."""

    min_tokens: int = Field(0, ge=0)
    """Minimum output tokens per sequence."""

    seed: int | None = None
    """Random seed for sampling. If None, no seeding is used."""

    # Strictly speaking, extra_body is not a sampling parameter, but it is the
    # easiest way to pass arbitrary extra parameters to the server via verifiers
    extra_body: dict[str, Any] = {}
    """Extra body forwarded with each request to the inference server."""

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
            warnings.warn(
                "'max_tokens' is deprecated, use 'max_completion_tokens' instead. "
                "Auto-translating for now, but this will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )
        return data


class EvalSamplingConfig(BaseConfig):
    temperature: float | None = Field(None, ge=0)
    """Sampling temperature. None defers to the inference server default."""

    repetition_penalty: float | None = Field(None, ge=0)
    """Repetition penalty. None defers to the inference server default."""

    top_p: float | None = None
    """Nucleus sampling threshold. None defers to the inference server default."""

    top_k: int | None = None
    """Top-k sampling. None defers to the inference server default."""

    min_p: float | None = Field(None, ge=0)
    """Min-p sampling threshold. None defers to the inference server default."""

    max_completion_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_completion_tokens", "max_tokens")
    )
    """Maximum output tokens per turn. None defers to the inference server default."""

    min_tokens: int | None = Field(None, ge=0)
    """Minimum output tokens per sequence. None defers to the inference server default."""

    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    """Reasoning effort constraint for reasoning models."""

    seed: int | None = None
    """Random seed for sampling. None means no seeding."""

    extra_body: dict[str, Any] = {}
    """Extra body parameters forwarded to the inference server."""

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
            warnings.warn(
                "'max_tokens' is deprecated, use 'max_completion_tokens' instead. "
                "Auto-translating for now, but this will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )
        return data


class EnvConfig(BaseConfig):
    id: str = "reverse-text"
    """Registered verifiers environment ID (e.g. ``math-env``, ``primeintellect/math-env``). May include an ``@version`` suffix for installation."""

    name: str | None = None
    """Display name for this environment in logs, metrics, and buffer keys. Defaults to the ``id`` without ``@version``. Must be unique across all envs in the same group."""

    args: dict = {}
    """Keyword arguments forwarded to ``vf.load_environment``. See the environment's docstring for accepted args."""

    extra_env_kwargs: dict[str, Any] = {}
    """Extra kwargs passed to the env (e.g. ``seq_len``, ``max_total_completion_tokens``). Auto-populated by the orchestrator; user overrides are generally discouraged. The main use case is matching ``extra_env_kwargs`` when running an env in an isolated environment server."""

    address: str | None = None
    """ZMQ address of an external env server (e.g. ``tcp://host:5000``). When set, the orchestrator connects to this server instead of spawning one; when None, a subprocess env server is spawned automatically."""

    num_workers: int | Literal["auto"] = "auto"
    """Worker processes for the spawned env server. ``auto`` scales to 1 worker per 256 concurrent rollouts. Ignored when ``address`` is set."""

    ratio: float | None = Field(None, gt=0)
    """Sampling weight for this environment in the buffer. When None for all envs, samples uniformly across all available problems. When set, must be set on all envs — values are relative weights normalized to probabilities (e.g. [1, 1] and [0.5, 0.5] are equivalent)."""

    max_retries: int = Field(3, ge=0)
    """Times the env server retries a failed rollout before returning an error."""

    max_total_completion_tokens: int = -1
    """Maximum total completion tokens across all turns in a multi-turn rollout. ``-1`` disables. Auto-populated into ``extra_env_kwargs``."""

    timeout: float | None = Field(None, validation_alias=AliasChoices("timeout", "timeout_seconds"))
    """Per-rollout wall-clock timeout in seconds. None disables."""

    state_columns: list[str] = []
    """Extra ``State`` fields to persist into the saved rollout records (in addition to the always-saved ``trajectory`` and ``sampling_args``). Values must be JSON-serializable."""

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

    @model_validator(mode="after")
    def resolve_timeout(self):
        if self.timeout is not None:
            self.extra_env_kwargs["timeout_seconds"] = self.timeout
        return self


class TrainEnvConfig(EnvConfig):
    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Per-env sampling overrides. Unset fields inherit from the group-level train sampling config."""


class EvalEnvConfig(EnvConfig):
    sampling: EvalSamplingConfig = EvalSamplingConfig()
    """Per-env sampling overrides. Unset fields inherit from the group-level eval sampling config."""

    num_examples: int = -1
    """Eval examples to sample from the dataset. ``-1`` uses all available examples."""

    rollouts_per_example: int = Field(1, ge=1)
    """Rollouts generated per example. Used for pass@k estimation (e.g. ``rollouts_per_example=8`` enables pass@1 through pass@8)."""

    interval: int = Field(100, ge=1)
    """Per-env eval interval. If unset, inherits from the group-level eval interval."""


class TrainConfig(BaseConfig):
    env: list[TrainEnvConfig] = [TrainEnvConfig()]
    """Training environments."""

    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Shared training sampling configuration."""

    num_workers: int | Literal["auto"] = "auto"
    """Default worker processes for env servers. Can be overridden per env."""

    max_retries: int = Field(3, ge=0)
    """Default retries for failed rollouts. Can be overridden per env."""

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
    env: list[EvalEnvConfig] = [EvalEnvConfig()]
    """Evaluation environments."""

    sampling: EvalSamplingConfig = Field(default_factory=EvalSamplingConfig)
    """Shared eval sampling configuration; can differ from training sampling."""

    num_examples: int = -1
    """Default eval examples per environment. ``-1`` uses all. Can be overridden per env."""

    rollouts_per_example: int = Field(1, ge=1)
    """Default rollouts per example. Can be overridden per env."""

    num_workers: int | Literal["auto"] = "auto"
    """Default worker processes for env servers. Can be overridden per env."""

    max_retries: int = Field(3, ge=0)
    """Default retries for failed rollouts. Can be overridden per env."""

    interval: int = Field(100, ge=1)
    """Step interval at which to evaluate the model."""

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

    eval_base_model: bool = True
    """Evaluate the base model we are training on."""

    skip_eval_on_resume: bool = Field(
        True, validation_alias=AliasChoices("skip_eval_on_resume", "skip_eval_on_restart")
    )
    """When resuming the orchestrator from a checkpoint, skip the (potentially redundant) online eval that would otherwise run immediately at the resumed step."""

    cancel_inflight_rollouts_on_eval: bool = False
    """Cancel in-flight training rollouts before starting online evals. Avoids congestion (no training + eval rollouts at the same time) at the cost of slower training steps as the pipeline has to refill after each eval."""


class CheckpointConfig(BaseConfig):
    interval: int | None = Field(None, ge=1)
    """Step interval at which to save the orchestrator checkpoint."""

    resume_step: int | None = Field(None, ge=-1)
    """Step to resume the orchestrator from. None starts from scratch; ``-1`` resumes from the latest checkpoint available."""

    wait_for_weights_timeout: int | None = Field(None, ge=1)
    """When resuming, wait up to this many seconds for the weight directory to appear. Useful when the orchestrator restarts while the trainer is still saving weights. If None, fail immediately when weights are not found."""

    keep_last: int | None = Field(None, ge=1)
    """Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency."""

    keep_interval: int | None = Field(None, ge=1)
    """Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping."""

    skip_progress: bool = False
    """Skip loading the progress from checkpoint."""

    skip_buffer: bool = False
    """Skip loading the buffer from checkpoint."""


class BufferConfig(BaseConfig):
    seed: int | None = None
    """Random seed for the buffer. When set, sampling from the buffer is deterministic."""

    easy_threshold: float | None = None
    """Average-reward threshold above which a problem is classified ``easy``."""

    hard_threshold: float | None = None
    """Average-reward threshold below which a problem is classified ``hard``."""

    easy_fraction: float = Field(0.0, ge=0, le=1)
    """Fraction of easy problems to convert to ``normal`` when resuming or starting training. Only problems with difficulty ``normal`` are sampled."""

    hard_fraction: float = Field(0.0, ge=0, le=1)
    """Fraction of hard problems to convert to ``normal`` when resuming or starting training. Only problems with difficulty ``normal`` are sampled."""

    online_difficulty_filtering: bool = False
    """Filter rollouts based on difficulty. When True, rollouts with average reward 0.0 or 1.0 are not added to the buffer."""

    hash_keys: list[str] = Field(["env_name", "prompt"], min_length=1)
    """Keys used to compute example hashes. Used to match examples from buffer checkpoints and determine buffer resume behavior."""

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.easy_threshold is not None and self.hard_threshold is not None:
            assert self.easy_threshold > self.hard_threshold, "easy_threshold must be greater than hard_threshold."
        return self


class TokensLengthPenaltyConfig(BaseConfig):
    type: Literal["tokens"] = "tokens"

    completion_weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    """Weight on model completion tokens. Finite and non-negative."""

    tool_response_weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    """Weight on tool-response tokens (read from the rollout's ``*_total_tool_response_tokens`` harness metric; 0 if absent). Finite and non-negative."""


class TurnsLengthPenaltyConfig(BaseConfig):
    type: Literal["turns"] = "turns"


LengthPenaltyConfig: TypeAlias = Annotated[
    TokensLengthPenaltyConfig | TurnsLengthPenaltyConfig,
    Field(discriminator="type"),
]


class DefaultAdvantageConfig(BaseConfig):
    type: Literal["default"] = "default"

    length_penalty: LengthPenaltyConfig | None = None
    """Correctness-gated length penalty. ``tokens`` shapes by weighted token cost; ``turns`` shapes by trajectory turn count; None disables shaping. In mixed groups, lower-cost correct rollouts get amplified advantage (up to 2x), higher-cost correct rollouts are unchanged, incorrect untouched. In all-correct groups, below-average-cost rollouts get advantage in [0, 1], others get 0."""


class CustomAdvantageConfig(BaseConfig):
    type: Literal["custom"] = "custom"

    import_path: str
    """Import path to the advantage function (e.g. ``my_module.my_advantage``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the advantage function."""


AdvantageConfig: TypeAlias = Annotated[
    DefaultAdvantageConfig | CustomAdvantageConfig,
    Field(discriminator="type"),
]


# Flags rare tokens generated at high entropy (Section 5.2, https://arxiv.org/abs/2510.02387).
class GibberishFilterConfig(BaseConfig):
    type: Literal["gibberish"] = "gibberish"

    enforce: bool = False
    """When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics."""

    token_id_threshold: int = 100_000
    """Token IDs above this are candidates for gibberish. BPE tokens are sorted by merge order."""

    logprob_offset: float = 2.0
    """Offset from uniform-distribution logprob. Threshold = ``-log(vocab_size) - logprob_offset``."""


# Flags rollouts stuck in a repetition loop: emits high-confidence tokens for an extended stretch.
# Flagged when `window` consecutive tokens are each sampled with probability above `prob_threshold`.
# (Section 3.2, https://arxiv.org/abs/2506.13585)
class RepetitionFilterConfig(BaseConfig):
    type: Literal["repetition"] = "repetition"

    enforce: bool = False
    """When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics."""

    window: int = Field(3_000, ge=1)
    """Consecutive high-probability steps required to flag the rollout."""

    prob_threshold: float = Field(0.99, gt=0, le=1)
    """Tokens sampled with probability above this are considered repetitive. Consecutive such tokens count toward the window."""


# Flags rollouts with zero advantage.
class ZeroAdvantageFilterConfig(BaseConfig):
    type: Literal["zero_advantage"] = "zero_advantage"

    enforce: bool = True
    """When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics."""


FilterConfig: TypeAlias = Annotated[
    GibberishFilterConfig | RepetitionFilterConfig | ZeroAdvantageFilterConfig,
    Field(discriminator="type"),
]


class FileSystemWeightBroadcastConfig(BaseConfig):
    type: Literal["filesystem"] = "filesystem"


class NCCLWeightBroadcastConfig(BaseConfig):
    type: Literal["nccl"] = "nccl"

    host: str = "localhost"
    """Host for the NCCL broadcast rendezvous."""

    port: int = 29501
    """Port for the NCCL broadcast rendezvous."""

    timeout: int = 1200
    """Timeout in seconds for the NCCL broadcast."""

    quantize_in_weight_transfer: bool = False
    """Use kernel-format FP8 quantized NCCL transfer for weight updates."""

    inference_world_size: int = Field(1, ge=1)
    """Total inference GPUs across all servers. Used by ``init_nccl_broadcast`` to compute per-server rank offsets."""


WeightBroadcastConfig: TypeAlias = Annotated[
    FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig, Field(discriminator="type")
]


class OrchestratorExperimentalConfig(BaseConfig):
    pass


class RolloutModelConfig(BaseConfig):
    model: ModelConfig = ModelConfig()

    client: ClientConfig = ClientConfig()


class OrchestratorConfig(BaseConfig):
    training_mode: Literal["rl", "opd", "sft"] = "rl"
    """Training mode. ``rl``: student generates rollouts, no teacher. ``opd``: student generates rollouts, teacher computes logprobs (teacher_tau > 0). ``sft``: teacher generates rollouts, student inference pool used for evals and weight sync."""

    student: RolloutModelConfig = Field(RolloutModelConfig(), validation_alias=AliasChoices("student", "model"))
    """Student rollout participant (model + client) — the model being trained."""

    teacher: RolloutModelConfig | None = Field(None, validation_alias=AliasChoices("teacher", "teacher_model"))
    """Teacher rollout participant (model + client). Role depends on ``training_mode``: ``opd`` — teacher computes logprobs; ``sft`` — teacher generates rollouts."""

    train: TrainConfig = TrainConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    renderer: RendererConfig = RendererConfig()
    """Client-side renderer configuration. Only consumed when ``use_renderer=true``."""

    optim: OptimizerConfig = OptimizerConfig()
    """Per-run optimizer configuration for multi-run training."""

    eval: EvalConfig | None = None
    """Evaluation configuration."""

    buffer: BufferConfig = BufferConfig()

    advantage: AdvantageConfig | None = DefaultAdvantageConfig()

    filters: list[FilterConfig] = [GibberishFilterConfig(), RepetitionFilterConfig(), ZeroAdvantageFilterConfig()]
    """Rollout filters. Each filter can ``monitor`` (default) or ``enforce`` (skip rollouts)."""

    log: LogConfig = LogConfig()

    wandb: WandbWithExtrasConfig | None = None

    prime_monitor: PrimeMonitorConfig | None = None

    collect_inference_metrics: bool = True
    """Collect inference-server metrics (requires wandb)."""

    ckpt: CheckpointConfig | None = None
    """Checkpoint configuration."""

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()
    """Transport used to receive updated weights from the trainer."""

    rollout_transport: TransportConfig = FileSystemTransportConfig()
    """Transport used to ship rollouts from orchestrator to trainer."""

    output_dir: Path = Path("outputs/run_default")
    """Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node."""

    tasks_per_minute: int | None = Field(None, ge=1)
    """Rate limit per environment worker, in tasks per minute. Recommended for sandbox-backed environments to prevent sandbox-not-ready errors during autoscaling. With multiple workers, the effective total rate is ``workers × this value``. None disables rate limiting."""

    batch_size: int | None = Field(None, ge=1)
    """Samples to train on per step (rollout-based batching). Set this OR ``token_batch_size``."""

    token_batch_size: int | None = Field(None, ge=1)
    """Tokens to train on per step (token-based batching). Set this OR ``batch_size``."""

    oversampling_factor: float | None = Field(None, gt=0)
    """Rollout-mode batching only. Multiplier used to derive ``max_inflight_rollouts`` from ``batch_size`` when ``max_inflight_rollouts`` is unset. Values below 1.0 intentionally cap in-flight rollout capacity below ``batch_size``."""

    max_inflight_rollouts: int | None = Field(None, ge=1)
    """Maximum number of rollouts kept in-flight. Required for token-based batching. With ``batch_size`` set, defaults to ``batch_size * oversampling_factor`` (or ``batch_size`` when ``oversampling_factor`` is unset)."""

    rollouts_per_example: int = Field(1, ge=1)
    """Output sequences returned per example during training."""

    seq_len: int = 2048
    """Training sequence length. Shorter samples are padded; longer samples are truncated."""

    # TODO(Mika): This should be automatic from the number of ZMQ connections
    num_train_workers: int = Field(1, ge=1)
    """Training workers to use."""

    max_steps: int | None = None
    """Maximum training steps. If None, runs indefinitely."""

    max_off_policy_steps: int = Field(8, ge=0)
    """Maximum policies allowed to generate a single rollout. Rollouts generated more than ``max_off_policy_steps`` ahead of training are discarded. Higher values yield better throughput at the cost of off-policy noise."""

    max_async_level: int = Field(1, ge=0)
    """Maximum steps inference can be ahead of training. ``0`` degenerates to synchronous on-policy RL; ``≥1`` overlaps training and inference."""

    strict_async_level: bool = False
    """Strictly enforce ``max_async_level``. When True, the rollout policy is always exactly ``max_async_level`` steps ahead of training. When False, any policy within ``max_async_level`` steps is allowed (always uses the latest available policy)."""

    bench: bool = False
    """Benchmark mode. Sets ``max_steps`` to 5, ``max_async_level`` to ~∞, and disables W&B."""

    seed: int | None = 42
    """Random seed for the orchestrator."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration for monitoring training progress."""

    use_renderer: bool = True
    """Use the renderer-backed TITO client (client-side tokenization via the ``renderers`` package, served by ``/v1/generate``). When True, the ``[orchestrator.renderer]`` block (name / tool_parser / reasoning_parser / pool_size) applies. Default for both text-only and VLM rollouts; VLMs require it. False falls back to MITO (``openai_chat_completions``)."""

    env_install_prerelease: bool = False
    """Allow pre-release versions when installing environments (e.g. ``verifiers>=0.1.12.dev5``). Passes ``--prerelease`` to ``prime env install``."""

    experimental: OrchestratorExperimentalConfig = OrchestratorExperimentalConfig()

    @model_validator(mode="before")
    @classmethod
    def fold_student_shortcuts(cls, data: Any) -> Any:
        """Accept top-level ``[orchestrator.model]`` / ``[orchestrator.client]``
        as shorthand for the student sub-config. Useful for ergonomic rl configs
        where ``[orchestrator.student.*]`` is overkill, and required for
        pre-refactor configs that used the flat layout to keep parsing:

        - [orchestrator.client.*]     -> [orchestrator.student.client.*]
        - [orchestrator.model.<k>]    -> [orchestrator.student.model.<k>]
          (where <k> is any ModelConfig field)

        Teacher must always be configured under [orchestrator.teacher.*]
        (no equivalent shortcut), because rl mode forbids a teacher and we
        don't want the same shortcut to silently route to two different roles.
        """
        if not isinstance(data, dict):
            return data

        def deep_merge(dst: dict, src: dict) -> None:
            """In-place recursive merge of ``src`` into ``dst``. ``src`` wins at the leaf."""
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v

        # 1. Re-nest top-level [orchestrator.client] under student.client.
        legacy_client = data.pop("client", None)
        if isinstance(legacy_client, dict):
            student = data.setdefault("student", {})
            if isinstance(student, dict):
                deep_merge(student.setdefault("client", {}), legacy_client)
            else:
                # Mismatched types - put it back and let pydantic surface the error.
                data["client"] = legacy_client

        # 2. Consolidate the legacy `model` alias into `student` so the
        # flat-layout fix-up below sees a single target. Deep-merge with the
        # legacy keys winning so a CLI `--model.<k>` overrides TOML `student.model.<k>`.
        legacy_model = data.pop("model", None)
        if legacy_model is not None:
            existing = data.get("student")
            if existing is None:
                data["student"] = legacy_model
            elif isinstance(existing, dict) and isinstance(legacy_model, dict):
                deep_merge(existing, legacy_model)
            else:
                # Mismatched types - put it back and let pydantic surface the error.
                data["model"] = legacy_model

        # 3. Re-nest flat ModelConfig keys under student.model.
        model_only_keys = set(ModelConfig.model_fields)
        student = data.get("student")
        if isinstance(student, dict):
            flat = {k: student.pop(k) for k in list(student) if k in model_only_keys}
            if flat:
                student.setdefault("model", {}).update(flat)

        return data

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
                    warnings.warn(
                        "'[[orchestrator.env]]' is deprecated, use '[[orchestrator.train.env]]' instead. "
                        "Auto-translating for now, but this will be removed in a future release.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    train.setdefault("env", data.pop("env"))
                if "sampling" in data:
                    warnings.warn(
                        "'[orchestrator.sampling]' is deprecated, use '[orchestrator.train.sampling]' instead. "
                        "Auto-translating for now, but this will be removed in a future release.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    train.setdefault("sampling", data.pop("sampling"))
        return data

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.student.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.student.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_session_headers(self):
        """Ensure X-Session-ID header is always set for sticky DP-aware routing at the inference router."""
        self.student.client.extra_headers_from_state.setdefault("X-Session-ID", "example_id")
        return self

    @model_validator(mode="after")
    def auto_setup_prime_monitor_run_name(self):
        """Default ``prime_monitor.run_name`` to the W&B run name when monitoring
        is enabled and the user hasn't named the prime-monitor run explicitly."""
        if self.prime_monitor is None or self.prime_monitor.run_name is not None:
            return self
        if self.wandb is not None and self.wandb.name:
            self.prime_monitor.run_name = self.wandb.name
        return self

    @model_validator(mode="after")
    def validate_unique_filter_types(self):
        types = [f.type for f in self.filters]
        if len(types) != len(set(types)):
            raise ValueError(f"Duplicate filter types: {types}. Each filter type may only appear once.")
        return self

    @model_validator(mode="after")
    def _force_no_renderer_for_sft(self):
        """SFT rolls out via the teacher's plain chat-completions endpoint; the
        renderer client doesn't apply. Force use_renderer=False so the user
        doesn't have to remember to set it. Declared before the renderer
        validators below so they see the corrected value."""
        if self.training_mode == "sft":
            self.use_renderer = False
        return self

    @model_validator(mode="after")
    def validate_training_mode(self):
        """Enforce training mode invariants that involve only orchestrator fields."""
        has_teacher = self.teacher is not None
        if self.training_mode == "rl" and has_teacher:
            raise ValueError("orchestrator.teacher must not be set when training_mode = 'rl'.")
        if self.training_mode in ("opd", "sft") and not has_teacher:
            raise ValueError(f"orchestrator.teacher must be configured when training_mode = '{self.training_mode}'.")
        return self

    @model_validator(mode="after")
    def validate_renderer_args(self):
        """``[orchestrator.renderer]`` knobs are only meaningful when
        ``use_renderer=True``. Reject otherwise so callers don't silently
        pass them and wonder why they're ignored."""
        if self.use_renderer:
            return self

        renderer_args_set = []
        if self.renderer.name != "auto":
            renderer_args_set.append(f"renderer.name={self.renderer.name!r}")
        if self.renderer.tool_parser is not None:
            renderer_args_set.append(f"renderer.tool_parser={self.renderer.tool_parser!r}")
        if self.renderer.reasoning_parser is not None:
            renderer_args_set.append(f"renderer.reasoning_parser={self.renderer.reasoning_parser!r}")
        if self.renderer.pool_size is not None:
            renderer_args_set.append(f"renderer.pool_size={self.renderer.pool_size!r}")
        if self.renderer.preserve_all_thinking:
            renderer_args_set.append(f"renderer.preserve_all_thinking={self.renderer.preserve_all_thinking!r}")
        if self.renderer.preserve_thinking_between_tool_calls:
            renderer_args_set.append(
                f"renderer.preserve_thinking_between_tool_calls={self.renderer.preserve_thinking_between_tool_calls!r}"
            )

        if renderer_args_set:
            raise ValueError(
                "Renderer-specific args set without orchestrator.use_renderer=True: "
                f"{', '.join(renderer_args_set)}. Either enable the renderer client or remove these knobs."
            )
        return self

    @model_validator(mode="after")
    def vlm_requires_renderer(self):
        """VLMs (``[model.vlm]`` block set) must go through the renderer.

        The renderer owns the processor per-slot, produces byte-identical
        tokens, and ships generic ``mm_kwargs`` keyed by whatever the
        model's forward signature expects.
        """
        if self.student.model.vlm is not None and not self.use_renderer:
            raise ValueError(
                "orchestrator.use_renderer must be true when model.vlm is set. "
                "VLMs must go through a renderer (e.g. Qwen3VLRenderer) that owns the processor."
            )
        return self

    @model_validator(mode="after")
    def validate_renderer_auto_resolves(self):
        """Reject the silent DefaultRenderer fallback at config time.

        When ``use_renderer=True`` with ``renderer.name='auto'`` and the
        model isn't in ``MODEL_RENDERER_MAP``, ``create_renderer`` would
        fall back to ``DefaultRenderer``. That fallback doesn't fix the
        position-dependent chat-template bug the renderer client exists to
        solve, and rejects envs that pass tools (the rollout dies with
        "RendererPool does not support tools") unless
        ``renderer.tool_parser`` is configured. Surface at config time so
        ``--dry-run`` reports the error.
        """
        if not self.use_renderer or self.renderer.name != "auto":
            return self
        from renderers.base import MODEL_RENDERER_MAP

        model_id = self.tokenizer.name or self.student.model.name
        if model_id in MODEL_RENDERER_MAP:
            return self
        raise ValueError(
            f"orchestrator.use_renderer=True with renderer.name='auto' but "
            f"{model_id!r} is not in renderers.base.MODEL_RENDERER_MAP, so it "
            f"would silently fall back to DefaultRenderer. Pick one: "
            f"(a) [orchestrator.renderer] name='default' — for fine-tunes / "
            f"vendored mirrors with custom chat templates (DefaultRenderer "
            f"calls apply_chat_template); pair with tool_parser=<name> if "
            f"the env uses tools. "
            f"(b) [orchestrator.renderer] name=<model-specific renderer> — "
            f"if {model_id!r} is template-identical to a mapped family "
            f"(and ideally also add it upstream to "
            f"renderers.base.MODEL_RENDERER_MAP). "
            f"(c) orchestrator.use_renderer=false — opt out of the renderer "
            f"client entirely."
        )

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
            oversampling_factor = self.oversampling_factor if self.oversampling_factor is not None else 1.0
            resolved_max_inflight_rollouts = max(
                self.rollouts_per_example,
                int(self.batch_size * oversampling_factor),
            )
            if self.max_inflight_rollouts is not None and self.oversampling_factor is not None:
                expected_max_inflight_rollouts = resolved_max_inflight_rollouts
                if self.max_inflight_rollouts != expected_max_inflight_rollouts:
                    raise ValueError("max_inflight_rollouts conflicts with oversampling_factor * batch_size")
            if self.max_inflight_rollouts is None:
                self.max_inflight_rollouts = resolved_max_inflight_rollouts

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
        is_vllm = self.training_mode != "sft"
        for env in self.train.env:
            env.extra_env_kwargs.update(max_seq_len=self.seq_len)
            if is_vllm:
                env.sampling.extra_body.setdefault("top_k", -1)
                env.sampling.extra_body.setdefault("min_p", 0.0)
                env.sampling.extra_body.setdefault("return_token_ids", True)
        return self
