import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import verifiers.v1 as vf
from pydantic import AliasChoices, Field, model_validator
from renderers import AutoRendererConfig, RendererConfig

from prime_rl.configs.algorithm import (
    AlgoConfig,
    GRPOAlgoConfig,
)
from prime_rl.configs.shared import (
    BaseModelConfig,
    ClientConfig,
    EnvVars,
    FileSystemTransportConfig,
    HeartbeatConfig,
    LogConfig,
    PrimeMonitorConfig,
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

    client: ClientConfig = ClientConfig()
    """Client of the live deployment (``[orchestrator.model.client]``)."""


class TrainSamplingConfig(BaseConfig):
    temperature: float = Field(1.0, ge=0)
    """Sampling temperature."""

    max_completion_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_completion_tokens", "max_tokens")
    )
    """Maximum output tokens per turn. If None, generates until max context length or EOS."""

    # Strictly speaking, extra_body is not a sampling parameter, but it is the
    # easiest way to pass arbitrary extra parameters to the server via verifiers
    extra_body: dict[str, Any] = {}
    """Extra body forwarded with each request to the inference server."""

    def to_sampling_args(self) -> dict[str, Any]:
        """Convert to OAI-compatible sampling args dict, omitting None values."""
        args: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": 1.0,
            "logprobs": True,
        }
        if self.max_completion_tokens is not None:
            args["max_completion_tokens"] = self.max_completion_tokens

        if self.extra_body:
            args["extra_body"] = dict(self.extra_body)

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

    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    """Reasoning effort constraint for reasoning models."""

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

        extra_body = dict(self.extra_body)
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
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


class EnvConfig(vf.EnvServerConfig):
    name: str | None = None
    """Display name for this environment in logs, metrics, and buffer keys. Defaults to the taskset id. Must be unique across all envs in the same group."""

    address: str | None = None
    """ZMQ address of an external env server (e.g. ``tcp://host:5000``). When set, the orchestrator connects to this server instead of spawning one; when None, a subprocess env server is spawned automatically. The ``pool`` sizes the spawned server."""

    ratio: float | None = Field(None, gt=0)
    """Sampling weight for this environment in the buffer. When None for all envs, samples uniformly across all available problems. When set, must be set on all envs — values are relative weights normalized to probabilities (e.g. [1, 1] and [0.5, 0.5] are equivalent)."""

    @model_validator(mode="before")
    @classmethod
    def _migrate_num_workers(cls, data):
        """Back-compat: the removed ``num_workers`` maps onto ``pool`` — an int becomes a
        fixed ``static`` pool, ``"auto"`` falls through to the default ``elastic`` pool. An
        explicit ``pool`` always wins."""
        if isinstance(data, dict) and "num_workers" in data:
            num_workers = data.pop("num_workers")
            if "pool" not in data and num_workers != "auto":
                data["pool"] = {"type": "static", "num_workers": num_workers}
        return data

    @property
    def is_legacy(self) -> bool:
        """A v0/legacy env (run via the bridge): an ``id`` is set and no v1 ``taskset`` is."""
        return not self.taskset.id

    @property
    def env_id(self) -> str:
        """The env identifier — the v1 taskset id (v1) or the legacy env id (v0)."""
        return self.taskset.id or self.id or ""

    @property
    def resolved_name(self) -> str:
        return self.name or self.env_id

    @model_validator(mode="after")
    def validate_env(self):
        if not self.taskset.id and not self.id:
            raise ValueError('no env configured — set taskset = { id = "<id>" } (v1) or id = "<id>" (v0/legacy)')
        if self.resolved_name == "agg":
            raise ValueError(
                'Environment name "agg" is reserved for cross-env metric aggregation. Use a different name or id.'
            )
        return self

    @model_validator(mode="after")
    def resolve_legacy_env_kwargs(self):
        """For a v0/legacy env, surface the v1 knobs the legacy bridge applies via
        ``extra_env_kwargs`` (``env.set_kwargs(...)``): the per-rollout wall-clock timeout and
        the multi-turn completion-token budget. (``max_seq_len`` is added per train run in
        ``OrchestratorConfig.resolve_env_config``, which knows ``seq_len``.)"""
        if self.is_legacy:
            if self.timeout.rollout is not None:
                self.extra_env_kwargs["timeout_seconds"] = self.timeout.rollout
            if self.max_output_tokens is not None:
                self.extra_env_kwargs["max_total_completion_tokens"] = self.max_output_tokens
        return self


class TrainEnvConfig(EnvConfig):
    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Per-env sampling overrides. Unset fields inherit from the group-level train sampling config."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
    """Rollouts generated per example for GRPO group-relative advantages.
    Inherits from ``orchestrator.group_size`` when unset."""

    algo: AlgoConfig | None = None
    """Training algorithm for this env. Inherits from the top-level
    ``orchestrator.algo`` when unset; set ``type`` (and its params) to give
    this env its own algorithm."""


class EvalEnvConfig(EnvConfig):
    sampling: EvalSamplingConfig = EvalSamplingConfig()
    """Per-env sampling overrides. Unset fields inherit from the group-level eval sampling config."""

    num_examples: int = -1
    """Eval examples to sample from the dataset. ``-1`` uses all available examples."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
    """Rollouts generated per example. Used for pass@k estimation (e.g. ``group_size=8`` enables pass@1 through pass@8)."""

    interval: int = Field(100, ge=1)
    """Per-env eval interval. If unset, inherits from the group-level eval interval."""


class TrainConfig(BaseConfig):
    env: list[TrainEnvConfig] = Field(default_factory=list)
    """Training environments."""

    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Shared training sampling configuration."""

    @model_validator(mode="after")
    def resolve_env_defaults(self):
        """Resolve per-env overrides: inherit group-level sampling (the worker ``pool``
        is configured per env, defaulting to elastic)."""
        group_sampling = self.sampling.model_dump()
        for env in self.env:
            if "sampling" not in env.model_fields_set:
                env.sampling = TrainSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | env.sampling.model_dump(exclude_unset=True)
                env.sampling = TrainSamplingConfig(**merged)
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
    env: list[EvalEnvConfig] = Field(default_factory=list)
    """Evaluation environments."""

    sampling: EvalSamplingConfig = Field(default_factory=EvalSamplingConfig)
    """Shared eval sampling configuration; can differ from training sampling."""

    num_examples: int = -1
    """Default eval examples per environment. ``-1`` uses all. Can be overridden per env."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
    """Default rollouts per example. Can be overridden per env."""

    interval: int = Field(100, ge=1)
    """Step interval at which to evaluate the model."""

    skip_first_step: bool = False
    """If True, skip the startup eval that otherwise runs before any
    train rollouts."""

    @model_validator(mode="after")
    def resolve_env_defaults(self):
        """Resolve per-env overrides: inherit group-level sampling, num_examples,
        group_size, and interval (the worker ``pool`` is configured per env, default elastic)."""
        group_sampling = self.sampling.model_dump()
        for env in self.env:
            if "sampling" not in env.model_fields_set:
                env.sampling = EvalSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | env.sampling.model_dump(exclude_unset=True)
                env.sampling = EvalSamplingConfig(**merged)
            if "num_examples" not in env.model_fields_set:
                env.num_examples = self.num_examples
            if "group_size" not in env.model_fields_set:
                env.group_size = self.group_size
            if "interval" not in env.model_fields_set:
                env.interval = self.interval
        return self

    @model_validator(mode="after")
    def validate_non_empty_envs(self):
        if not self.env:
            raise ValueError(
                "EvalConfig must define at least one env. Either drop the "
                "[orchestrator.eval] block entirely (to disable eval) or "
                "add a [[orchestrator.eval.env]] block."
            )
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


class OrchestratorConfig(BaseConfig):
    algo: AlgoConfig = GRPOAlgoConfig()
    """Training algorithm: sampling plus the per-token training signal (credit
    assignment and loss routing, fused — its ``type`` names the algorithm).
    Defaults to ``grpo``. Override per env via ``[[orchestrator.train.env]]``'s
    ``algo``."""

    model: ModelConfig = ModelConfig()
    """The model being trained: its model fields plus the client of the live
    vLLM deployment (``[orchestrator.model] name = ...`` with
    ``[orchestrator.model.client]``). Algorithm components reference it as
    ``"policy"``."""

    train: TrainConfig = TrainConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    renderer: RendererConfig = AutoRendererConfig()
    """Typed renderer config (``renderers.RendererConfig`` discriminated union), required —
    training is renderer-only. Defaults to ``"auto"``, which resolves from
    ``tokenizer.name_or_path`` via ``MODEL_RENDERER_MAP``. RL/OPD roll out through the renderer
    client; SFT uses it to backfill tokens for its chat-completions teacher."""

    pool_size: int | None = Field(None, ge=1)
    """Number of renderer slots shared across concurrent rollouts. Bump
    for long multi-turn prompts where client-side jinja tokenization
    serializes."""

    optim: OptimizerConfig = OptimizerConfig()
    """Per-run optimizer configuration for multi-run training."""

    eval: EvalConfig | None = None
    """Evaluation configuration."""

    pre_batch_filters: list[FilterConfig] = [
        GibberishFilterConfig(enforce=False),
        RepetitionFilterConfig(enforce=False),
        ZeroAdvantageFilterConfig(enforce=False),
    ]
    """Filters applied *before* a rollout enters the training batch buffer.
    All three filter types are registered in monitor mode by default; flip ``enforce=true`` per type
    to drop matching rollouts before they consume a slot in the batch (e.g. a zero-advantage group
    never makes it into a training batch)."""

    post_batch_filters: list[FilterConfig] = [
        GibberishFilterConfig(),
        RepetitionFilterConfig(),
        ZeroAdvantageFilterConfig(),
    ]
    """Filters applied *after* a batch has been assembled. Each filter annotates each rollout;
    rollouts flagged by an enforcing filter are still recorded but not shipped to the trainer."""

    log: LogConfig = LogConfig()

    env_vars: EnvVars = {}
    """Extra environment variables for the orchestrator process(es). Merged on top of the launcher defaults."""

    wandb: WandbWithExtrasConfig | None = None

    prime_monitor: PrimeMonitorConfig | None = None

    collect_inference_metrics: bool = True
    """Collect inference-server metrics (requires wandb)."""

    inference_metrics_roles: list[Literal["prefill", "decode"]] | None = None
    """Role for each policy admin client when collecting P/D inference metrics."""

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

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
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

    bench: bool = False
    """Benchmark mode. Sets ``max_steps`` to 5 and disables W&B."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration for monitoring training progress."""

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
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_session_headers(self):
        """Ensure X-Session-ID header is always set for sticky DP-aware routing at the inference router."""
        self.model.client.extra_headers_from_state.setdefault("X-Session-ID", "trajectory_id")
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
        for slot_name in ("pre_batch_filters", "post_batch_filters"):
            types = [f.type for f in getattr(self, slot_name)]
            if len(types) != len(set(types)):
                raise ValueError(
                    f"Duplicate filter types in {slot_name}: {types}. Each filter type may only appear once per slot."
                )
        return self

    @model_validator(mode="after")
    def inherit_env_algorithms(self):
        """Envs without their own algorithm inherit the top-level one.
        Declared before any validator that reads ``algo``."""
        for env_cfg in self.train.env:
            if env_cfg.algo is None:
                env_cfg.algo = self.algo.model_copy(deep=True)
        return self

    @property
    def any_policy_sourced(self) -> bool:
        """True when at least one train env samples rollouts from the live policy."""
        return any(env.algo is not None and env.algo.sampling.source == "policy" for env in self.train.env)

    @model_validator(mode="after")
    def validate_pool_size(self):
        """``pool_size`` sizes the renderer-client pool for policy-sourced
        sampling. Reject it when that path never runs — no train env samples
        from the policy — so callers don't silently pass it and wonder why
        it's ignored."""
        if self.pool_size is None:
            return self
        if not self.any_policy_sourced:
            raise ValueError(
                f"orchestrator.pool_size={self.pool_size!r} is set but no train env samples "
                "from the policy — the renderer-client sampling pool never runs (the renderer "
                "is still used for client-side tokenization). Remove pool_size."
            )
        return self

    @model_validator(mode="after")
    def validate_renderer_auto_resolves(self):
        """Reject the silent DefaultRenderer fallback at config time.

        When ``renderer.name='auto'`` and the model isn't in
        ``MODEL_RENDERER_MAP``, ``create_renderer`` would fall back to
        ``DefaultRenderer``. That fallback doesn't fix the
        position-dependent chat-template bug the renderer client exists
        to solve, and rejects envs that pass tools (the rollout dies
        with "RendererPool does not support tools") unless
        ``DefaultRendererConfig.tool_parser`` is configured. Surface at
        config time so ``--dry-run`` reports the error.
        """
        if self.renderer.name != "auto":
            return self
        from renderers.base import MODEL_RENDERER_MAP

        model_id = self.tokenizer.name or self.model.name
        if model_id in MODEL_RENDERER_MAP:
            return self
        raise ValueError(
            f"orchestrator.renderer.name='auto' but "
            f"{model_id!r} is not in renderers.base.MODEL_RENDERER_MAP, so it "
            f"would silently fall back to DefaultRenderer. Pick one: "
            f"(a) [orchestrator.renderer] name='default' — for fine-tunes / "
            f"vendored mirrors with custom chat templates (DefaultRenderer "
            f"calls apply_chat_template); set tool_parser=<name> if the env "
            f"uses tools. "
            f"(b) [orchestrator.renderer] name=<model-specific renderer> — "
            f"if {model_id!r} is template-identical to a mapped family "
            f"(and ideally also add it upstream to "
            f"renderers.base.MODEL_RENDERER_MAP)."
        )

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
            if self.batch_size % self.group_size != 0:
                raise ValueError("Batch size must be divisible by the number of samples per problem")
            oversampling_factor = self.oversampling_factor if self.oversampling_factor is not None else 1.0
            resolved_max_inflight_rollouts = max(
                self.group_size,
                int(self.batch_size * oversampling_factor),
            )
            if self.max_inflight_rollouts is not None and self.oversampling_factor is not None:
                expected_max_inflight_rollouts = resolved_max_inflight_rollouts
                if self.max_inflight_rollouts != expected_max_inflight_rollouts:
                    raise ValueError("max_inflight_rollouts conflicts with oversampling_factor * batch_size")
            if self.max_inflight_rollouts is None:
                self.max_inflight_rollouts = resolved_max_inflight_rollouts

        if self.max_inflight_rollouts is not None and self.max_inflight_rollouts < self.group_size:
            raise ValueError("max_inflight_rollouts must be at least the number of rollouts per example")

        # Propagate the top-level ``group_size`` into each train env that didn't set its own.
        for env_cfg in self.train.env:
            if "group_size" not in env_cfg.model_fields_set:
                env_cfg.group_size = self.group_size

        return self

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.max_steps = 4  # Run for 1 warmup step + 3 evaluation steps

            # Disable evaluation
            self.eval = None
            if self.wandb:
                self.wandb.log_extras = None
            if self.prime_monitor:
                self.prime_monitor.log_extras = None

        return self

    @model_validator(mode="after")
    def resolve_env_config(self):
        """Set vLLM sampling defaults + legacy env kwargs on each train env from top-level fields."""
        for env in self.train.env:
            # Policy-sourced rollouts hit our vLLM server; frozen-sourced
            # rollouts may hit external OAI endpoints that reject these knobs.
            assert env.algo is not None
            if env.algo.sampling.source == "policy":
                env.sampling.extra_body.setdefault("top_k", -1)
                env.sampling.extra_body.setdefault("min_p", 0.0)
                env.sampling.extra_body.setdefault("return_token_ids", True)
            if env.is_legacy:
                # v0 env: cap per-turn response tokens to the training budget (the legacy
                # bridge applies extra_env_kwargs via env.set_kwargs).
                env.extra_env_kwargs["max_seq_len"] = self.seq_len
        return self
