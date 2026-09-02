import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import verifiers.v1 as vf
from pydantic import Field, SerializeAsAny, model_validator
from renderers import AutoRendererConfig, RendererConfig

from prime_rl.configs.algorithm import (
    AlgoConfig,
    GRPOAlgoConfig,
)
from prime_rl.configs.monitors import OrchestratorMonitorsConfig
from prime_rl.configs.shared import (
    BaseModelConfig,
    BaseWeightBroadcastConfig,
    ClientConfig,
    EnvVars,
    HeartbeatConfig,
    LogConfig,
    ResumeConfig,
    TransportConfig,
    ZMQTransportConfig,
)
from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.utils.config import BaseConfig, default_output_dir


class LoRAConfig(BaseConfig):
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
    temperature: float = Field(1.0, ge=0, le=2.0)
    """Sampling temperature."""

    top_p: float = Field(1.0, gt=0, le=1.0)
    """Nucleus (top-p) sampling for train rollouts. Values below 1.0 truncate the sampling
    distribution; the ``rl`` entrypoint auto-enables sampling replay so trainer and
    rollout distributions stay consistent — see docs/inference.md (Sampling Replay)."""

    top_k: int | None = Field(None, ge=1)
    """Top-k sampling for train rollouts. Truncation triggers sampling replay, and
    a default top-k is injected when only top-p truncates so sampling masks stay
    bounded — see docs/inference.md (Sampling Replay)."""

    max_completion_tokens: int | None = None
    """Maximum output tokens per turn. If None, generates until max context length or EOS."""

    # Strictly speaking, extra_body is not a sampling parameter, but it is the
    # easiest way to pass arbitrary extra parameters to the server via verifiers
    extra_body: dict[str, Any] = {}
    """Extra body forwarded with each request to the inference server."""

    def truncates_distribution(self) -> bool:
        return self.top_p < 1.0 or self.top_k is not None

    @model_validator(mode="after")
    def validate_no_extra_body_truncation(self):
        """Truncating values must come from the typed fields — the replay policy reads
        them. Disabled values pass so resolved configs (where ``resolve_env_config``
        stamped the ``top_k = -1`` / ``min_p = 0.0`` sentinels) re-validate cleanly."""
        smuggled = [
            key
            for key, truncates in (
                ("top_p", self.extra_body.get("top_p", 1.0) < 1.0),
                ("top_k", self.extra_body.get("top_k") not in (None, -1, 0)),
                ("min_p", self.extra_body.get("min_p", 0.0) > 0.0),
            )
            if truncates
        ]
        if smuggled:
            raise ValueError(
                f"extra_body carries truncating {smuggled}; set them as fields on the train "
                "sampling config instead (they drive sampling replay)."
            )
        return self

    def to_sampling_args(self) -> dict[str, Any]:
        """Convert to OAI-compatible sampling args dict, omitting None values."""
        args: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": True,
        }
        if self.max_completion_tokens is not None:
            args["max_completion_tokens"] = self.max_completion_tokens

        # top_k rides extra_body (like EvalSamplingConfig), overriding the sentinel.
        extra_body = dict(self.extra_body)
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if extra_body:
            args["extra_body"] = extra_body

        return args


class EvalSamplingConfig(BaseConfig):
    temperature: float | None = Field(None, ge=0, le=2.0)
    """Sampling temperature. None defers to the inference server default."""

    top_p: float | None = None
    """Nucleus sampling threshold. None defers to the inference server default."""

    top_k: int | None = None
    """Top-k sampling. None defers to the inference server default."""

    min_p: float | None = Field(None, ge=0)
    """Min-p sampling threshold. None defers to the inference server default."""

    max_completion_tokens: int | None = None
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


class EnvConfig(BaseConfig):
    """One environment a run pulls from: the verifiers blocks it composes (``env`` — what
    runs, ``serve`` — how it's hosted) plus this orchestrator's own per-env knobs."""

    env: SerializeAsAny[vf.EnvConfig] = vf.SingleAgentEnvConfig()
    """The verifiers environment — which env, its seed taskset, each agent, its knobs. Narrowed to the selected env's config class by the env id, else the taskset id."""

    serve: vf.ServeConfig = vf.ServeConfig()
    """How this source's env server is hosted. The sizing knobs are consumed by the launcher, which writes each source's env-server config with an unset ``address`` filled in as the derived ``tcp://127.0.0.1:<env_server_base_port + index>``. Setting ``address`` marks the server externally managed: the launchers neither write its env-server TOML nor spawn a server for it, and the orchestrator connects to the given address — e.g. a k8s deployment running env servers in their own pods."""

    name: str | None = None
    """Display name for this environment in logs, metrics, and buffer keys. Defaults to the taskset id. Must be unique across all envs in the same group."""

    ratio: float = Field(1.0, gt=0)
    """Sampling weight for this environment in the buffer. Relative weights are normalized to probabilities across envs (e.g. [1, 1] and [0.5, 0.5] are equivalent). Defaults to 1, i.e. equal weight per env."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        """Narrow ``env`` to the selected env's config class."""
        return vf.resolve_env_field(data, vf.narrowed_env_annotation(cls))

    @property
    def env_id(self) -> str:
        return self.env.env_id or ""

    @property
    def resolved_name(self) -> str:
        return self.name or self.env_id

    @model_validator(mode="after")
    def validate_env(self):
        if not self.env_id:
            raise ValueError('no env configured — set env = { taskset = { id = "<id>" } }')
        if self.resolved_name == "agg":
            raise ValueError(
                'Environment name "agg" is reserved for cross-env metric aggregation. Use a different name or id.'
            )
        return self


class StandardSamplerConfig(BaseConfig):
    type: Literal["standard"] = "standard"


class DifficultyPoolConfig(BaseConfig):
    threshold: float
    """Inclusive maximum reward assigned to this pool."""

    weight: float = Field(ge=0)
    """Relative per-task sampling weight."""


def default_difficulty_pools() -> dict[str, DifficultyPoolConfig]:
    return {
        "hard": DifficultyPoolConfig(threshold=0.25, weight=0.2),
        "normal": DifficultyPoolConfig(threshold=0.75, weight=1.0),
        "easy": DifficultyPoolConfig(threshold=1.0, weight=0.2),
    }


class DifficultyPoolSamplerConfig(BaseConfig):
    type: Literal["difficulty_pool"] = "difficulty_pool"

    pools: dict[str, DifficultyPoolConfig] = Field(default_factory=default_difficulty_pools)
    """Named pools ordered by their reward thresholds."""

    seed: int = 42

    @model_validator(mode="after")
    def validate_pools(self):
        if not self.pools:
            raise ValueError("DifficultyPoolSampler requires at least one pool")
        thresholds = [pool.threshold for pool in self.pools.values()]
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("Difficulty pool thresholds must be unique")
        if not any(pool.weight > 0 for pool in self.pools.values()):
            raise ValueError("At least one difficulty pool must have a positive weight")
        return self


TaskSamplerConfig: TypeAlias = Annotated[
    StandardSamplerConfig | DifficultyPoolSamplerConfig,
    Field(discriminator="type"),
]


class AdvRangeGateConfig(BaseConfig):
    type: Literal["advantage_range"] = "advantage_range"

    reject_min: float = 0.0
    reject_max: float = 0.0

    @model_validator(mode="after")
    def validate_range(self):
        if self.reject_min > self.reject_max:
            raise ValueError("reject_min must be less than or equal to reject_max")
        return self


AdmissionGateConfig: TypeAlias = AdvRangeGateConfig


class CurriculumConfig(BaseConfig):
    sampler: TaskSamplerConfig = Field(default_factory=StandardSamplerConfig)
    """Task selection policy. The default cycles through the task iterator in source order."""

    gates: dict[str, AdmissionGateConfig] = Field(default_factory=dict)
    """Named admission policies. Every gate observes every finalized group,
    and a group trains only when every gate admits it."""


class TrainSourceConfig(EnvConfig):
    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Per-env sampling overrides. Unset fields inherit from the group-level train sampling config."""

    group_size: int = Field(1, ge=1)
    """Rollouts generated per example for GRPO group-relative advantages.
    Inherits from ``orchestrator.group_size`` when unset."""

    algo: AlgoConfig | None = None
    """Training algorithm for this env. Inherits from the top-level
    ``orchestrator.algo`` when unset; set ``type`` (and its params) to give
    this env its own algorithm."""

    curriculum: CurriculumConfig | None = None
    """User-authored task sampler and admission gates. The default cycles
    through the taskset and admits every finalized group."""


class EvalSourceConfig(EnvConfig):
    sampling: EvalSamplingConfig = EvalSamplingConfig()
    """Per-env sampling overrides. Unset fields inherit from the group-level eval sampling config."""

    num_examples: int = -1
    """Eval examples to sample from the dataset. ``-1`` uses all available examples."""

    group_size: int = Field(1, ge=1)
    """Rollouts generated per example. Used for pass@k estimation (e.g. ``group_size=8`` enables pass@1 through pass@8)."""

    interval: int = Field(100, ge=1)
    """Per-env eval interval. If unset, inherits from the group-level eval interval."""


class TrainConfig(BaseConfig):
    source: list[TrainSourceConfig] = Field(default_factory=list)
    """Training sources."""

    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Shared training sampling configuration."""

    @model_validator(mode="after")
    def resolve_env_defaults(self):
        """Resolve per-env overrides: inherit group-level sampling (the worker ``pool``
        is configured per env, defaulting to elastic)."""
        group_sampling = self.sampling.model_dump()
        for env in self.source:
            if "sampling" not in env.model_fields_set:
                env.sampling = TrainSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | env.sampling.model_dump(exclude_unset=True)
                env.sampling = TrainSamplingConfig(**merged)
        return self

    @model_validator(mode="after")
    def validate_unique_env_names(self):
        env_names = [env.resolved_name for env in self.source]
        duplicates = [n for n in env_names if env_names.count(n) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate training environment names: {set(duplicates)}. Each env must have a unique name."
            )
        return self


class EvalConfig(BaseConfig):
    source: list[EvalSourceConfig] = Field(default_factory=list)
    """Evaluation sources."""

    sampling: EvalSamplingConfig = Field(default_factory=EvalSamplingConfig)
    """Shared eval sampling configuration; can differ from training sampling."""

    num_examples: int = -1
    """Default eval examples per environment. ``-1`` uses all. Can be overridden per env."""

    group_size: int = Field(1, ge=1)
    """Default rollouts per example. Can be overridden per env."""

    interval: int = Field(100, ge=1)
    """Step interval at which to evaluate the model."""

    skip_first_step: bool = False
    """If True, skip the startup eval that otherwise runs before any
    train rollouts."""

    retrigger_on_resume: bool = False
    """If True, re-trigger evals at the checkpoint step on resume (e.g. after a
    crash that left in-flight evals unfinished). By default, assumes a clean
    exit where all evals already completed."""

    @model_validator(mode="after")
    def resolve_env_defaults(self):
        """Resolve per-env overrides: inherit group-level sampling, num_examples,
        group_size, and interval (the worker ``pool`` is configured per env, default elastic)."""
        group_sampling = self.sampling.model_dump()
        for source in self.source:
            if "sampling" not in source.model_fields_set:
                source.sampling = EvalSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | source.sampling.model_dump(exclude_unset=True)
                source.sampling = EvalSamplingConfig(**merged)
            if "num_examples" not in source.model_fields_set:
                source.num_examples = self.num_examples
            if "group_size" not in source.model_fields_set:
                source.group_size = self.group_size
            if "interval" not in source.model_fields_set:
                source.interval = self.interval
        return self

    @model_validator(mode="after")
    def validate_non_empty_sources(self):
        if not self.source:
            raise ValueError(
                "EvalConfig must define at least one source. Either drop the "
                "[orchestrator.eval] block entirely (to disable eval) or "
                "add a [[orchestrator.eval.source]] block."
            )
        return self

    @model_validator(mode="after")
    def validate_unique_env_names(self):
        env_names = [source.resolved_name for source in self.source]
        duplicates = [n for n in env_names if env_names.count(n) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate evaluation environment names: {set(duplicates)}. Each env must have a unique name."
            )
        return self


class CheckpointConfig(BaseConfig):
    interval: int | None = Field(None, ge=1)
    """Step interval at which to save the orchestrator checkpoint."""

    wait_for_weights_timeout: int | None = Field(None, ge=1)
    """Wait up to this many seconds for the startup weight directory to appear (the trainer broadcasts the incoming policy — v0 from scratch, the resumed step's version on resume — before the first step). If None, fall back to a default timeout. Raise this for large models on slow shared filesystems."""

    keep_last: int | None = Field(None, ge=1)
    """Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency."""

    keep_interval: int | None = Field(None, ge=1)
    """Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping."""

    skip_progress: bool = False
    """Skip loading the progress from checkpoint."""


class FileSystemWeightBroadcastConfig(BaseWeightBroadcastConfig):
    type: Literal["filesystem"] = "filesystem"


class SparseFileSystemWeightBroadcastConfig(BaseWeightBroadcastConfig):
    type: Literal["sparse_filesystem"] = "sparse_filesystem"


class InMemoryWeightBroadcastConfig(BaseWeightBroadcastConfig):
    host: str = "localhost"
    """Weight transfer host."""

    port: int
    """Weight transfer port."""

    inference_world_size: int = Field(1, ge=1)
    """Total inference workers across all servers."""


class NCCLWeightBroadcastConfig(InMemoryWeightBroadcastConfig):
    type: Literal["nccl"] = "nccl"

    port: int = 29501
    """Port for the NCCL broadcast rendezvous."""

    quantize_in_weight_transfer: bool = False
    """Use kernel-format FP8 quantized NCCL transfer for weight updates."""


class NIXLWeightBroadcastConfig(InMemoryWeightBroadcastConfig):
    type: Literal["nixl"] = "nixl"

    port: int = 8001
    """ModelExpress gRPC port."""

    session_id: str = "default"
    """ModelExpress session ID."""

    overlap_transfer_and_replay: bool = False
    """Allocate two transfer arenas so inference can replay one weight group while receiving the next."""


WeightBroadcastConfig: TypeAlias = Annotated[
    FileSystemWeightBroadcastConfig
    | SparseFileSystemWeightBroadcastConfig
    | NCCLWeightBroadcastConfig
    | NIXLWeightBroadcastConfig,
    Field(discriminator="type"),
]


class ConcurrencyConfig(BaseConfig):
    """Adaptive in-flight concurrency control. The orchestrator sizes the
    in-flight episode cap from engine KV capacity and learned per-env episode
    costs; these fields only bound and seed it."""

    initial_inflight: int | None = Field(None, ge=1)
    """Optional initial in-flight episodes to start from. Set it when a good value is known to skip the initial ramp; otherwise auto-derive a pessimistic bound at runtime."""

    min_inflight: int = Field(1, ge=1)
    """Minimum number of in-flight episodes. Set ``min_inflight = max_inflight`` to recover fixed concurrency."""

    max_inflight: int | None = Field(1024, ge=1)
    """Maximum number of in-flight episodes. Set it to avoid runaway concurrency, especially to limit other external resources (e.g. sandboxes). None removes the ceiling."""

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.max_inflight is not None:
            if self.initial_inflight is not None and self.initial_inflight > self.max_inflight:
                raise ValueError("concurrency.initial_inflight must not exceed concurrency.max_inflight")
            if self.min_inflight > self.max_inflight:
                raise ValueError("concurrency.min_inflight must not exceed concurrency.max_inflight")
        return self


# Top-k injected on truncated policy sampling that has none, and the hard upper
# bound for explicit top-k. vLLM's native sampling-mask capture requires a
# per-request top_k > 0 to bound mask sizes, and the trainer pads each micro
# batch's masks to the largest sampling mask, so the bound also caps trainer
# mask tensors. Large enough that a 0.95-0.99 nucleus rarely reaches it (the
# sampling policy is essentially unchanged).
TRAIN_TOP_K_BOUND = 512


class OrchestratorConfig(BaseConfig):
    algo: AlgoConfig = GRPOAlgoConfig()
    """Training algorithm: sampling plus the per-token training signal (credit
    assignment and loss routing, fused — its ``type`` names the algorithm).
    Defaults to ``grpo``. Override per source via ``[[orchestrator.train.source]]``'s
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

    eval: EvalConfig | None = None
    """Evaluation configuration."""

    log: LogConfig = LogConfig()

    env_vars: EnvVars = {}
    """Extra environment variables for the orchestrator process(es). Merged on top of the launcher defaults."""

    monitors: OrchestratorMonitorsConfig = OrchestratorMonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``, ``monitors.prime``)."""

    collect_inference_metrics: bool = True
    """Mirror inference-server metrics to W&B (requires wandb). The ``/metrics`` poll itself always runs — it feeds the concurrency controller."""

    inference_metrics_roles: list[Literal["prefill", "decode"]] | None = None
    """Role for each policy admin client when collecting P/D inference metrics."""

    ckpt: CheckpointConfig | None = None

    resume: ResumeConfig | None = None
    """Resume the orchestrator from a checkpoint. None starts from scratch; an empty block resumes from the latest checkpoint, ``resume.step`` from that step, ``resume.dir`` from an external checkpoint step directory. Without ``ckpt`` the run loads but saves no new checkpoints."""
    """Checkpoint configuration."""

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()
    """Transport used to receive updated weights from the trainer."""

    rollout_transport: TransportConfig = ZMQTransportConfig()
    """Transport used to ship rollouts from orchestrator to trainer."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Shared with the trainer; should be a persistent directory with enough disk space and unique per experiment running on a single node. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    tasks_per_minute: int | None = Field(None, ge=1)
    """Global rate limit on task dispatch, in tasks per minute. Recommended for sandbox-backed environments to prevent sandbox-not-ready errors during autoscaling. None disables rate limiting."""

    env_server_base_port: int = Field(5000, ge=1, le=65535)
    """First port of the env-server port range: the source at position ``i`` (train, then eval) is served at ``tcp://127.0.0.1:<base + i>``. Sources with an explicit ``serve.address`` keep it instead, without shifting the other sources' ports (indices stay positional). Give concurrent runs on one host distinct bases (e.g. one per multi-run orchestrator)."""

    batch_size: int | None = Field(None, ge=1)
    """Samples to train on per step (rollout-based batching). Set this OR ``token_batch_size``."""

    constant_trainer_batch_size: bool = True
    """Require each batch to reach its effective sample target."""

    token_batch_size: int | None = Field(None, ge=1)
    """Tokens to train on per step (token-based batching). Set this OR ``batch_size``."""

    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    """Adaptive in-flight concurrency control (``[orchestrator.concurrency]``)."""

    group_size: int = Field(1, ge=1)
    """Output sequences returned per example during training."""

    seq_len: int = 2048
    """Training sequence length. Shorter samples are padded; longer samples are truncated."""

    num_train_workers: int = Field(1, ge=1)
    """Trainer data-parallel world size (trainer world size // cp). The orchestrator packs one micro-batch list per DP rank, so this must match the trainer topology. Auto-filled by the ``rl`` entrypoint; set explicitly for standalone orchestrator runs."""

    pad_to_multiple_of: int = Field(1, ge=1)
    """Pad each packed micro batch to a multiple of this value (the trainer's cp degree). Auto-filled by the ``rl`` entrypoint; set explicitly for standalone orchestrator runs with cp > 1."""

    max_steps: int | None = None
    """Maximum training steps. If None, runs indefinitely."""

    max_off_policy_steps: int = Field(8, ge=0)
    """Maximum staleness of a trained rollout: the version a batch trains on (v{step-1}) minus the oldest version that generated the rollout (a rollout can span several weight updates), queue time included. Episodes past the bound are dropped, in-flight and queued; a group shares one dispatch version, so its episodes age out together. Higher values yield better throughput at the cost of off-policy noise."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration for monitoring training progress."""

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_prime_monitor_name(self):
        """Default ``monitors.prime.name`` to the W&B run name when monitoring
        is enabled and the user hasn't named the platform run explicitly."""
        if self.monitors.prime is None or self.monitors.prime.name is not None:
            return self
        if self.monitors.wandb is not None and self.monitors.wandb.name:
            self.monitors.prime.name = self.monitors.wandb.name
        return self

    @model_validator(mode="after")
    def inherit_env_algorithms(self):
        """Envs without their own algorithm inherit the top-level one.
        Declared before any validator that reads ``algo``."""
        for env_cfg in self.train.source:
            if env_cfg.algo is None:
                env_cfg.algo = self.algo.model_copy(deep=True)
        return self

    @model_validator(mode="after")
    def validate_env_algorithms(self):
        """Let each algorithm reject environments it cannot score correctly."""
        for env_cfg in self.train.source:
            assert env_cfg.algo is not None  # resolved by inherit_env_algorithms
            env_cfg.algo.validate_env(env_cfg.env)
        return self

    @model_validator(mode="after")
    def setup_truncated_sampling(self):
        """Truncated policy sampling trains with sampling replay (rollout
        logprobs are renormalized — see docs/inference.md, Sampling Replay).
        Owned here: every truncating config gets a top-k bound (bounds the sampling
        masks); opd/opsd is rejected (full-vocab prefill refs would mix
        normalizations). Frozen-source envs sample externally and are exempt."""
        policy_samplings = [
            env.sampling for env in self.train.source if env.algo is not None and env.algo.sampling.source == "policy"
        ] or ([self.train.sampling] if not self.train.source else [])
        truncating = [sampling for sampling in policy_samplings if sampling.truncates_distribution()]
        if not truncating:
            return self

        if any(sampling.temperature == 0 for sampling in truncating):
            raise ValueError(
                "Truncated train sampling (top_p/top_k) requires temperature > 0: greedy sampling has "
                "no truncated distribution to replay, and the inference server rejects such requests "
                "while sampling-mask capture is on."
            )

        oversized = [sampling.top_k for sampling in truncating if (sampling.top_k or 0) > TRAIN_TOP_K_BOUND]
        if oversized:
            raise ValueError(
                f"Truncated train sampling with top_k = {max(oversized)} exceeds the sampling-replay "
                f"bound ({TRAIN_TOP_K_BOUND}): the trainer pads each micro batch's masks to the largest "
                f"sampling mask, so unbounded masks blow up trainer memory. Use top_k <= {TRAIN_TOP_K_BOUND}."
            )

        unbounded = [sampling for sampling in truncating if sampling.top_k is None]
        if unbounded:
            warnings.warn(
                f"Truncated train sampling: defaulting top_k = {TRAIN_TOP_K_BOUND} so every sampling mask is "
                "bounded and sampling replay stays exact. Set top_k explicitly to override.",
                stacklevel=2,
            )
            for sampling in unbounded:
                sampling.top_k = TRAIN_TOP_K_BOUND

        algos = [env.algo for env in self.train.source if env.algo is not None] or [self.algo]
        if any(algo.type in ("opd", "opsd") for algo in algos):
            raise ValueError(
                "opd/opsd is not supported with truncated train sampling: reference logprobs are full-vocab "
                "prefill scores while trainer logprobs are renormalized over the sampling mask, biasing the "
                "ref_kl term. Remove the truncation (top_p/top_k) or the opd/opsd algo."
            )

        return self

    @property
    def any_policy_sourced(self) -> bool:
        """True when at least one train env samples rollouts from the live policy."""
        return any(env.algo is not None and env.algo.sampling.source == "policy" for env in self.train.source)

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

        if self.batch_size is not None and self.batch_size % self.group_size != 0:
            raise ValueError("Batch size must be divisible by the number of samples per problem")

        for field in ("max_inflight", "initial_inflight"):
            value = getattr(self.concurrency, field)
            if value is not None and value < self.group_size:
                raise ValueError(f"concurrency.{field} must be at least the number of rollouts per example")

        # Propagate the top-level ``group_size`` into each train env that didn't set its own.
        for env_cfg in self.train.source:
            if "group_size" not in env_cfg.model_fields_set:
                env_cfg.group_size = self.group_size

        return self

    @model_validator(mode="after")
    def resolve_env_config(self):
        """Set vLLM sampling defaults on each train env from top-level fields."""
        for env in self.train.source:
            # Policy-sourced rollouts hit our vLLM server; frozen-sourced
            # rollouts may hit external OAI endpoints that reject these knobs.
            assert env.algo is not None
            if env.algo.sampling.source == "policy":
                env.sampling.extra_body.setdefault("top_k", -1)
                env.sampling.extra_body.setdefault("min_p", 0.0)
                env.sampling.extra_body.setdefault("return_token_ids", True)
        return self

    @model_validator(mode="after")
    def validate_policy_top_k_consistency(self):
        """Require one top-k capture mode across the live policy server."""
        policy_sources = [
            env for env in self.train.source if env.algo is not None and env.algo.sampling.source == "policy"
        ]
        enabled = [env for env in policy_sources if env.sampling.top_k is not None]
        disabled = [env for env in policy_sources if env.sampling.top_k is None]
        if enabled and disabled:
            names = ", ".join(env.resolved_name for env in disabled)
            raise ValueError(
                "Live-policy train sources cannot mix top_k > 0 and top_k = -1 because "
                "sampling-mask capture is engine-wide. Set top_k > 0 for these sources: "
                f"{names}."
            )
        return self

    @property
    def env_sources(self) -> list[tuple[str, EnvConfig]]:
        """Every ``(split, source)`` this run pulls from, train first then eval — the
        order that fixes each source's deterministic env-server port."""
        sources: list[tuple[str, EnvConfig]] = [("train", source) for source in self.train.source]
        if self.eval is not None:
            sources += [("eval", source) for source in self.eval.source]
        return sources

    @property
    def env_addresses(self) -> dict[tuple[str, str], str]:
        """Where each source's env server lives, keyed by ``(split, resolved_name)``:
        the source's own ``serve.address`` when set (an externally managed server), else
        ``tcp://127.0.0.1:<port>`` with ports from ``env_server_base_port`` in
        ``env_sources`` order. The launcher binds env servers at exactly these addresses
        and the orchestrator connects to them, so both sides agree from the config
        alone."""
        return {
            (split, source.resolved_name): source.serve.address
            or f"tcp://127.0.0.1:{self.env_server_base_port + index}"
            for index, (split, source) in enumerate(self.env_sources)
        }
