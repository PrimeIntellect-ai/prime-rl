import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.monitors import TrainMonitorsConfig
from prime_rl.configs.sft import BaseDataConfig, FakeDataConfig
from prime_rl.configs.shared import EnvVars, HeartbeatConfig, ResumeConfig, RunConfig, TrainerLogConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    CheckpointConfig,
    GCConfig,
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
)
from prime_rl.utils.config import BaseConfig, default_output_dir


class PretrainModelConfig(ModelConfig):
    arch: dict[str, Any] = {}
    """HF config fields that override the config of the baseline model ``name``, e.g. ``hidden_size`` or ``num_hidden_layers``. The baseline provides the architecture class, the tokenizer and the vocabulary. The launcher writes the resulting config and tokenizer to ``<run_dir>/model`` and trains from there."""


class TextDataConfig(BaseDataConfig):
    type: Literal["text"] = "text"

    name: str = "HuggingFaceFW/fineweb-edu"
    """HF dataset name or local path."""

    subsets: list[str] | None = None
    """Subsets (HF configs) to load."""

    splits: list[str] | None = None
    """Splits to load. Defaults to ``train``."""

    revision: str | None = None
    """HF dataset revision."""

    probabilities: list[float] | None = None
    """Sampling probabilities when interleaving several subsets/splits."""

    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted"
    """Stopping strategy when interleaving several subsets/splits."""

    text_column: str = "text"
    """Column holding the raw document text."""

    shuffle: bool = True
    """Shuffle the documents at the start of each epoch."""

    seed: int = 0
    """Shuffle seed. Each epoch adds its index to it."""

    seq_len: int = Field(2048, ge=1)
    """Tokens per row. Documents are packed back to back and continue across rows; nothing is truncated or padded."""

    batch_size: int = Field(256, ge=1)
    """Rows per optimizer step, so a step trains on ``batch_size * seq_len`` tokens."""

    document_masking: bool = True
    """Restrict attention to the current document and restart positions at every document boundary. If false, a row is one sequence."""

    holdout_docs: int = Field(0, ge=0)
    """Hold the last N documents (in dataset order, before shuffling) out of training. With ``[val]`` set, they are evaluated as the ``holdout`` source."""

    @model_validator(mode="after")
    def validate_subsets_and_splits(self):
        if self.subsets is not None and self.splits is not None and len(self.subsets) != len(self.splits):
            raise ValueError("data.subsets and data.splits must have the same length")
        sources = len(self.subsets or self.splits or [None])
        if self.probabilities is not None and len(self.probabilities) != sources:
            raise ValueError("data.probabilities needs one entry per subset/split")
        return self


PretrainDataConfig: TypeAlias = Annotated[FakeDataConfig | TextDataConfig, Field(discriminator="type")]


class ValSourceConfig(BaseConfig):
    name: str
    """Label for the metrics (``val/{name}/loss``, ``/ppl``, ``/bpb``)."""

    dataset: str
    """HF dataset name or local path."""

    subset: str | None = None
    """HF config to load."""

    split: str = "validation"
    """Split to load."""

    revision: str | None = None
    """HF dataset revision."""

    text_column: str = "text"
    """Column holding the raw document text."""

    max_tokens: int = Field(10_000_000, ge=1)
    """Token budget. Documents are taken in dataset order until it is reached."""


class ValConfig(BaseConfig):
    """Perplexity evals: loss, perplexity and bits per byte on fixed held-out token sets.
    Each set is tokenized and packed once at startup, so every eval sees the same tokens."""

    interval: int = Field(500, ge=1)
    """Evaluate every N steps, after that step's optimizer update."""

    eval_on_start: bool = True
    """Evaluate the initial weights before the first step (logged as step 0)."""

    source: list[ValSourceConfig] = []
    """Held-out sets. With ``data.holdout_docs > 0`` the training holdout is added as ``holdout``."""


StepsOrFraction: TypeAlias = Annotated[int, Field(ge=0)] | Annotated[float, Field(gt=0, lt=1)]


def resolve_steps(value: int | float, total_steps: int | None) -> int:
    """Steps as given, or a fraction of ``total_steps``."""
    return round(value * total_steps) if isinstance(value, float) else value


class WSDScheduleConfig(BaseConfig):
    """Warmup, stable at 1, decay. ``f(0) = start_ratio``, the warmup reaches 1 at its last update
    (``warmup - 1``; a warmup of 0 or 1 starts at 1), and the decay reaches ``end_ratio`` at the last
    update of the run."""

    type: Literal["wsd"] = "wsd"

    warmup: StepsOrFraction = 0.1
    """Warmup updates, or a fraction of ``max_steps``."""

    decay: StepsOrFraction = 0.9
    """Decay updates at the end of the run, or a fraction of ``max_steps``. The default decays over all
    updates after the default warmup."""

    start_ratio: float = Field(0.0, ge=0)
    """Multiplier of the first update."""

    end_ratio: float = Field(0.0, ge=0)
    """Multiplier of the last update."""

    decay_shape: Literal["linear", "cosine", "1-sqrt"] = "linear"
    """Shape of the decay from 1 to ``end_ratio``."""


class CosineScheduleConfig(BaseConfig):
    """Linear warmup, then cosine decay from 1 to ``end_ratio`` over the remaining updates."""

    type: Literal["cosine"] = "cosine"

    warmup: StepsOrFraction = 0.1
    """Warmup updates, or a fraction of ``max_steps``."""

    start_ratio: float = Field(0.0, ge=0)
    """Multiplier of the first update."""

    end_ratio: float = Field(0.0, ge=0)
    """Multiplier of the last update."""


class ConstantScheduleConfig(BaseConfig):
    """Multiplier 1 at every step."""

    type: Literal["constant"] = "constant"


class CustomScheduleConfig(BaseConfig):
    type: Literal["custom"] = "custom"

    import_path: str
    """Import path to the schedule function, called as ``fn(step, total_steps=max_steps, **kwargs)`` and returning the multiplier."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the schedule function."""


ScheduleConfig: TypeAlias = Annotated[
    WSDScheduleConfig | CosineScheduleConfig | ConstantScheduleConfig | CustomScheduleConfig,
    Field(discriminator="type"),
]


class SchedulesConfig(BaseConfig):
    """Each schedule is a multiplier ``f(step)``: update ``step`` (counted from 0) uses ``base * f(step)``,
    where ``base`` is the configured value of each param group."""

    optim: dict[str, ScheduleConfig] = Field(default_factory=lambda: {"lr": WSDScheduleConfig()})
    """Schedules for optimizer hyperparameters, keyed by their ``optim`` field name (``lr``, ``weight_decay``, ``betas1``, ``betas2``, and ``mu`` for Muon). Unscheduled ones stay constant."""

    clip_norm: ScheduleConfig | None = None
    """Schedule for the gradient clipping norm ``optim.max_norm``. If None, it stays constant."""


class PretrainDeploymentConfig(BaseConfig):
    num_train_gpus: int = Field(1, ge=1)
    """GPUs for the trainer, on this node."""


class PretrainConfig(BaseConfig):
    model: PretrainModelConfig = PretrainModelConfig()

    env_vars: EnvVars = {}
    """Extra environment variables for the trainer processes."""

    tokenizer: TokenizerConfig = TokenizerConfig()

    data: PretrainDataConfig = TextDataConfig()

    val: ValConfig | None = None
    """Perplexity evals. If None, none run."""

    optim: OptimizerConfig = AdamWConfig()

    schedules: SchedulesConfig = SchedulesConfig()
    """Hyperparameter schedules. Defaults to a WSD learning rate schedule."""

    ckpt: CheckpointConfig | None = CheckpointConfig()
    """Checkpointing. Defaults to a checkpoint at the end of training."""

    resume: ResumeConfig | None = None
    """Resume the run from a checkpoint. If None, does not resume."""

    log: TrainerLogConfig = TrainerLogConfig()

    monitors: TrainMonitorsConfig = TrainMonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``)."""

    run: RunConfig = Field(default_factory=RunConfig)
    """Run metadata. ``run.name`` names the run directory under ``output_dir``."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory that groups related runs. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    clean: bool = False
    """Delete the run directory before starting. Required to overwrite a previous run's artifacts when not resuming."""

    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Precision for float32 matrix multiplications (``torch.set_float32_matmul_precision``)."""

    max_steps: int | None = None
    """Maximum training steps. If None, runs indefinitely."""

    gc: GCConfig | None = GCConfig()
    """Deterministic garbage collection every N steps. Set to null for Python's default GC."""

    dist_timeout_seconds: int = 900
    """Timeout in seconds for torch distributed ops."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration."""

    deployment: PretrainDeploymentConfig = PretrainDeploymentConfig()

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit."""

    @property
    def run_dir(self) -> Path:
        assert self.run.dir is not None  # resolved at construction
        return self.output_dir / self.run.dir

    @model_validator(mode="before")
    @classmethod
    def default_random_init(cls, data):
        """Pretraining starts from randomly initialized weights unless ``model.debug.random_init = false``
        (continued pretraining from ``model.name``)."""
        if isinstance(data, dict):
            model = data.setdefault("model", {})
            if isinstance(model, dict):
                debug = model.setdefault("debug", {})
                if isinstance(debug, dict):
                    debug.setdefault("random_init", True)
        return data

    @model_validator(mode="after")
    def resolve_autos(self):
        """Resolve the shared ``auto`` fields for pretraining: real random init, a trainable fp32
        MoE router, and weight decay 0.1."""
        if self.model.debug.random_init_mode == "auto":
            self.model.debug.random_init_mode = "init"
        if self.model.moe_router_dtype == "auto":
            self.model.moe_router_dtype = "float32"
        if self.model.freeze_moe_router == "auto":
            self.model.freeze_moe_router = False
        if self.optim.weight_decay == "auto":
            self.optim.weight_decay = 0.1
        return self

    @model_validator(mode="after")
    def auto_setup_run_identity(self):
        if self.run.name is None:
            dataset = str(getattr(self.data, "name", "fake")).split("/")[-1]
            model = self.model.name.split("/")[-1]
            self.run.name = "--".join([dataset, model, uuid.uuid4().hex[:8]]).lower()
        if self.run.dir is None:
            self.run.dir = self.run.name
        if self.monitors.wandb is not None and self.monitors.wandb.name is None:
            self.monitors.wandb.name = self.run.name
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def validate_monitors(self):
        if self.monitors.prime is not None:
            raise ValueError("monitors.prime is not supported for pretraining yet.")
        return self

    @model_validator(mode="after")
    def validate_cp(self):
        if self.model.cp > 1:
            if self.data.seq_len % self.model.cp != 0:
                raise ValueError("data.seq_len must be divisible by model.cp")
            if self.data.micro_batch_size != 1:
                raise ValueError("data.micro_batch_size must be 1 with context parallelism")
            if self.val is not None:
                raise ValueError("[val] does not support context parallelism yet")
        return self

    @model_validator(mode="after")
    def validate_holdout(self):
        if self.val is not None and self.data.type == "text" and self.data.holdout_docs == 0 and not self.val.source:
            raise ValueError("[val] needs at least one source, or data.holdout_docs > 0")
        return self

    @model_validator(mode="after")
    def validate_schedules(self):
        unsupported = set(self.schedules.optim) - set(self.optim.schedulable)
        if unsupported:
            raise ValueError(
                f"schedules.optim: {sorted(unsupported)} cannot be scheduled with optim.type = {self.optim.type!r} "
                f"(schedulable: {sorted(self.optim.schedulable)})"
            )
        if self.schedules.clip_norm is not None and self.optim.max_norm is None:
            raise ValueError("schedules.clip_norm needs optim.max_norm")
        named = {f"schedules.optim.{name}": schedule for name, schedule in self.schedules.optim.items()}
        named["schedules.clip_norm"] = self.schedules.clip_norm
        for name, schedule in named.items():
            if isinstance(schedule, WSDScheduleConfig):
                phases = [schedule.warmup, schedule.decay]
                needs_max_steps = schedule.decay != 0 or isinstance(schedule.warmup, float)
            elif isinstance(schedule, CosineScheduleConfig):
                phases = [schedule.warmup]
                needs_max_steps = True
            else:
                continue
            if self.max_steps is None:
                if needs_max_steps:
                    raise ValueError(f"{name} needs max_steps (it decays or has fractional phases)")
            elif sum(resolve_steps(phase, self.max_steps) for phase in phases) > self.max_steps:
                raise ValueError(f"{name}: its phases exceed max_steps ({self.max_steps})")
        return self
