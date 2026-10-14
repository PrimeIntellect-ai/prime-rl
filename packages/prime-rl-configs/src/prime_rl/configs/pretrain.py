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
    ConstantSchedulerConfig,
    GCConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    validate_scheduler,
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

    scheduler: SchedulerConfig = ConstantSchedulerConfig()

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

    moe_bias_freeze: float = Field(0.1, ge=0, lt=1)
    """Fraction of ``max_steps`` at the end of training during which the MoE selection biases stop updating, as
    DeepSeek-V3, GLM-4.5 and Moonlight do in their final stage. Without ``max_steps`` the biases never freeze."""

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
    def validate_completep(self):
        if self.model.init.scheme != "completep":
            return self
        if self.optim.type not in ("adamw", "muon") or self.model.offload != "none":
            raise ValueError("model.init.scheme = 'completep' needs AdamW or Muon and model.offload = 'none'.")
        if self.scheduler.type == "cosine" and self.scheduler.min_lr > 0:
            # CosineAnnealingLR decays every parameter group to the same absolute min_lr.
            raise ValueError("model.init.scheme = 'completep' needs scheduler.min_lr = 0 with the cosine scheduler.")
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
    def validate_scheduler_steps(self):
        validate_scheduler(self.scheduler, self.max_steps)
        return self
