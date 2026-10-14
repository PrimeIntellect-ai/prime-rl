import math
import uuid
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.monitors import TrainMonitorsConfig
from prime_rl.configs.shared import (
    EnvVars,
    HeartbeatConfig,
    LogConfig,
    ResumeConfig,
    RunConfig,
    SFTDeploymentConfig,
    SingleNodeDeploymentConfig,
    SlurmConfig,
    TrainerLogConfig,
)
from prime_rl.configs.trainer import (
    AdamWConfig,
    CheckpointConfig,
    GCConfig,
    ModelConfig,
    MoERuntimeConfig,
    MuonConfig,
    OffloadMode,
    TokenizerConfig,
)
from prime_rl.utils.config import BaseConfig, default_output_dir, find_package_resource

TOKENIZER_FIELDS = ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id")


class InitConfig(BaseConfig):
    """Random weight initialization: every weight matrix and the embedding from N(0, std), scaled per role under
    CompleteP (see ``[parametrization]``). Norm weights start at 1, biases at 0."""

    std: float = Field(0.02, gt=0)
    """Base standard deviation."""

    seed: int = 0
    """Seed for the initialization. The values depend only on the seed, not on the number of GPUs."""


class ParametrizationConfig(BaseConfig):
    """CompleteP (muP extended to depth, Dey et al. 2025, alpha = 1) relative to a base shape: init std, learning
    rate, Adam eps and weight decay scale per parameter role with ``m_N = hidden_size / base_hidden_size`` and
    ``m_L = num_layers / base_num_layers``, so hyperparameters tuned on the base shape carry over."""

    base_hidden_size: int = Field(ge=1)
    """Hidden size of the base shape."""

    base_num_layers: int = Field(ge=1)
    """Number of layers of the base shape."""


class MoEBalanceConfig(BaseConfig):
    """MoE load balancing as in DeepSeek-V3: an aux-loss-free selection bias plus a small sequence-wise balance
    loss."""

    bias_update_rate: float = Field(3e-3, ge=0)
    """Step size of the selection-bias update (gamma): every step, each layer's bias moves by this towards
    underloaded experts, mean-centred. 0 turns the update off."""

    bias_freeze: float = Field(0.1, ge=0, lt=1)
    """Fraction of ``max_steps`` at the end of training during which the biases stop updating, as DeepSeek-V3,
    GLM-4.5 and Moonlight do in their final stage."""

    seq_aux_coeff: float = Field(1e-3, ge=0)
    """Weight of the sequence-wise balance loss (alpha), computed per packed row. 0 turns it off."""


class PretrainMoEConfig(MoERuntimeConfig):
    balance: MoEBalanceConfig = MoEBalanceConfig()
    """Load balancing of the routed experts."""


class PretrainModelConfig(ModelConfig):
    arch: dict[str, Any]
    """The architecture: ``model_type`` (a model in ``prime_rl.trainer.models``, e.g. ``glm4_moe``) and the fields of its config class. ``vocab_size`` and the special token ids come from the tokenizer. The launcher writes the config and the tokenizer to ``<run_dir>/configs/model``, which ``name`` points at."""

    init: InitConfig = InitConfig()
    """Random weight initialization. Used unless the run resumes from a checkpoint."""

    moe: PretrainMoEConfig = PretrainMoEConfig()

    offload: OffloadMode = "none"
    """CPU offloading. Pretraining's per-role parameter groups need ``none``."""

    @model_validator(mode="after")
    def validate_arch(self):
        if "model_type" not in self.arch:
            raise ValueError("model.arch needs a model_type, e.g. 'glm4_moe'")
        if derived := [key for key in TOKENIZER_FIELDS if key in self.arch]:
            raise ValueError(f"model.arch must not set {derived}: they come from the tokenizer")
        return self


class TokenSourceConfig(BaseConfig):
    path: Path
    """Folder of datatrove tokenized shards: ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``. The files are read in name order as one token stream."""

    max_tokens: int | None = Field(None, ge=1)
    """Tokens available from this source: the first ``max_tokens`` of its stream. Defaults to all tokens in the shards."""


class MixturePhaseConfig(BaseConfig):
    start: float = Field(0.0, ge=0, lt=1)
    """Fraction of ``max_steps`` at which this phase starts."""

    weights: dict[str, float]
    """Token share of each source in this phase. Shares are non-negative and sum to 1; unlisted sources get 0."""

    @model_validator(mode="after")
    def validate_weights(self):
        if any(weight < 0 for weight in self.weights.values()) or not math.isclose(sum(self.weights.values()), 1.0):
            raise ValueError(f"Phase weights must be non-negative and sum to 1, got {self.weights}")
        return self


class BaseBatchConfig(BaseConfig):
    seq_len: int = Field(4096, ge=1)
    """Tokens per row. Documents continue across rows; attention restarts at every document and every row."""

    batch_tokens: int = Field(ge=1)
    """Tokens per optimizer step, over all GPUs. Fixed when ``seq_len`` changes, so the number of steps stays the same."""

    micro_batch_tokens: int = Field(ge=1)
    """Tokens per forward pass on one GPU."""

    num_workers: int = Field(1, ge=1)
    """Dataloader worker processes per GPU."""

    seed: int = 0
    """Seed for choosing each row's source (and the fake tokens)."""

    @property
    def batch_size(self) -> int:
        """Rows per optimizer step."""
        return self.batch_tokens // self.seq_len

    @property
    def micro_batch_size(self) -> int:
        """Rows per forward pass."""
        return self.micro_batch_tokens // self.seq_len

    @model_validator(mode="after")
    def validate_batch_tokens(self):
        if self.batch_tokens % self.seq_len or self.micro_batch_tokens % self.seq_len:
            raise ValueError("data.batch_tokens and data.micro_batch_tokens must be multiples of data.seq_len")
        if self.batch_tokens % self.micro_batch_tokens:
            raise ValueError("data.batch_tokens must be a multiple of data.micro_batch_tokens")
        return self


class FakeDataConfig(BaseBatchConfig):
    """Uniformly random token ids, for smoke tests."""

    type: Literal["fake"] = "fake"


class TokensDataConfig(BaseBatchConfig):
    """Rows of pre-tokenized datatrove shards, each row drawn from one source of a weighted mixture."""

    type: Literal["tokens"] = "tokens"

    sources: dict[str, TokenSourceConfig]
    """Token sources by name. The name labels the source's metrics."""

    phases: list[MixturePhaseConfig]
    """Mixture phases in order. The first starts at 0."""

    max_epochs: float = Field(1.0, gt=0)
    """Most passes over a source's available tokens that the run (``max_steps``) may make. Checked at startup."""

    @model_validator(mode="after")
    def validate_phases(self):
        starts = [phase.start for phase in self.phases]
        if not starts or starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError(f"Phase starts must begin at 0 and increase, got {starts}")
        for phase in self.phases:
            if unknown := phase.weights.keys() - self.sources.keys():
                raise ValueError(f"Phase weights name unknown sources {sorted(unknown)}")
        return self


PretrainDataConfig: TypeAlias = Annotated[FakeDataConfig | TokensDataConfig, Field(discriminator="type")]


class ValConfig(BaseConfig):
    """Held-out loss, perplexity and bits per byte, computed like the training loss on held-out token shards."""

    interval: int = Field(500, ge=1)
    """Evaluate every N steps, after that step's optimizer update."""

    eval_on_start: bool = True
    """Evaluate the initial weights before the first step (logged as step 0)."""

    sources: dict[str, TokenSourceConfig]
    """Held-out sources by name, logged as ``val/<name>/loss``, ``/ppl`` and ``/bpb``. Each eval reads the same
    windows: as many as fill whole micro-batches on every data-parallel rank."""


StepsOrFraction: TypeAlias = Annotated[int, Field(ge=0)] | Annotated[float, Field(gt=0, lt=1)]


def resolve_steps(value: int | float, total_steps: int) -> int:
    """Steps as given, or a fraction of ``total_steps`` rounded down, so fractions that sum to at most 1 fit."""
    return math.floor(value * total_steps) if isinstance(value, float) else value


class ScheduleConfig(BaseConfig):
    """Learning-rate multiplier ``f(step)`` (warmup, stable at 1, decay), where ``step`` counts the updates
    completed before the current one. Every param group uses its own learning rate times ``f(step)``, so
    ratios between groups hold. ``f(0) = 0``, the warmup reaches 1 at its last update (``warmup - 1``; a
    warmup of 0 or 1 starts at 1), and the decay reaches ``end_ratio`` at the last update of the run."""

    warmup: StepsOrFraction = 0.1
    """Warmup updates, or a fraction of ``max_steps``."""

    decay: StepsOrFraction = 0.9
    """Decay updates at the end of the run, or a fraction of ``max_steps``. The default decays over all
    updates after the default warmup."""

    end_ratio: float = Field(0.0, ge=0)
    """Multiplier of the last update."""

    decay_shape: Literal["linear", "cosine", "1-sqrt"] = "linear"
    """Shape of the decay from 1 to ``end_ratio``."""


class DiagnosticsConfig(BaseConfig):
    """Diagnostic step: on every ``interval``-th step, the optimizer update's size per parameter group and a
    forward pass on a fixed probe batch (the first ``probe_tokens`` tokens of each rank's first micro-batch)
    that records attention max logits and entropy per head and the residual-stream size per layer."""

    interval: int = Field(100, ge=1)
    """Run the diagnostics every N steps."""

    probe_tokens: int = Field(2048, ge=64)
    """Tokens per rank in the probe batch. Fixed across sequence lengths so attention entropy stays comparable."""


class EvalSnapshotsConfig(BaseConfig):
    """Model-only snapshots for the asynchronous eval worker (`prime_rl.trainer.pretrain.eval_worker`), written in
    the background to ``<run_dir>/snapshots/step_N``. The worker deletes each one after scoring it."""

    interval: int = Field(ge=1)
    """Write a snapshot every N steps, and at the final step."""


class EvalSuiteConfig(BaseConfig):
    """What the asynchronous eval worker scores."""

    data_dir: Path
    """Eval sets, one directory per set with ``test.parquet`` and ``meta.json``."""

    sets: list[str]
    """Names of the sets to score."""

    seq_lens: list[int] = [2048]
    """Context lengths to score every set at: the window length for ``ppl`` sets and the context limit for
    ``ctxcont`` sets. Metrics are named ``eval/<set>/<metric>@<length>`` (e.g. ``bpb@4k``). Requests that are the
    same at several lengths are scored once."""

    token_budget: int = Field(32768, ge=1)
    """Tokens per packed row (one forward). At least the longest context length plus one."""

    poll_seconds: int = Field(30, ge=1)
    """Seconds to wait before looking for a new snapshot again."""

    save_arrays: bool = True
    """Write per-document arrays (npz) and summaries to ``<run_dir>/evals/step_N/``."""

    @model_validator(mode="after")
    def validate_token_budget(self):
        if self.token_budget <= max(self.seq_lens):
            raise ValueError("token_budget must exceed the longest of seq_lens")
        return self


class EvalWorkerConfig(EvalSuiteConfig):
    """The asynchronous eval worker (``python -m prime_rl.trainer.pretrain.eval_worker``): scores the newest snapshot
    of a training run on the eval suite, logs the results into the run's monitors at the snapshot's step, and deletes
    scored snapshots. The pretrain launcher writes this config and starts the worker for ``[eval_worker]``."""

    run_dir: Path
    """The training run's directory (with ``configs/latest/resolved/pretrain.json`` and ``snapshots/``)."""

    log: LogConfig = LogConfig()


class PretrainEvalWorkerConfig(EvalSuiteConfig):
    """The eval worker, started by the launcher next to the trainer on GPUs of its own."""

    num_gpus: int = Field(1, ge=1)
    """Single-node runs: GPUs of the node for the worker, next to the trainer's ``deployment.num_train_gpus``."""

    num_nodes: int = Field(1, ge=1)
    """Multi-node runs: nodes for the worker, next to the trainer's ``deployment.num_train_nodes``."""


class PretrainAdamWConfig(AdamWConfig):
    eps: float = Field(1e-8, gt=0)
    """Adam epsilon of the base shape (scaled per role under CompleteP)."""


class PretrainMuonConfig(MuonConfig):
    eps: float = Field(1e-8, gt=0)
    """Adam epsilon of the base shape for the AdamW groups (scaled per role under CompleteP)."""


PretrainOptimizerConfig: TypeAlias = Annotated[PretrainAdamWConfig | PretrainMuonConfig, Field(discriminator="type")]


class PretrainConfig(BaseConfig):
    model: PretrainModelConfig

    env_vars: EnvVars = {}
    """Extra environment variables for the trainer processes."""

    tokenizer: TokenizerConfig
    """The tokenizer (``tokenizer.name`` is required). It sets the vocabulary and special tokens of the model."""

    data: PretrainDataConfig

    val: ValConfig | None = None
    """Held-out evals. If None, none run."""

    diagnostics: DiagnosticsConfig | None = DiagnosticsConfig()
    """Diagnostic step (update sizes, attention and activation probe). If None, it never runs."""

    eval_snapshots: EvalSnapshotsConfig | None = None
    """Model-only snapshots for the asynchronous eval worker. If None, none are written."""

    eval_worker: PretrainEvalWorkerConfig | None = None
    """The asynchronous eval worker, started by the launcher on GPUs of its own. Needs ``[eval_snapshots]``. If None,
    none is started (a worker can still be started by hand)."""

    optim: PretrainOptimizerConfig = PretrainMuonConfig()
    """Optimizer. ``optim.lr`` is required."""

    schedule: ScheduleConfig = ScheduleConfig()
    """Learning-rate schedule. Defaults to a warmup over 10% of the steps and a linear decay to 0 over the rest."""

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

    max_steps: int = Field(ge=1)
    """Training steps."""

    parametrization: ParametrizationConfig | None = None
    """CompleteP relative to a base shape. If None, the standard parametrization: the same per-role parameter
    groups with every multiplier at 1 (norms and biases without weight decay)."""

    gc: GCConfig | None = GCConfig()
    """Deterministic garbage collection every N steps. Set to null for Python's default GC."""

    dist_timeout_seconds: int = 900
    """Timeout in seconds for torch distributed ops."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration."""

    deployment: SFTDeploymentConfig = SingleNodeDeploymentConfig()
    """``single_node`` (``num_train_gpus`` on this node) or ``multi_node`` (``num_train_nodes`` x ``gpus_per_node``, needs ``[slurm]``). The online-eval fields (``num_infer_*``) are not used."""

    slurm: SlurmConfig | None = None
    """SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit."""

    @property
    def run_dir(self) -> Path:
        assert self.run.dir is not None  # resolved at construction
        return self.output_dir / self.run.dir

    @model_validator(mode="after")
    def resolve_autos(self):
        """Resolve the shared ``auto`` fields for pretraining: a trainable fp32 MoE router and weight decay 0.1."""
        if self.model.moe_router_dtype == "auto":
            self.model.moe_router_dtype = "float32"
        if self.model.freeze_moe_router == "auto":
            self.model.freeze_moe_router = False
        if self.optim.weight_decay == "auto":
            self.optim.weight_decay = 0.1
        return self

    @model_validator(mode="after")
    def validate_lr(self):
        if "lr" not in self.optim.model_fields_set:
            raise ValueError("optim.lr is required")
        return self

    @model_validator(mode="after")
    def auto_setup_run_identity(self):
        if self.run.name is None:
            self.run.name = f"{self.model.arch['model_type']}--{uuid.uuid4().hex[:8]}"
        if self.run.dir is None:
            self.run.dir = self.run.name
        if self.monitors.wandb is not None and self.monitors.wandb.name is None:
            self.monitors.wandb.name = self.run.name
        return self

    @model_validator(mode="after")
    def auto_setup_model_dir(self):
        """The model is trained from ``<run_dir>/configs/model``, written by the launcher from ``model.arch``."""
        self.model.name = (self.run_dir / "configs" / "model").as_posix()
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            raise ValueError("tokenizer.name is required")
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        """A multi-node deployment needs SLURM and sets ``model.dp_replicate`` from ``nodes_per_fsdp_group``; the
        SLURM job name defaults to the run name and the template to the bundled single- or multi-node template."""
        deployment = self.deployment
        if deployment.type == "multi_node":
            if self.slurm is None:
                raise ValueError("Must use SLURM for multi-node deployment.")
            if deployment.nodes_per_fsdp_group is not None:
                if deployment.num_train_nodes % deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_train_nodes ({deployment.num_train_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({deployment.nodes_per_fsdp_group})"
                    )
                self.model.dp_replicate = deployment.num_train_nodes // deployment.nodes_per_fsdp_group
        if self.slurm is not None:
            if self.slurm.job_name is None:
                self.slurm.job_name = self.run.name
            if self.slurm.template_path is None:
                templates_dir = find_package_resource("templates")
                if templates_dir is not None:
                    self.slurm.template_path = templates_dir / f"{deployment.type}_sft.sbatch.j2"
        return self

    @model_validator(mode="after")
    def deepep_disables_grad_clipping(self):
        if self.model.ep != 1 and self.model.moe.dispatch.type == "deepep" and self.optim.max_norm is not None:
            warnings.warn(
                "Gradient clipping is not compatible with DeepEP. "
                "Automatically setting optim.max_norm to None (disabled).",
                stacklevel=1,
            )
            self.optim.max_norm = None
        return self

    @model_validator(mode="after")
    def validate_offload(self):
        if self.model.offload != "none":
            raise ValueError("Pretraining's per-role parameter groups need model.offload = 'none'.")
        return self

    @model_validator(mode="after")
    def validate_eval_worker(self):
        if self.eval_worker is not None and self.eval_snapshots is None:
            raise ValueError("[eval_worker] needs [eval_snapshots]")
        return self

    @model_validator(mode="after")
    def validate_cp(self):
        if self.model.cp > 1:
            if self.data.seq_len % self.model.cp != 0:
                raise ValueError("data.seq_len must be divisible by model.cp")
            if self.data.micro_batch_size != 1:
                raise ValueError("data.micro_batch_tokens must equal data.seq_len with context parallelism")
            if self.diagnostics is not None:
                raise ValueError("[diagnostics] does not support context parallelism yet; set diagnostics = null")
        return self

    @model_validator(mode="after")
    def validate_schedule(self):
        phases = resolve_steps(self.schedule.warmup, self.max_steps) + resolve_steps(
            self.schedule.decay, self.max_steps
        )
        if phases > self.max_steps:
            raise ValueError(f"schedule.warmup + schedule.decay ({phases}) exceed max_steps ({self.max_steps})")
        return self
