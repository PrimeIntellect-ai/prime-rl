import uuid
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.sft import BaseDataConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    CheckpointConfig,
    InitConfig,
    LinearSchedulerConfig,
    ModelConfig,
    MoERuntimeConfig,
    MuonConfig,
    SchedulerConfig,
    TrainerLaunchConfig,
)
from prime_rl.utils.config import BaseConfig


class MoEBalanceConfig(BaseConfig):
    """MoE load balancing as in DeepSeek-V3: an aux-loss-free selection bias plus a small sequence-wise balance
    loss."""

    bias_update: Literal["sign", "quantile"] = "quantile"
    """``quantile`` (default): quantile balancing (Su 2026; Kimi K3, Marin): every step, each layer's bias is set to
    the value with which every expert would have been selected by ``top_k / num_experts`` of the step's tokens (null
    experts included), estimated as ``quantile_estimator`` sets; no rate. Marin found it
    works best on the router logits (``router_bias_on_logits`` in the model config, on by default). ``sign``: every
    step, each layer's bias moves by ``bias_update_rate`` towards underloaded experts."""

    quantile_estimator: Literal["histogram", "micro_batch_mean"] = "histogram"
    """How ``quantile`` estimates the thresholds. ``histogram`` (default; Kimi K3 report, §2.3.3): per expert, a
    histogram of the step's margins summed over the micro-batches and ranks (one all-reduce per step), read with in-bin
    interpolation; the quantile of the pooled step, independent of how the tokens are split. Its grid is the previous
    step's global margin range (``[-1, 1]`` on the first step after a start or resume). ``micro_batch_mean``: each
    micro-batch's exact quantile, averaged over the micro-batches and ranks."""

    quantile_bins: int = Field(1024, ge=2)
    """Bins of the ``histogram`` estimator's grid."""

    bias_update_rate: float = Field(3e-3, ge=0)
    """Step size of the sign update (gamma): every step, each layer's bias moves by this towards underloaded experts,
    mean-centred. 0 turns the update off."""

    bias_freeze: float = Field(0.1, ge=0, lt=1)
    """Fraction of ``max_steps`` at the end of training during which the biases stop updating, as DeepSeek-V3,
    GLM-4.5 and Moonlight do in their final stage."""

    seq_aux_coeff: float = Field(1e-3, ge=0)
    """Weight of the sequence-wise balance loss (alpha), computed per packed row. 0 turns it off."""


class PretrainMoEConfig(MoERuntimeConfig):
    balance: MoEBalanceConfig = MoEBalanceConfig()
    """Load balancing of the routed experts."""


class PretrainModelConfig(ModelConfig):
    init: InitConfig = InitConfig()
    """Random weight initialization. Used unless the run resumes from a checkpoint."""

    moe: PretrainMoEConfig = PretrainMoEConfig()

    offload: Literal["none", "optimizer"] = "none"
    """``optimizer`` keeps the optimizer states in CPU RAM between steps, as in SFT and RL. ``full`` (CPU optimizer
    steps during backward) is not supported by the pretraining step."""


class BaseBatchConfig(BaseDataConfig):
    seq_len: int = Field(8192, ge=1)
    """Tokens per row. Documents continue across rows; attention restarts at every document and every row."""

    seed: int = 0
    """Seed for the per-epoch window order of the pool (and the fake tokens)."""


class FakeDataConfig(BaseBatchConfig):
    """Uniformly random token ids, for smoke tests."""

    type: Literal["fake"] = "fake"


class TokensDataConfig(BaseBatchConfig):
    """Rows of pre-tokenized datatrove shards: random windows of one token pool. Mixtures are made offline, as shard
    folders."""

    type: Literal["tokens"] = "tokens"

    paths: list[Path] = Field(min_length=1)
    """Folders of datatrove tokenized shards: ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``.
    The files of all folders (in folder order, then file name order) are read as one token pool."""

    max_epochs: float = Field(1.0, gt=0)
    """Most passes over the pool that the run (``max_steps``) may make. Checked at startup."""

    document_masking: bool = True
    """Attention and positions restart at every document. If False, a row is one sequence: attention spans the
    whole row and positions run over it."""


PretrainDataConfig: TypeAlias = Annotated[FakeDataConfig | TokensDataConfig, Field(discriminator="type")]


class TokenEvalSetConfig(BaseConfig):
    """A held-out folder of datatrove tokenized shards, scored like a ``ppl`` set: every document on its own (in
    windows of each of ``seq_lens``), all documents of the folder."""

    name: str
    """Names the metrics, ``eval/<name>/<metric>@<length>``."""

    path: Path
    """Folder of ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``, as for training."""


class EvalsConfig(BaseConfig):
    """The evals, scored inside the trainer: forward passes without gradients on the training model, every
    ``interval`` steps. The sets are log-likelihood evals in the format of the pretrain-evals offline suite, and
    held-out token shards."""

    interval: int = Field(ge=1)
    """Score the sets every N steps (after that step's optimizer update), and at the last step."""

    eval_on_start: bool = True
    """Also score the weights the run starts from (the initial weights, logged as step 0, or the checkpoint it
    resumes from, logged at that step)."""

    data_dir: Path | None = None
    """Directory of the pretrain-evals sets, one directory per set with ``test.parquet`` and ``meta.json``."""

    sets: list[str | TokenEvalSetConfig] = Field(min_length=1)
    """The sets to score: the name of a pretrain-evals set in ``data_dir``, or a held-out token folder."""

    seq_lens: list[int] | None = None
    """Context lengths to score every set at: the window length for ``ppl`` and token sets and the context limit for
    ``ctxcont`` sets. Defaults to ``data.seq_len``. Metrics are named ``eval/<set>/<metric>@<length>`` (e.g.
    ``bpb@4k``). Requests that are the same at several lengths are scored once."""

    micro_batch_tokens: int | None = Field(None, ge=1)
    """Tokens per packed eval row (one forward pass on one GPU). Defaults to the training micro-batch
    (``data.micro_batch_size * data.seq_len``) or the longest of ``seq_lens``, whichever is larger; must be at least
    the longest of ``seq_lens``."""

    save_arrays: bool = False
    """Write per-document arrays (npz) and summaries to ``<run_dir>/evals/step_N/``."""

    @model_validator(mode="after")
    def validate_data_dir(self):
        if self.data_dir is None and any(isinstance(eval_set, str) for eval_set in self.sets):
            raise ValueError("evals.data_dir is required for pretrain-evals sets")
        return self


PretrainOptimizerConfig: TypeAlias = Annotated[AdamWConfig | MuonConfig, Field(discriminator="type")]


class PretrainConfig(TrainerLaunchConfig):
    # A trainable router, and weight decay 0.1.
    auto_freeze_moe_router = False
    auto_weight_decay = 0.1

    model: PretrainModelConfig = PretrainModelConfig()
    """The model. ``model.name`` is an HF model id or a local model directory with a ``config.json`` (as in SFT and
    RL); its weights are not loaded: the model is initialized from scratch (``model.init``) or resumed."""

    data: PretrainDataConfig

    evals: EvalsConfig | None = None
    """The evals, scored inside the trainer. If None, they never run."""

    optim: PretrainOptimizerConfig = MuonConfig()
    """Optimizer. ``optim.lr`` is required."""

    scheduler: SchedulerConfig = LinearSchedulerConfig(warmup_steps=0.1, decay_steps=0.9)
    """Learning-rate schedule. Defaults to a warmup over 10% of the steps and a linear decay to 0 over the rest."""

    ckpt: CheckpointConfig | None = CheckpointConfig()
    """Checkpointing. Defaults to a checkpoint at the end of training."""

    max_steps: int = Field(ge=1)
    """Training steps."""

    def default_run_name(self) -> str:
        return f"{Path(self.model.name).name}--{uuid.uuid4().hex[:8]}"

    @model_validator(mode="after")
    def validate_unsupported(self):
        unsupported = [name for name in ("trace_path", "memory_profiler_path", "control") if getattr(self, name)]
        if unsupported:
            raise ValueError(f"Pretraining does not support {', '.join(unsupported)} yet")
        return self

    @model_validator(mode="after")
    def validate_lr(self):
        if "lr" not in self.optim.model_fields_set:
            raise ValueError("optim.lr is required")
        return self

    @model_validator(mode="after")
    def validate_cp(self):
        if self.model.cp > 1:
            if self.data.seq_len % self.model.cp != 0:
                raise ValueError("data.seq_len must be divisible by model.cp")
            if self.data.micro_batch_size != 1:
                raise ValueError("data.micro_batch_size must be 1 with context parallelism")
            if self.evals is not None:
                raise ValueError("[evals] does not support context parallelism yet")
        return self

    @model_validator(mode="after")
    def auto_setup_evals(self):
        if self.evals is not None:
            if self.evals.seq_lens is None:
                self.evals.seq_lens = [self.data.seq_len]
            longest = max(self.evals.seq_lens)
            if self.evals.micro_batch_tokens is None:
                self.evals.micro_batch_tokens = max(self.data.micro_batch_size * self.data.seq_len, longest)
            elif self.evals.micro_batch_tokens < longest:
                raise ValueError(f"evals.micro_batch_tokens must be at least the longest of evals.seq_lens ({longest})")
        return self
