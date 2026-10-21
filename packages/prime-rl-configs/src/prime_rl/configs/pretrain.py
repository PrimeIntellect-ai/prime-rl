import math
import uuid
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

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
    """``optimizer`` keeps the optimizer states in CPU RAM between steps, as in SFT and RL. ``full`` is not supported:
    its CPU optimizer handles only AdamW and SignSGD, without pretraining's per-role parameter groups."""


class TokenSourceConfig(BaseConfig):
    path: Path
    """Folder of datatrove tokenized shards: ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``. The files are read in name order as one token stream."""

    max_tokens: int | None = Field(None, ge=1)
    """Tokens available from this source: the first ``max_tokens`` of its stream. Defaults to all tokens in the shards."""


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
    """Seed for choosing each row's source, the per-epoch order of every source (and the fake tokens)."""

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

    weights: dict[str, float]
    """Token share of each source. Shares are non-negative and sum to 1; unlisted sources get 0."""

    max_epochs: float = Field(1.0, gt=0)
    """Most passes over a source's available tokens that the run (``max_steps``) may make. Checked at startup."""

    @model_validator(mode="after")
    def validate_weights(self):
        if any(weight < 0 for weight in self.weights.values()) or not math.isclose(sum(self.weights.values()), 1.0):
            raise ValueError(f"data.weights must be non-negative and sum to 1, got {self.weights}")
        if unknown := self.weights.keys() - self.sources.keys():
            raise ValueError(f"data.weights name unknown sources {sorted(unknown)}")
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


class EvalsConfig(BaseConfig):
    """The eval suite, scored inside the trainer: forward passes without gradients on the training model, every
    ``interval`` steps. The sets are log-likelihood evals in the format of the pretrain-evals offline suite."""

    interval: int = Field(ge=1)
    """Score the suite every N steps (after that step's optimizer update), and at the last step."""

    eval_on_start: bool = True
    """Also score the weights the run starts from (the initial weights, logged as step 0, or the checkpoint it
    resumes from, logged at that step)."""

    data_dir: Path
    """Eval sets, one directory per set with ``test.parquet`` and ``meta.json``."""

    sets: list[str]
    """Names of the sets to score."""

    seq_lens: list[int] = [2048]
    """Context lengths to score every set at: the window length for ``ppl`` sets and the context limit for
    ``ctxcont`` sets. Metrics are named ``eval/<set>/<metric>@<length>`` (e.g. ``bpb@4k``). Requests that are the
    same at several lengths are scored once."""

    micro_batch_tokens: int | None = Field(None, ge=1)
    """Tokens per packed eval row (one forward pass on one GPU). Defaults to ``data.micro_batch_tokens`` or the
    longest of ``seq_lens``, whichever is larger; must be at least the longest of ``seq_lens``."""

    save_arrays: bool = False
    """Write per-document arrays (npz) and summaries to ``<run_dir>/evals/step_N/``."""


PretrainOptimizerConfig: TypeAlias = Annotated[AdamWConfig | MuonConfig, Field(discriminator="type")]


class PretrainConfig(TrainerLaunchConfig):
    # A trainable router, and weight decay 0.1.
    auto_freeze_moe_router = False
    auto_weight_decay = 0.1

    model: PretrainModelConfig = PretrainModelConfig()
    """The model. ``model.name`` is an HF model id or a local model directory with a ``config.json`` (as in SFT and
    RL); its weights are not loaded: the model is initialized from scratch (``model.init``) or resumed."""

    data: PretrainDataConfig

    val: ValConfig | None = None
    """Held-out loss on token shards (``val/*``). If None, it never runs."""

    evals: EvalsConfig | None = None
    """The eval suite, scored inside the trainer. If None, it never runs."""

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
                raise ValueError("data.micro_batch_tokens must equal data.seq_len with context parallelism")
            if self.evals is not None:
                raise ValueError("[evals] does not support context parallelism yet")
        return self

    @model_validator(mode="after")
    def auto_setup_eval_micro_batch_tokens(self):
        if self.evals is not None:
            longest = max(self.evals.seq_lens)
            if self.evals.micro_batch_tokens is None:
                self.evals.micro_batch_tokens = max(self.data.micro_batch_tokens, longest)
            elif self.evals.micro_batch_tokens < longest:
                raise ValueError(f"evals.micro_batch_tokens must be at least the longest of evals.seq_lens ({longest})")
        return self

    @model_validator(mode="after")
    def auto_setup_model_seq_len(self):
        self.model.seq_len = self.data.seq_len
        return self
