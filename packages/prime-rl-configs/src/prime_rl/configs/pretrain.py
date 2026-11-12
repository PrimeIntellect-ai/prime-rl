from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from prime_rl.configs.supervised import BaseDataConfig, BaseValConfig, SupervisedTrainerConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    BaseOptimizerConfig,
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
    LinearSchedulerConfig,
    MuonConfig,
    SignSGDConfig,
)
from prime_rl.utils.config import BaseConfig


class DatatroveDataConfig(BaseDataConfig):
    """Tokenized shards written by datatrove's ``DocumentTokenizer``, read as one token pool and cut into samples of
    ``seq_len`` tokens."""

    type: Literal["datatrove"] = "datatrove"

    paths: list[Path] = Field(min_length=1)
    """Folders of datatrove tokenized shards: ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``.
    The files of all folders (in folder order, then file name order) are read as one token pool."""

    seed: int = 0
    """Seed for the sample order: each epoch reads every ``seq_len`` window of the pool once, in a new random order
    and from a new random offset."""

    document_masking: bool = True
    """Attention and positions restart at every document. If False, a sample is one sequence: attention spans the
    whole sample and positions run over it."""

    packed_rows: bool = False
    """Every document of the shards is one pre-packed row of ``seq_len + 1`` tokens. Rows are read consecutively in
    shard order without shuffling, row ``r`` by data rank ``r % data_world_size``, one input/target pair per row.
    With ``seed`` other than 0, rows are read in a pseudo-random order drawn from the seed instead, shuffled
    within blocks of 2^20 rows."""

    mask_target_id: int | None = None
    """Exclude targets equal to this token id from the loss. For metrics that count every token but this one."""


class PretrainValConfig(BaseValConfig):
    data: DatatroveDataConfig
    """Held-out shard folders."""

    num_samples: int | None = Field(None, ge=1)
    """Windows per validation, the same ones every time. If None, one full pass over the held-out pool."""

    bpb: bool = False
    """Also log ``val/bpb``, nanochat's bits per byte: summed nats over targets whose token has more than 0 bytes,
    divided by the summed bytes and ln 2 (``nanochat/loss_eval.py:evaluate_bpb``)."""

    final_bpb: bool = False
    """Validate once more after the last optimizer step and log it as ``eval/final_bpb`` (and ``val/*``)."""

    core: bool = False
    """Run nanochat's CORE metric (``nanochat/core_eval.py``) on the final weights, logged as ``eval/core`` and
    per-task ``core/<task>`` (centered) and ``core_acc/<task>``."""

    core_max_per_task: int = -1
    """Examples per CORE task (nanochat's ``--core-metric-max-per-task``); -1 uses every example."""

    eval_bundle: str = "/volume/eval_bundle"
    """Folder of the extracted CORE eval bundle (``core.yaml``, ``eval_meta_data.csv``, ``eval_data/``)."""

    core_pack_tokens: int = Field(16384, ge=2048)
    """Tokens per packed CORE forward."""


class PretrainAdamWConfig(AdamWConfig):
    lr: float = Field(1e-3, ge=0)
    """Peak learning rate. Defaults to PyTorch's AdamW default."""


class PretrainMuonConfig(MuonConfig):
    lr: float = Field(1e-3, ge=0)
    """Peak learning rate. Defaults to 1e-3, as for AdamW."""


class NanochatMuonAdamWConfig(BaseOptimizerConfig):
    """nanochat's combined optimizer (``nanochat/optim.py`` @ ff46300): Muon on the block matrices with polar-express
    orthogonalization, NorMuon variance reduction and cautious weight decay, AdamW on embeddings, unembedding and
    per-layer scalars. Needs fully replicated weights, so ``model.dp_replicate`` must equal the number of GPUs."""

    type: Literal["nanochat_muon"] = "nanochat_muon"

    lr: float = Field(1.0, gt=0)
    """Unused: every parameter group has its own learning rate below. The scheduler scales them all."""

    max_norm: float | None = None

    weight_decay: float = Field(0.0, ge=0)
    """Muon cautious weight decay at step 0, scaled down linearly to 0 over ``total_steps``."""

    matrix_lr: float = Field(0.02, gt=0)
    """Muon learning rate of the block matrices."""

    embedding_lr: float = Field(0.3, gt=0)
    """AdamW learning rate of the token and value embeddings."""

    unembedding_lr: float = Field(0.004, gt=0)
    """AdamW learning rate of the LM head."""

    scalar_lr: float = Field(0.5, gt=0)
    """AdamW learning rate of the per-layer scalars: ``x0_lambda`` gets ``scalar_lr``, ``resid_lambda`` 1% of it."""

    extra_scalar_lr: float = Field(0.02, gt=0)
    """AdamW learning rate of the 1-D parameters of the residual connection."""

    connection_matrix_optimizer: Literal["adamw", "muon"] = "adamw"
    """Optimizer of the residual connection's matrices (the gated residual's low-rank projections)."""

    connection_lr: float = Field(0.004, gt=0)
    """AdamW learning rate of the residual connection's matrices."""

    adam_betas: tuple[float, float] = (0.8, 0.95)

    x0_beta1: float = Field(0.96, ge=0)

    total_steps: int = Field(ge=1)
    """Number of optimizer steps of the run: the horizon of the weight decay schedule."""


PretrainOptimizerConfig: TypeAlias = Annotated[
    PretrainAdamWConfig | PretrainMuonConfig | SignSGDConfig | NanochatMuonAdamWConfig, Field(discriminator="type")
]


class PretrainLinearSchedulerConfig(LinearSchedulerConfig):
    decay_steps: int = Field(0, ge=0)
    """Steps to decay the learning rate during the final portion of training. Defaults to 0: a decay needs
    ``max_steps``."""


class PretrainWarmdownSchedulerConfig(BaseConfig):
    """Nanochat's schedule: linear warmup, flat, linear warmdown, as fractions of ``max_steps``."""

    type: Literal["warmdown"] = "warmdown"

    warmup_ratio: float = Field(0.0, ge=0, le=1)

    warmdown_ratio: float = Field(0.5, ge=0, le=1)

    final_lr_frac: float = Field(0.0, ge=0)


PretrainSchedulerConfig: TypeAlias = Annotated[
    ConstantSchedulerConfig | PretrainLinearSchedulerConfig | CosineSchedulerConfig | PretrainWarmdownSchedulerConfig,
    Field(discriminator="type"),
]


class PretrainConfig(SupervisedTrainerConfig):
    """Pretraining: next-token loss on tokenized shards, with the weights initialized from scratch unless resuming."""

    init_from_scratch = True

    mode: Literal["pretrain"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run pretrain`` accepts only
    "pretrain" (or no mode)."""

    data: DatatroveDataConfig

    val: PretrainValConfig | None = None
    """Validation on held-out shards. If None, no validation runs."""

    optim: PretrainOptimizerConfig = PretrainAdamWConfig()

    scheduler: PretrainSchedulerConfig = PretrainLinearSchedulerConfig()
    """Learning rate schedule. Defaults to a linear warmup over 10 steps, then constant: from a random init, Adam's
    first updates at the full learning rate are unstable."""
