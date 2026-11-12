from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from prime_rl.configs.supervised import BaseDataConfig, BaseValConfig, SupervisedTrainerConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
    InitConfig,
    LinearSchedulerConfig,
    ModelConfig,
    MuonConfig,
    SignSGDConfig,
)


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


class PretrainValConfig(BaseValConfig):
    data: DatatroveDataConfig
    """Held-out shard folders. Every validation scores each of their documents on its own, from its first token, in
    non-overlapping windows of ``seq_len`` scored tokens (a longer document continues in the next window, without the
    tokens before). Windows are packed into rows of ``seq_len * micro_batch_size`` tokens; ``seed`` and
    ``document_masking`` do not apply."""

    num_samples: int | None = Field(None, ge=1)
    """Documents per validation: the first ``num_samples`` documents of the held-out shards (in folder order, then file
    name order), the same ones every time. If None, every document."""


class PretrainAdamWConfig(AdamWConfig):
    lr: float = Field(1e-3, ge=0)
    """Peak learning rate. Defaults to PyTorch's AdamW default."""


class PretrainMuonConfig(MuonConfig):
    lr: float = Field(1e-3, ge=0)
    """Peak learning rate. Defaults to 1e-3, as for AdamW."""


PretrainOptimizerConfig: TypeAlias = Annotated[
    PretrainAdamWConfig | PretrainMuonConfig | SignSGDConfig, Field(discriminator="type")
]


class PretrainLinearSchedulerConfig(LinearSchedulerConfig):
    decay_steps: int = Field(0, ge=0)
    """Steps to decay the learning rate during the final portion of training. Defaults to 0: a decay needs
    ``max_steps``."""


PretrainSchedulerConfig: TypeAlias = Annotated[
    ConstantSchedulerConfig | PretrainLinearSchedulerConfig | CosineSchedulerConfig, Field(discriminator="type")
]


class PretrainInitConfig(InitConfig):
    type: Literal["scratch", "hf", "ckpt"] = "scratch"


class PretrainModelConfig(ModelConfig):
    init: PretrainInitConfig = PretrainInitConfig()


class PretrainConfig(SupervisedTrainerConfig):
    """Pretraining: next-token loss on tokenized shards, with the weights initialized from scratch by default."""

    model: PretrainModelConfig = PretrainModelConfig()

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
