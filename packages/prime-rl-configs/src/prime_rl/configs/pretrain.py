import math
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.supervised import BaseDataConfig, BaseValConfig, SupervisedTrainerConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
    InitConfig,
    LinearSchedulerConfig,
    ModelConfig,
    MuonConfig,
    ScratchInit,
    SignSGDConfig,
)
from prime_rl.utils.config import BaseConfig


class DatatroveDataConfig(BaseDataConfig):
    """Held-out shards written by datatrove's ``DocumentTokenizer``."""

    type: Literal["datatrove"] = "datatrove"

    paths: list[Path] = Field(min_length=1)
    """Folders of datatrove tokenized shards: ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``,
    read in folder order, then file name order."""


class TokenSourceConfig(BaseConfig):
    """A folder of datatrove tokenized shards, one source of the training data."""

    name: str
    """Names the source in the ``progress/<name>/*`` metrics."""

    path: Path
    """Folder of ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``."""

    weight: float | None = Field(None, ge=0)
    """The source's share of the training tokens. If None on every source, its share of all sources' tokens."""

    repetitions: float | None = Field(None, ge=1)
    """How often every token of a fixed subset of the source is seen over the run (needs ``max_steps``). If None, the
    whole source is read."""


class PretrainDataConfig(BaseDataConfig):
    """Training shards: documents sampled from one or more sources into rows of ``seq_len`` tokens."""

    type: Literal["datatrove"] = "datatrove"

    sources: list[TokenSourceConfig] = Field(min_length=1)
    """The folders to sample documents from."""

    seed: int = Field(0, ge=0)
    """Seed for the order of the documents and the subsets."""

    document_masking: bool = True
    """Attention and positions restart at every document. If False, a row is one sequence."""

    eod_mask_loss: bool = True
    """Don't train the model to predict the first token of the next, unrelated document: no loss where the input is
    the document separator. The separator is the special token the training shards' documents end with, detected at
    startup; startup fails if they don't all end with one, unless they start with BOS (then there is nothing to
    mask)."""

    @model_validator(mode="after")
    def validate_sources(self):
        names = [source.name for source in self.sources]
        if len(set(names)) < len(names):
            raise ValueError(f"data.sources: the names must be unique, got {names}")
        weights = [source.weight for source in self.sources]
        if None in weights and any(weight is not None for weight in weights):
            raise ValueError("data.sources: set weight on every source or on none (natural shares)")
        if None not in weights and not math.isclose(sum(weights), 1, abs_tol=1e-6):
            raise ValueError(f"data.sources: the weights sum to {sum(weights)}, not 1")
        return self


class PretrainValConfig(BaseValConfig):
    data: DatatroveDataConfig
    """Held-out shard folders. Every validation scores each of their documents on its own, from its first token, in
    non-overlapping windows of ``seq_len`` scored tokens (a longer document continues in the next window, without the
    tokens before). Windows are packed into rows of ``seq_len * micro_batch_size`` tokens."""

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


class PretrainModelConfig(ModelConfig):
    init: InitConfig = ScratchInit()


class PretrainConfig(SupervisedTrainerConfig):
    """Pretraining: next-token loss on tokenized shards, with the weights initialized from scratch by default."""

    model: PretrainModelConfig = PretrainModelConfig()

    mode: Literal["pretrain"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run pretrain`` accepts only
    "pretrain" (or no mode)."""

    data: PretrainDataConfig

    val: PretrainValConfig | None = None
    """Validation on held-out shards. If None, no validation runs."""

    optim: PretrainOptimizerConfig = PretrainAdamWConfig()

    scheduler: PretrainSchedulerConfig = PretrainLinearSchedulerConfig()
    """Learning rate schedule. Defaults to a linear warmup over 10 steps, then constant: from a random init, Adam's
    first updates at the full learning rate are unstable."""

    @model_validator(mode="after")
    def validate_repetitions(self):
        if self.max_steps is None and any(source.repetitions is not None for source in self.data.sources):
            raise ValueError(
                "data.sources: repetitions sizes a source's subset by the run's tokens, so it needs max_steps"
            )
        return self
