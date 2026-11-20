from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.supervised import BaseDataConfig, BaseSourceConfig, SupervisedTrainerConfig, check_sources
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


class TokenSourceConfig(BaseSourceConfig):
    """A folder of datatrove tokenized shards, one source of the training data."""

    path: Path
    """Folder of ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``."""

    repetitions: float | None = Field(None, ge=1)
    """How often every token of a fixed subset of the source is seen over the run (needs ``max_steps``). If None, the
    whole source is read."""


class PretrainDataConfig(BaseDataConfig):
    """Training shards: documents sampled from one or more sources into rows of ``seq_len`` tokens."""

    type: Literal["datatrove"] = "datatrove"

    source: list[TokenSourceConfig] = Field(min_length=1)
    """The folders to sample documents from (``[[data.source]]``)."""

    seed: int = Field(0, ge=0)
    """Seed for the order of the documents, within rows too, and for the subsets."""

    document_masking: bool = True
    """Attention and positions restart at every document and piece. If False, a row is one sequence."""

    eod_mask_loss: bool = True
    """Don't train the model to predict the first token of the next, unrelated document: no loss where the input is
    the document separator. The separator is the special token the training shards' documents end with, detected at
    startup; startup fails if they don't all end with one, unless they start with BOS (then there is nothing to
    mask)."""

    @model_validator(mode="after")
    def validate_sources(self):
        check_sources(self.source)
        return self


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

    optim: PretrainOptimizerConfig = PretrainAdamWConfig()

    scheduler: PretrainSchedulerConfig = PretrainLinearSchedulerConfig()
    """Learning rate schedule. Defaults to a linear warmup over 10 steps, then constant: from a random init, Adam's
    first updates at the full learning rate are unstable."""

    @model_validator(mode="after")
    def validate_repetitions(self):
        if self.max_steps is None and any(source.repetitions is not None for source in self.data.source):
            raise ValueError(
                "data.source: repetitions sizes a source's subset by the run's tokens, so it needs max_steps"
            )
        return self
