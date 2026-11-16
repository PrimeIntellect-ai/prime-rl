from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, ValidationInfo, field_validator, model_validator

from prime_rl.configs.supervised import BaseDataConfig, BaseValConfig, SupervisedTrainerConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
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


class PretrainDataConfig(DatatroveDataConfig):
    """Training shards."""

    eod_mask_loss: bool = True
    """Don't train the model to predict the first token of the next, unrelated document: no loss where the input is
    the document separator. The separator is the special token the training shards' documents end with, detected at
    startup; startup fails if they don't all end with one, unless they start with BOS (then there is nothing to
    mask)."""


class HeldOutDataConfig(BaseConfig):
    name: str
    """Names the metrics: ``val/<name>/loss`` and ``val/<name>/bpb``."""

    paths: list[Path] = Field(min_length=1)
    """Folders of datatrove tokenized shards, as for ``data.paths``."""

    seq_len: int | None = Field(None, ge=1)
    """Context length to score the set at: every document is scored in non-overlapping windows of ``seq_len`` scored
    tokens. Defaults to the training ``data.seq_len``."""


class PretrainValConfig(BaseValConfig):
    data: list[HeldOutDataConfig] = []
    """Named sets of held-out shard folders. Every validation scores each document of a set on its own, from its first
    token, in non-overlapping windows of the set's ``seq_len`` scored tokens (a longer document continues in the next
    window, without the tokens before). The windows are packed into rows of ``micro_batch_tokens`` tokens."""

    num_samples: int | None = Field(None, ge=1)
    """Documents per set and validation: the first ``num_samples`` documents of each set (in folder order, then file
    name order), the same ones every time. If None, every document."""

    micro_batch_tokens: int | None = Field(None, ge=1)
    """Tokens per packed row of the sets (one forward pass on one GPU). Defaults to the training micro-batch
    (``data.micro_batch_size * data.seq_len``) or the longest of the sets' context lengths, whichever is larger; must
    be at least that longest length."""

    def row_len(self) -> int:
        return self.micro_batch_tokens

    @model_validator(mode="after")
    def validate_data(self):
        if not self.data:
            raise ValueError("[val] scores nothing: add a [[val.data]] set")
        names = [data.name for data in self.data]
        if len(set(names)) < len(names):
            raise ValueError(f"val.data names must be unique: {names}")
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


class PretrainConfig(SupervisedTrainerConfig):
    """Pretraining: next-token loss on tokenized shards, with the weights initialized from scratch unless resuming."""

    init_from_scratch = True

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

    # A field validator, so that the shared model validators (CP, Mega MoE) see the validation row size.
    @field_validator("val")
    @classmethod
    def auto_setup_val(cls, val: PretrainValConfig | None, info: ValidationInfo) -> PretrainValConfig | None:
        if val is None or "data" not in info.data:
            return val
        data = info.data["data"]
        for held_out in val.data:
            if held_out.seq_len is None:
                held_out.seq_len = data.seq_len
        longest = max(held_out.seq_len for held_out in val.data)
        if val.micro_batch_tokens is None:
            val.micro_batch_tokens = max(data.micro_batch_size * data.seq_len, longest)
        elif val.micro_batch_tokens < longest:
            raise ValueError(
                f"val.micro_batch_tokens must be at least the longest context length of the sets ({longest})"
            )
        return val
