from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

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


class EvalSetConfig(BaseConfig):
    """An eval set in ``val.data_dir`` with its own context lengths."""

    name: str
    """Names the metrics, ``val/<name>/<metric>@<length>``, and the set's directory in ``val.data_dir``."""

    seq_lens: list[int] | None = None
    """Context lengths to score this set at instead of ``val.seq_lens``."""


class PretrainValConfig(BaseValConfig):
    data: DatatroveDataConfig
    """Held-out shard folders. Every validation scores each of their documents on its own, from its first token, in
    non-overlapping windows of ``seq_len`` scored tokens (a longer document continues in the next window, without the
    tokens before). Windows are packed into rows of ``seq_len * micro_batch_size`` tokens; ``seed`` and
    ``document_masking`` do not apply."""

    num_samples: int | None = Field(None, ge=1)
    """Documents per validation: the first ``num_samples`` documents of the held-out shards (in folder order, then file
    name order), the same ones every time. If None, every document."""

    sets: list[str | EvalSetConfig] = []
    """Log-likelihood eval sets to score, by name (a directory in ``data_dir``) or with their own ``seq_lens``. Each
    directory holds ``test.parquet`` and ``meta.json`` with the set's ``kind``: ``ppl`` (column ``text``), ``ctxcont``
    (``context``, ``continuation``, optionally ``choices`` and ``gold`` or ``choice_contexts``) or ``pairs`` (``good``,
    ``bad``). Logged as ``val/<set>/<metric>@<length>``."""

    data_dir: Path | None = None
    """Directory of the eval sets, one directory per set."""

    seq_lens: list[int] | None = None
    """Context lengths to score every set at: the window length for ``ppl`` sets and the context limit for
    ``ctxcont`` sets. Defaults to ``data.seq_len`` of training."""

    micro_batch_tokens: int | None = Field(None, ge=1)
    """Tokens per packed row of the sets (one forward pass on one GPU). Defaults to the training micro-batch
    (``data.micro_batch_size * data.seq_len``) or the longest of the sets' context lengths, whichever is larger; must
    be at least that longest length."""

    save_arrays: bool = False
    """Write the sets' per-document and per-item arrays (npz) and summaries (json) to ``<run_dir>/val/step_N/``."""

    @model_validator(mode="after")
    def validate_data_dir(self):
        if self.sets and self.data_dir is None:
            raise ValueError("val.sets needs val.data_dir")
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

    @model_validator(mode="after")
    def auto_setup_val_sets(self):
        if self.val is None or not self.val.sets:
            return self
        if self.model.cp > 1:
            raise ValueError("val.sets does not support context parallelism")
        if self.val.seq_lens is None:
            self.val.seq_lens = [self.data.seq_len]
        own = [s.seq_lens for s in self.val.sets if not isinstance(s, str) and s.seq_lens]
        longest = max(max(lens) for lens in [self.val.seq_lens, *own])
        if self.val.micro_batch_tokens is None:
            self.val.micro_batch_tokens = max(self.data.micro_batch_size * self.data.seq_len, longest)
        elif self.val.micro_batch_tokens < longest:
            raise ValueError(
                f"val.micro_batch_tokens must be at least the longest context length of the sets ({longest})"
            )
        return self
