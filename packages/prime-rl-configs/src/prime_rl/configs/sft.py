from typing import Literal

from pydantic import Field, model_validator

from prime_rl.configs.supervised import (
    BaseDataConfig,
    BaseSourceConfig,
    HFDatasetConfig,
    LossMaskConfig,
    SFTColumnsConfig,
    SupervisedTrainerConfig,
    check_sources,
)
from prime_rl.utils.config import BaseConfig


class HFDatasetSourceConfig(HFDatasetConfig, BaseSourceConfig):
    """One ``(dataset, subset, split)`` of a Hugging Face dataset, a source of the training data."""


class FakeDataConfig(BaseConfig):
    """Synthetic token sequences of up to ``data.seq_len`` tokens, for throughput and plumbing tests."""

    length: Literal["fixed", "variable"] = "fixed"
    """Use fixed-length samples or variable-length samples."""

    input_ids: Literal["increasing", "random"] = "increasing"
    """Token id generator: ``increasing`` for deterministic sequences, ``random`` for random ids."""


class SFTDataConfig(BaseDataConfig):
    """Samples drawn from one or more sources into packed rows of ``seq_len`` tokens."""

    source: list[HFDatasetSourceConfig] = Field(
        default_factory=lambda: [HFDatasetSourceConfig(name="reverse-text", dataset="PrimeIntellect/Reverse-Text-SFT")],
        min_length=1,
    )
    """The sources to draw samples from (``[[data.source]]``)."""

    fake: FakeDataConfig | None = None
    """Train on synthetic token sequences instead of the sources. Needs no renderer."""

    shuffle: bool = True
    """Shuffle every source at the start of each of its passes."""

    seed: int = 0
    """Seed for the order of every source, re-shuffled per pass by adding the pass count, and for fake tokens."""

    columns: SFTColumnsConfig = SFTColumnsConfig()
    """Columns of every source. A source's ``columns`` override single fields."""

    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types contribute to the loss."""

    @model_validator(mode="after")
    def validate_sources(self):
        check_sources(self.source)
        for source in self.source:
            source.resolve_columns(self.columns)
        return self


class SFTConfig(SupervisedTrainerConfig):
    mode: Literal["sft"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run sft`` accepts only
    "sft" (or no mode)."""

    data: SFTDataConfig = SFTDataConfig()

    def renders_conversations(self) -> bool:
        return self.data.fake is None or super().renders_conversations()

    @model_validator(mode="after")
    def validate_vlm_constraints(self):
        if self.model.vlm is None:
            return self
        if self.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires data.micro_batch_size = 1.")
        return self
