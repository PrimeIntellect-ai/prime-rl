from typing import Any, Literal

from pydantic import Field, model_validator
from renderers import AutoRendererConfig, DefaultRendererConfig, RendererConfig
from renderers.base import MODEL_RENDERER_MAP

from prime_rl.configs.supervised import (
    BaseDataConfig,
    BaseSourceConfig,
    BaseValConfig,
    SupervisedTrainerConfig,
    check_sources,
)
from prime_rl.utils.config import BaseConfig


class LossMaskConfig(BaseConfig):
    system: bool = False
    """System messages contribute to the loss."""

    user: bool = False
    """User messages contribute to the loss."""

    assistant: bool = True
    """Assistant messages contribute to the loss."""

    tool: bool = False
    """Tool messages contribute to the loss."""


class SFTColumnsConfig(BaseConfig):
    """Dataset columns the trainer reads. Set a field to read a column under another name."""

    messages: str = "messages"
    """Column with the whole chat as a list of OpenAI chat messages."""

    prompt: str = "prompt"
    """Column with the prompt as a list of OpenAI chat messages, read with ``completion`` when a row has no messages."""

    completion: str = "completion"
    """Column with the completion as a list of OpenAI chat messages."""

    tools: str = "tools"
    """Column with the tool schemas in OpenAI function-calling format."""

    message_loss_mask: str = "message_loss_mask"
    """Optional column of booleans or integer 0/1 flags, one per resolved message (prompt + completion for split rows). Zero excludes a message from loss; one preserves its normal supervision. Missing or null masks leave supervision unchanged."""

    renderer: dict[str, str] = {"reasoning_effort": "reasoning_effort"}
    """Per-sample renderer arguments as ``renderer field = dataset column``, e.g. ``reasoning_effort = "effort"``. A row's non-null value overrides the ``[renderer]`` setting; rows and datasets without the column use it unchanged."""


class HFDatasetSourceConfig(BaseSourceConfig):
    """One ``(dataset, subset, split)`` of a Hugging Face dataset."""

    dataset: str
    """HF dataset name or local path."""

    revision: str | None = None
    """HF dataset revision to load. Ignored for a local path."""

    subset: str | None = None
    """Subset (config name) of the dataset."""

    split: str = "train"
    """Split of the dataset."""

    columns: SFTColumnsConfig | None = None
    """Columns of this source. Unset fields keep ``data.columns``."""

    renderer: dict[str, Any] = {}
    """Chat-template kwargs for this source's rows, applied over ``[renderer]`` and under a row's ``columns.renderer``
    values, e.g. ``{ reasoning_effort = "high" }``."""


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

    name: str | None = None
    """Optional legacy volume-path hint kept for Hosted SFT policy validation alongside ``[[data.source]]``."""

    @model_validator(mode="after")
    def validate_sources(self):
        check_sources(self.source)
        for source in self.source:
            overrides = source.columns.model_dump(exclude_unset=True) if source.columns is not None else {}
            source.columns = SFTColumnsConfig.model_validate({**self.columns.model_dump(), **overrides})
        return self


class SFTValConfig(BaseValConfig):
    data: SFTDataConfig

    @model_validator(mode="after")
    def validate_sources(self):
        if self.data.fake is not None:
            raise ValueError("val.data.fake: validation scores the rows of its sources, not fake data")
        return self


class SFTConfig(SupervisedTrainerConfig):
    mode: Literal["sft"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run sft`` accepts only
    "sft" (or no mode)."""

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer config. Defaults to auto-selecting from the tokenizer model name."""

    data: SFTDataConfig = SFTDataConfig()

    val: SFTValConfig | None = None
    """Validation configuration. If None, no validation runs."""

    @model_validator(mode="after")
    def validate_typed_renderer(self):
        """Require a typed renderer whenever SFT renders real samples."""
        if self.data.fake is not None and self.val is None:
            return self

        model_id = self.tokenizer.name or self.model.name
        if isinstance(self.renderer, AutoRendererConfig):
            if model_id in MODEL_RENDERER_MAP:
                return self
            reason = f"no typed renderer is registered for {model_id!r}"
        elif isinstance(self.renderer, DefaultRendererConfig):
            reason = "renderer.name='default' selects DefaultRenderer"
        else:
            return self

        raise ValueError(
            f"SFT requires a typed renderer with sampled-token and content attribution, but {reason}. "
            "Implement and register the renderer in the renderers package, or explicitly select an existing "
            "typed renderer only when its template is verified to match."
        )

    @model_validator(mode="after")
    def validate_vlm_constraints(self):
        if self.model.vlm is None:
            return self
        if self.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires data.micro_batch_size = 1.")
        if self.val is not None and self.val.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires val.data.micro_batch_size = 1.")
        return self
