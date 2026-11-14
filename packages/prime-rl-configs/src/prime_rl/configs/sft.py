from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator
from renderers import AutoRendererConfig, DefaultRendererConfig, RendererConfig
from renderers.base import MODEL_RENDERER_MAP

from prime_rl.configs.shared import PluginConfig
from prime_rl.configs.supervised import BaseDataConfig, BaseValConfig, SupervisedTrainerConfig
from prime_rl.utils.config import BaseConfig


class FakeDataConfig(BaseDataConfig):
    type: Literal["fake"] = "fake"

    length: Literal["fixed", "variable"] = "fixed"
    """Use fixed-length samples or variable-length samples."""

    input_ids: Literal["increasing", "random"] = "increasing"
    """Token id generator: ``increasing`` for deterministic sequences, ``random`` for random ids."""

    seed: int = 0
    """Seed for the per-rank packing/token generator, combined with the data rank."""


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


class SFTDataConfig(BaseDataConfig):
    type: Literal["sft"] = "sft"

    name: str = "PrimeIntellect/Reverse-Text-SFT"
    """HF dataset name or path."""

    revision: str | None = None
    """HF dataset revision to load. Ignored for a local path."""

    subsets: list[str] | None = None
    """Subsets to load from the HF dataset."""

    splits: list[str] | None = None
    """Splits to load from the HF dataset."""

    probabilities: list[float] | None = None
    """Sampling probabilities for each subset/split."""

    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted"
    """Stopping strategy when interleaving multiple subsets/splits."""

    shuffle: bool = True
    """Shuffle the dataset at the start of each epoch."""

    seed: int = 0
    """Random seed for shuffling. Re-shuffled per epoch by adding the epoch count to the seed."""

    columns: SFTColumnsConfig = SFTColumnsConfig()
    """Columns that carry per-sample renderer arguments."""

    # Configuring
    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types contribute to the loss."""

    @model_validator(mode="after")
    def validate_subsets_and_splits(self):
        if self.subsets is not None or self.splits is not None:
            if self.subsets is not None and self.splits is not None:
                if len(self.subsets) != len(self.splits):
                    raise ValueError(
                        "Number of subsets must be equal to number of splits. Please specify which split to load for each subset."
                    )
            if self.subsets is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.subsets):
                    raise ValueError(
                        "Number of probabilities must be equal to number of subsets. Please specify a probability for each subset."
                    )
            if self.splits is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.splits):
                    raise ValueError(
                        "Number of probabilities must be equal to number of splits. Please specify a probability for each split."
                    )
        return self


class SFTCrossEntropyLossConfig(BaseConfig):
    """Token cross-entropy: the stock SFT loss."""

    type: Literal["ce"] = "ce"


class SFTCustomLossConfig(PluginConfig):
    """A custom SFT loss: a function computed from the backbone's final hidden states, in place of token
    cross-entropy, called as ``my_loss(inputs, **kwargs)`` with a ``prime_rl.trainer.sft.loss.LossInputs`` and
    returning a ``LossOutputs``."""

    type: Literal["custom"] = "custom"


SFTLossConfig: TypeAlias = Annotated[SFTCrossEntropyLossConfig | SFTCustomLossConfig, Field(discriminator="type")]


class SFTValConfig(BaseValConfig):
    data: SFTDataConfig


DataConfig: TypeAlias = Annotated[FakeDataConfig | SFTDataConfig, Field(discriminator="type")]


class SFTConfig(SupervisedTrainerConfig):
    mode: Literal["sft"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run sft`` accepts only
    "sft" (or no mode)."""

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer config. Defaults to auto-selecting from the tokenizer model name."""

    data: DataConfig = SFTDataConfig()

    val: SFTValConfig | None = None
    """Validation configuration. If None, no validation runs."""

    loss: SFTLossConfig = SFTCrossEntropyLossConfig()
    """The training loss: token cross-entropy (``type = "ce"``, the default) or a custom loss (``type = "custom"``)."""

    @model_validator(mode="after")
    def validate_typed_renderer(self):
        """Require a typed renderer whenever SFT renders real samples."""
        if self.data.type == "fake" and self.val is None:
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
    def validate_loss(self):
        if self.loss.type != "custom":
            return self
        if self.model.cp > 1 or self.model.vlm is not None or self.model.offload == "full":
            raise ValueError(
                "a custom loss is not supported with context parallelism, VLM training or model.offload='full'"
            )
        return self

    @model_validator(mode="after")
    def validate_vlm_constraints(self):
        if self.model.vlm is None:
            return self
        if self.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires data.micro_batch_size = 1.")
        if self.val is not None and self.val.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires val.data.micro_batch_size = 1.")
        return self
