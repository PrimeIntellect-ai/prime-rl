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


class HFDataConfig(BaseDataConfig):
    """Rows of a HF dataset, loaded with ``datasets.load_dataset`` and interleaved across subsets and splits."""

    name: str
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


class SFTDataConfig(HFDataConfig):
    type: Literal["sft"] = "sft"

    name: str = "PrimeIntellect/Reverse-Text-SFT"
    """HF dataset name or path."""

    columns: SFTColumnsConfig = SFTColumnsConfig()
    """Columns that carry per-sample renderer arguments."""

    # Configuring
    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types contribute to the loss."""


class SystemOneColumnsConfig(BaseConfig):
    """Dataset columns of a System One record. Set a field to read a column under another name."""

    id: str = "id"
    """Column with the record id, used in warnings and errors. If the column is absent, the row index is used."""

    state: str = "state"
    """Column with the state to decide on, as JSON text (a text state as a JSON string)."""

    questions: str = "questions"
    """Column with the System One questions as JSON text: question id -> ``{type, instructions, criteria}``."""

    label: str = "label"
    """Column with hard labels as JSON text: question id -> ``true``/``false`` for noul, an option id for choice, a
    level for score. Any question may go without one."""

    soft_label: str = "soft_label"
    """Column with soft labels as JSON text: question id -> option id -> probability. A question's soft label is
    used over its hard label."""


class SystemOneEncoderConfig(PluginConfig):
    """The model's System One encoder: ``import_path`` names a subclass of
    ``prime_rl.trainer.sft.system_one.data.SystemOneEncoder``, built as ``Encoder(tokenizer, **kwargs)`` and
    called per record as ``encoder(record, max_tokens)``. It returns an ``EncodedRecord``, or None to skip the
    record. It never sees labels."""


class SystemOneDataConfig(HFDataConfig):
    """System One records: a state and typed questions (noul, choice, score) per row, with hard and/or soft labels.
    Each question trains toward its label at the token spans the encoder marks for the model's head. The encoder owns
    any chat template: ``[renderer]`` is not used, and per-row columns such as ``reasoning_effort`` are not read."""

    type: Literal["system_one"] = "system_one"

    columns: SystemOneColumnsConfig = SystemOneColumnsConfig()
    """Dataset columns of a record."""

    encoder: SystemOneEncoderConfig
    """How a record becomes tokens, attention segments and read spans. It must match the model's head."""


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
    data: Annotated[SFTDataConfig | SystemOneDataConfig, Field(discriminator="type")]


DataConfig: TypeAlias = Annotated[FakeDataConfig | SFTDataConfig | SystemOneDataConfig, Field(discriminator="type")]


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

    @model_validator(mode="before")
    @classmethod
    def propagate_data_to_val(cls, data):
        """Fill a missing ``val.data.type`` from ``[data]``: "system_one" for System One data, else "sft". Nothing
        else is copied."""
        if not isinstance(data, dict):
            return data
        val = data.get("val")
        val_data = val.get("data") if isinstance(val, dict) else None
        if isinstance(val_data, dict) and "type" not in val_data:
            train_data = data.get("data")
            train_type = train_data.get("type") if isinstance(train_data, dict) else getattr(train_data, "type", None)
            val_data["type"] = "system_one" if train_type == "system_one" else "sft"
        return data

    @model_validator(mode="after")
    def validate_system_one_data(self):
        """System One data is used for training and validation alike, with one encoder and one ``seq_len``, a loss
        other than token cross-entropy, no ``[renderer]`` and no padding."""
        data_configs = {"data": self.data}
        if self.val is not None:
            data_configs["val.data"] = self.val.data
        if all(data.type != "system_one" for data in data_configs.values()):
            return self
        if any(data.type != "system_one" for data in data_configs.values()):
            types = ", ".join(f'{key}.type = "{data.type}"' for key, data in data_configs.items())
            raise ValueError(f'data.type and val.data.type must both be "system_one" or neither, but {types}.')

        if self.loss.type == "ce":
            raise ValueError(
                'data.type = "system_one" cannot train with loss.type = "ce": a record\'s targets are per question, '
                'not per token. Set [loss] type = "custom".'
            )
        # A resolved config writes the default [renderer] out, so compare with it instead of checking that it was set.
        if self.renderer != AutoRendererConfig():
            raise ValueError(
                '[renderer] is not used with data.type = "system_one": the encoder ([data.encoder]) owns any chat '
                "template. Remove [renderer]."
            )
        for key, data in data_configs.items():
            if data.pad_to_seq_len:
                raise ValueError(
                    f'{key}.pad_to_seq_len is not supported with {key}.type = "system_one": the padding would join a '
                    "record's last attention segment."
                )
        if self.val is not None:
            if self.val.data.encoder != self.data.encoder:
                raise ValueError(
                    "val.data.encoder must equal data.encoder: the model has one head, so validation records are "
                    "encoded like training records."
                )
            if self.val.data.seq_len != self.data.seq_len:
                raise ValueError(
                    f"val.data.seq_len ({self.val.data.seq_len}) must equal data.seq_len ({self.data.seq_len}): it is "
                    "the token budget of a record, and [val.data] does not inherit it."
                )
        return self

    @model_validator(mode="after")
    def validate_typed_renderer(self):
        """Require a typed renderer whenever SFT renders real samples."""
        if self.data.type == "fake" and self.val is None:
            return self
        # Nothing renders with [renderer]: validate_system_one_data, which runs first, requires System One val.data too.
        if self.data.type == "system_one":
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
