from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator
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


class HFSourceConfig(BaseSourceConfig):
    """One ``(dataset, subset, split)`` of a Hugging Face dataset."""

    dataset: str
    """HF dataset name or local path."""

    revision: str | None = None
    """HF dataset revision to load. Ignored for a local path."""

    subset: str | None = None
    """Subset (config name) of the dataset."""

    split: str = "train"
    """Split of the dataset."""


class HFDatasetSourceConfig(HFSourceConfig):
    """One ``(dataset, subset, split)`` of chat rows."""

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


class HFDataConfig(BaseDataConfig):
    """Samples drawn from one or more HF sources into packed rows of ``seq_len`` tokens. The base of the data types,
    never instantiated itself: a subclass declares the ``source`` list and the ``columns`` of every source, which
    ``validate_sources`` reads."""

    shuffle: bool = True
    """Shuffle every source at the start of each of its passes."""

    seed: int = 0
    """Seed for the order of every source, re-shuffled per pass by adding the pass count."""

    @model_validator(mode="after")
    def validate_sources(self):
        check_sources(self.source)
        for source in self.source:
            overrides = source.columns.model_dump(exclude_unset=True) if source.columns is not None else {}
            source.columns = type(self.columns).model_validate({**self.columns.model_dump(), **overrides})
        return self


class SFTDataConfig(HFDataConfig):
    """Chat samples: messages, or a prompt and a completion, per row."""

    type: Literal["sft"] = "sft"

    source: list[HFDatasetSourceConfig] = Field(
        default_factory=lambda: [HFDatasetSourceConfig(name="reverse-text", dataset="PrimeIntellect/Reverse-Text-SFT")],
        min_length=1,
    )
    """The sources to draw samples from (``[[data.source]]``)."""

    fake: FakeDataConfig | None = None
    """Train on synthetic token sequences instead of the sources, generated from ``seed``. Needs no renderer."""

    @field_validator("fake", mode="before")
    @classmethod
    def expand_bare_flag(cls, value: Any) -> Any:
        """``--data.fake`` and ``--no-data.fake`` arrive as booleans: the CLI expands a bare flag into a table only
        outside a union, and ``[data]`` is one."""
        if isinstance(value, bool):
            return {} if value else None
        return value

    columns: SFTColumnsConfig = SFTColumnsConfig()
    """Columns of every source. A source's ``columns`` override single fields."""

    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types contribute to the loss."""


class SystemOneColumnsConfig(BaseConfig):
    """Dataset columns of a System One record. Set a field to read a column under another name."""

    id: str = "id"
    """Column with the record id for warnings and errors. Absent or null, the id is the source's name, its split and
    the row's index."""

    state: str = "state"
    """Column with the state to decide on, as JSON text (a text state as a JSON string)."""

    questions: str = "questions"
    """Column with the System One questions as JSON text: question id -> ``{type, instructions, criteria}``."""

    label: str = "label"
    """Column with hard labels as JSON text: question id -> option. Any question may go without one."""

    soft_label: str = "soft_label"
    """Column with soft labels as JSON text: question id -> option id -> probability. Used over the hard label."""


class SystemOneSourceConfig(HFSourceConfig):
    """One ``(dataset, subset, split)`` of System One records."""

    columns: SystemOneColumnsConfig | None = None
    """Columns of this source. Unset fields keep ``data.columns``."""


class SystemOneDataConfig(HFDataConfig):
    """System One records: a state and typed questions per row, with hard and/or soft labels."""

    type: Literal["system_one"] = "system_one"

    source: list[SystemOneSourceConfig] = Field(min_length=1)
    """The sources to draw records from (``[[data.source]]``)."""

    columns: SystemOneColumnsConfig = SystemOneColumnsConfig()
    """Dataset columns of every source's records. A source's ``columns`` override single fields."""


class SFTCrossEntropyLossConfig(BaseConfig):
    """Token cross-entropy: the stock SFT loss."""

    type: Literal["ce"] = "ce"


class SFTDecisionLossConfig(BaseConfig):
    """Soft cross-entropy plus ``brier_weight`` x Brier score per labelled question of System One data."""

    type: Literal["decision"] = "decision"

    brier_weight: float = Field(0.0, ge=0, allow_inf_nan=False)
    """Weight of the Brier score (the squared error between a question's distribution and its target)."""


SFTLossConfig: TypeAlias = Annotated[SFTCrossEntropyLossConfig | SFTDecisionLossConfig, Field(discriminator="type")]


DataConfig: TypeAlias = Annotated[SFTDataConfig | SystemOneDataConfig, Field(discriminator="type")]


class SFTValConfig(BaseValConfig):
    data: DataConfig

    @model_validator(mode="after")
    def validate_sources(self):
        if self.data.type == "sft" and self.data.fake is not None:
            raise ValueError("val.data.fake: validation scores the rows of its sources, not fake data")
        return self


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
    """The training loss. Defaults to ``decision`` with System One data, else ``ce``."""

    @model_validator(mode="before")
    @classmethod
    def fill_defaults_from_data(cls, data):
        """Fill a missing ``val.data.type`` and ``loss.type`` from ``data.type``, and with System One data a missing
        ``val.data.seq_len`` from ``data.seq_len``."""
        if not isinstance(data, dict):
            return data
        # A dict, or a config whose fields iterate as pairs
        train_data = dict(data.get("data") or {})
        system_one = train_data.get("type") == "system_one"
        val = data.get("val")
        val_data = val.get("data") if isinstance(val, dict) else None
        if isinstance(val_data, dict):
            val_data.setdefault("type", "system_one" if system_one else "sft")
            if system_one and "seq_len" in train_data:
                val_data.setdefault("seq_len", train_data["seq_len"])
        if system_one:
            loss = data.setdefault("loss", {})
            if isinstance(loss, dict):
                loss.setdefault("type", "decision")
        elif isinstance(data.get("loss"), dict):
            data["loss"].setdefault("type", "ce")
        return data

    @model_validator(mode="after")
    def validate_system_one_data(self):
        """System One data and the decision loss go together, for training and validation alike."""
        data_configs = {"data": self.data}
        if self.val is not None:
            data_configs["val.data"] = self.val.data
        if all(data.type != "system_one" for data in data_configs.values()):
            if self.loss.type == "decision":
                raise ValueError(f'loss.type = "decision" needs data.type = "system_one", got "{self.data.type}"')
            return self
        if any(data.type != "system_one" for data in data_configs.values()):
            types = ", ".join(f'{key}.type = "{data.type}"' for key, data in data_configs.items())
            raise ValueError(f'data.type and val.data.type must both be "system_one" or neither, but {types}.')

        if self.loss.type == "ce":
            raise ValueError(
                'data.type = "system_one" cannot train with loss.type = "ce": a record\'s targets are per question, '
                'not per token. Remove [loss], or set [loss] type = "decision".'
            )
        if self.model.lora is not None:
            raise ValueError(
                'data.type = "system_one" is not supported with [model.lora]: how a decision model\'s own readout '
                "trains next to adapters is undefined, and its LoRA checkpoint can be neither exported nor served. "
                "For now this holds for all System One data. Remove [model.lora]."
            )
        if self.eval is not None:
            raise ValueError(
                'data.type = "system_one" is not supported with [eval]: online evals score generated text, and vLLM '
                "serves a decision model with its own readout as a pooling model, which generates none. For now this "
                "holds for all System One data. Remove [eval], and score checkpoints with [val] or outside the "
                "trainer instead."
            )
        if self.model.cp > 1:
            raise ValueError(
                'data.type = "system_one" does not support context parallelism (model.cp > 1) yet: it is a planned '
                "follow-up. Set model.cp = 1."
            )
        if self.model.vlm is not None:
            raise ValueError(
                'data.type = "system_one" does not support [model.vlm] yet: it is a planned follow-up. Remove '
                "[model.vlm]."
            )
        # Compare with the default: a resolved config writes it out
        if self.renderer != AutoRendererConfig():
            raise ValueError(
                '[renderer] is not used with data.type = "system_one": the model\'s strategy owns any chat template. '
                "Remove [renderer]."
            )
        if self.val is not None and self.val.data.seq_len != self.data.seq_len:
            raise ValueError(
                f"val.data.seq_len ({self.val.data.seq_len}) must equal data.seq_len ({self.data.seq_len}): it is "
                "the token budget of a record. Leave it unset to take data.seq_len."
            )
        return self

    @model_validator(mode="after")
    def validate_typed_renderer(self):
        """Require a typed renderer whenever SFT renders real samples."""
        # The strategy renders System One data
        if self.data.type == "system_one":
            return self
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
