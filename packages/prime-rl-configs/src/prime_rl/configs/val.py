"""Validation (``[val]``): named sets scored on the trainer's weights by the SFT, pretraining and RL trainers."""

from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

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


class BaseValSetConfig(BaseConfig):
    name: str = Field(pattern=r"^[^/]+$")
    """Names the set's metrics, ``val/<name>/<metric>@<seq_len>``, e.g. ``val/fineweb-edu/bpb@2k``."""

    seq_len: int | None = Field(None, ge=1)
    """Context length: the most input tokens a scored sequence has. Defaults to the training sequence length
    (``data.seq_len`` in SFT and pretraining, ``model.seq_len`` in RL)."""


class DocumentsValSetConfig(BaseValSetConfig):
    """Pre-tokenized documents. Every document is scored on its own, from its first token, in non-overlapping windows
    of ``seq_len`` scored tokens: a longer document continues in the next window, which does not see the tokens
    before it."""

    type: Literal["documents"] = "documents"

    path: Path
    """Folder of datatrove tokenized shards, as for pretraining's ``data.paths``. The tokens are scored as stored."""


class ChatValSetConfig(BaseValSetConfig):
    """Conversations, rendered as SFT renders its training data. The tokens SFT trains on (``loss_mask``) are scored,
    and a conversation is cut after ``seq_len`` input tokens."""

    type: Literal["chat"] = "chat"

    path: str
    """HF dataset name or local directory, as ``datasets.load_dataset`` takes it."""

    split: str
    """Split of the dataset. Its rows are scored in order."""

    subset: str | None = None
    """Subset (HF config name) of the dataset."""

    revision: str | None = None
    """HF dataset revision to load. Ignored for a local directory."""

    columns: SFTColumnsConfig = SFTColumnsConfig()
    """Dataset columns, as for SFT's ``data.columns``."""

    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types are scored."""


ValSetConfig: TypeAlias = Annotated[DocumentsValSetConfig | ChatValSetConfig, Field(discriminator="type")]


class ValConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Run validation every N training steps."""

    eval_on_start: bool = False
    """Run validation before the first training step."""

    sets: list[ValSetConfig] = Field(min_length=1)
    """The validation sets (``[[val.sets]]``). Each logs ``val/<name>/loss@<seq_len>``, the mean loss per scored
    token, and with a byte-level BPE tokenizer ``val/<name>/bpb@<seq_len>``, bits per UTF-8 byte of the scored
    tokens."""

    num_samples: int | None = Field(None, ge=1)
    """Items per set and validation: the first ``num_samples`` documents or rows of each set, the same ones every
    time. If None, every item."""

    micro_batch_tokens: int | None = Field(None, ge=1)
    """Tokens per packed row of the sets (one forward pass). Defaults to the training micro-batch
    (``data.seq_len * data.micro_batch_size`` in SFT and pretraining, ``model.seq_len`` in RL) or the longest set
    ``seq_len``, whichever is larger, rounded up to a multiple of ``model.cp``; must be at least that longest
    ``seq_len`` and divisible by ``model.cp``."""

    save_arrays: bool = False
    """Write the per-item arrays (npz) and summary (json) of every set to ``<run_dir>/val/step_N/``."""

    @model_validator(mode="after")
    def validate_unique_names(self):
        names = [val_set.name for val_set in self.sets]
        if len(set(names)) < len(names):
            raise ValueError(f"val.sets names must be unique: {names}")
        return self

    def resolve(self, seq_len: int, micro_batch_tokens: int, cp: int) -> None:
        """Fill the defaults from the trainer's sequence length and micro-batch tokens, and check the row size."""
        for val_set in self.sets:
            if val_set.seq_len is None:
                val_set.seq_len = seq_len
        longest = max(val_set.seq_len for val_set in self.sets)
        if self.micro_batch_tokens is None:
            self.micro_batch_tokens = -(-max(micro_batch_tokens, longest) // cp) * cp
        elif self.micro_batch_tokens < longest:
            raise ValueError(f"val.micro_batch_tokens must be at least the longest set seq_len ({longest})")
        elif self.micro_batch_tokens % cp != 0:
            raise ValueError("val.micro_batch_tokens must be divisible by CP degree")
