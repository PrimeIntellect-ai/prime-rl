"""System One data: an encoder plugin turns each record into tokens, and prime-rl builds the sample around them."""

from abc import ABC, abstractmethod
from typing import NotRequired, TypedDict

from datasets import Dataset, Value
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import SystemOneColumnsConfig, SystemOneDataConfig
from prime_rl.trainer.data import Sample
from prime_rl.trainer.sft.data import HFDataset
from prime_rl.trainer.sft.system_one.record import SystemOneRecord, build_targets, parse_record
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import import_object


class EncodedRecord(TypedDict):
    """What an encoder returns for one record: its tokens, and where the model's head reads each question.
    prime-rl checks it and derives the positions, the loss mask and the batch's ``sample_data`` from it."""

    input_ids: list[int]
    seq_lens: list[int]
    """Attention segments, each > 0, summing to ``len(input_ids)``. Positions restart at 0 in each."""
    reads: list[list[tuple[int, int]]]
    """Per question, in record order: the ``[start, end)`` token spans the head reads, at least one. Spans are
    non-empty, disjoint and increasing across the whole record."""
    options: NotRequired[list[list[str]]]
    """Per question: its option ids in the order of the head's logits, when that is not the order of
    ``Question.options``. Targets are reordered to match."""


class SystemOneEncoder(ABC):
    """Base class of System One encoders: how a model sees a record.

    Built in the main process as ``Encoder(tokenizer, **kwargs)`` from ``[data.encoder]``, for the training dataset and
    again at every validation run; dataloader workers get it by fork. Keep ``__init__`` cheap: create heavy state
    lazily, on first call. It never sees labels, and must be deterministic, so a resumed run reads the same samples."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        """The run's tokenizer, so token ids match the model's embeddings. Any chat template is the encoder's own."""

    @abstractmethod
    def __call__(self, record: SystemOneRecord, max_tokens: int) -> EncodedRecord | None:
        """The record's tokens, attention segments and read spans, or None to skip the record. ``max_tokens`` is
        ``data.seq_len``: a longer encoding is skipped, never truncated."""


def _find_violation(encoded: EncodedRecord, record: SystemOneRecord) -> str | None:
    required, optional = EncodedRecord.__required_keys__, EncodedRecord.__optional_keys__
    if not isinstance(encoded, dict) or not required <= set(encoded) <= required | optional:
        got = sorted(encoded) if isinstance(encoded, dict) else type(encoded).__name__
        return f"expected a dict with input_ids, seq_lens, reads and optionally options, got {got}"

    input_ids, seq_lens, reads = encoded["input_ids"], encoded["seq_lens"], encoded["reads"]
    for key in ("input_ids", "seq_lens", "reads"):
        if not isinstance(encoded[key], list):
            return f"{key} must be a list, got {type(encoded[key]).__name__}"
    if not all(isinstance(token, int) for token in input_ids):
        return f"input_ids must hold ints, got {sorted({type(token).__name__ for token in input_ids})}"
    if not seq_lens or any(not isinstance(seq_len, int) or seq_len <= 0 for seq_len in seq_lens):
        return f"seq_lens must hold at least one segment, each a positive int, got {seq_lens}"
    if sum(seq_lens) != len(input_ids):
        return f"seq_lens sum to {sum(seq_lens)}, but there are {len(input_ids)} input_ids"

    if len(reads) != len(record.questions):
        return f"reads has {len(reads)} entries for {len(record.questions)} questions"
    previous_end = 0
    for question, spans in zip(record.questions, reads):
        if not isinstance(spans, (list, tuple)) or not spans:
            return f"question {question.id!r}: reads must be a non-empty list of spans, got {spans!r}"
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 2 or not all(isinstance(i, int) for i in span):
                return f"question {question.id!r}: read span {span!r} must be a (start, end) pair of ints"
            start, end = span
            if not previous_end <= start < end <= len(input_ids):
                return (
                    f"question {question.id!r}: read span {span!r} must satisfy {previous_end} <= start < end <= "
                    f"{len(input_ids)} (spans are non-empty, within the record, disjoint and increasing)"
                )
            previous_end = end

    if "options" in encoded:
        options = encoded["options"]
        if not isinstance(options, list) or len(options) != len(record.questions):
            return f"options must list the option ids of each of the {len(record.questions)} questions, got {options!r}"
        for question, order in zip(record.questions, options):
            # The canonical options are distinct, so an order of their length that holds each of them is a permutation.
            same_length = isinstance(order, (list, tuple)) and len(order) == len(question.options)
            if not same_length or not all(option in order for option in question.options):
                return f"question {question.id!r}: options {order!r} must be a permutation of {question.options}"
    return None


def check_encoded(encoded: EncodedRecord, record: SystemOneRecord) -> None:
    """Raise ``ValueError`` unless the encoder's output keeps the ``EncodedRecord`` contract for this record."""
    violation = _find_violation(encoded, record)
    if violation is not None:
        raise ValueError(f"record {record.id!r}: the encoder broke the EncodedRecord contract: {violation}")


def build_sample(encoded: EncodedRecord, record: SystemOneRecord, targets: list[list[float] | None]) -> Sample:
    """A checked encoding and the record's targets (in canonical option order) -> the training sample.

    The loss mask marks the read tokens, and ``sample_data`` carries what a loss needs per question: its options,
    its read spans and its target, in the encoder's option order.
    """
    num_tokens = len(encoded["input_ids"])
    loss_mask = [False] * num_tokens
    orders = encoded.get("options", [question.options for question in record.questions])
    questions = []
    for question, spans, order, target in zip(record.questions, encoded["reads"], orders, targets, strict=True):
        for start, end in spans:
            loss_mask[start:end] = [True] * (end - start)
        if target is not None:
            target = [target[question.options.index(option)] for option in order]
        questions.append(
            {
                "id": question.id,
                "type": question.type,
                "options": list(order),
                "reads": [list(span) for span in spans],
                "target": target,
            }
        )
    return {
        "input_ids": encoded["input_ids"],
        "position_ids": [position for seq_len in encoded["seq_lens"] for position in range(seq_len)],
        "seq_lens": encoded["seq_lens"],
        "loss_mask": loss_mask,
        # Targets are per question, in sample_data: there are no token targets and no causal shift.
        "target_ids": [0] * num_tokens,
        "mm_kwargs": None,
        "mm_token_type_ids": None,
        "sample_data": {"id": record.id, "num_tokens": num_tokens, "questions": questions},
    }


def encode_sample(
    encoder: SystemOneEncoder, record: SystemOneRecord, seq_len: int, targets: list[list[float] | None] | None = None
) -> Sample | None:
    """The record's sample, or None when the encoder declines the record or its encoding has more than ``seq_len``
    tokens. Without ``targets`` its questions carry none."""
    encoded = encoder(record, seq_len)
    if encoded is None:
        get_logger().debug(f"Skipping record {record.id!r}: the encoder returned None")
        return None
    # Checked before the length, so that a broken encoding of an over-long record raises instead of being skipped.
    check_encoded(encoded, record)
    if len(encoded["input_ids"]) > seq_len:
        get_logger().warning(
            f"Skipping record {record.id!r}: its encoding has {len(encoded['input_ids'])} tokens, more than "
            f"seq_len ({seq_len}). Records are never truncated."
        )
        return None
    return build_sample(encoded, record, targets or [None] * len(record.questions))


class SystemOneDataset(HFDataset):
    """A HF dataset of System One records, one sample per record with a labelled question."""

    def __init__(
        self,
        dataset: Dataset,
        encoder: SystemOneEncoder,
        columns: SystemOneColumnsConfig = SystemOneColumnsConfig(),
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        max_epochs: int | None = None,
    ):
        super().__init__(dataset, shuffle=shuffle, seed=seed, non_dp_size=non_dp_size, max_epochs=max_epochs)
        self.encoder = encoder
        self.columns = columns
        self.seq_len = seq_len

        names = dataset.column_names
        # The state and the questions are required. The other default names are optional; a name set in the config
        # must exist.
        for field in ("id", "state", "questions", "label", "soft_label"):
            column = getattr(columns, field)
            if (field in ("state", "questions") or column != field) and column not in names:
                raise ValueError(f"data.columns.{field} is {column!r}, but the dataset has only {names}")
        # A label column that is null in every row has the null type, and counts as absent.
        label_fields = [
            field
            for field in ("label", "soft_label")
            if getattr(columns, field) in names and dataset.features[getattr(columns, field)] != Value("null")
        ]
        if not label_fields:
            raise ValueError(
                f"System One data needs a label column, {columns.label!r} (data.columns.label) or "
                f"{columns.soft_label!r} (data.columns.soft_label), that is not null in every row, but the dataset "
                f"has only {names}"
            )
        # Stored as structs, rows would gain the keys of other rows as nulls: spurious options and state fields.
        for field in ("state", "questions", *label_fields):
            column = getattr(columns, field)
            feature = dataset.features[column]
            if not (isinstance(feature, Value) and feature.dtype in ("string", "large_string", "string_view")):
                raise ValueError(
                    f"data.columns.{field} is {column!r}, a column of type {feature}, but System One columns hold "
                    "JSON text: store each value with json.dumps (a text state as a JSON string)"
                )

    def _process(self, example: dict) -> dict | None:
        record, labels = parse_record(example, self.columns)
        targets = build_targets(record, labels)
        if all(target is None for target in targets):
            self.logger.warning(f"Skipping record {record.id!r}: none of its questions has a label")
            return None
        return encode_sample(self.encoder, record, self.seq_len, targets)


def setup_system_one_dataset(
    tokenizer: PreTrainedTokenizer,
    config: SystemOneDataConfig,
    raw_dataset: Dataset,
    *,
    non_dp_size: int = 1,
    max_epochs: int | None = None,
) -> SystemOneDataset:
    encoder_cls = import_object(config.encoder.import_path)
    if not (isinstance(encoder_cls, type) and issubclass(encoder_cls, SystemOneEncoder)):
        raise ValueError(
            f"data.encoder.import_path must name a SystemOneEncoder subclass, got {config.encoder.import_path!r} "
            f"({encoder_cls!r})"
        )
    return SystemOneDataset(
        raw_dataset,
        encoder_cls(tokenizer, **config.encoder.kwargs),
        columns=config.columns,
        shuffle=config.shuffle,
        seed=config.seed,
        seq_len=config.seq_len,
        non_dp_size=non_dp_size,
        max_epochs=max_epochs,
    )
