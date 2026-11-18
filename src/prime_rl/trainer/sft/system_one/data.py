"""System One data: the model's strategy renders each record into a training sample."""

from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, NotRequired, TypedDict

from datasets import Dataset, Value
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import SystemOneColumnsConfig, SystemOneDataConfig
from prime_rl.trainer.data import Sample
from prime_rl.trainer.sft.data import HFDataset
from prime_rl.trainer.sft.system_one.record import QuestionType, SystemOneRequest, build_targets, parse_record
from prime_rl.utils.logger import get_logger


class RenderedRequest(TypedDict):
    """A strategy's rendering of one request."""

    input_ids: list[int]
    seq_lens: list[int]
    """Attention segments, summing to ``len(input_ids)``."""
    read_spans: list[list[tuple[int, int]]]
    """Per question, the ``[start, end)`` token spans its logits are read from: disjoint, increasing, each within one
    segment."""
    options: NotRequired[list[list[str]]]
    """Per question, its option ids in logit order, when that differs from ``Question.options``."""


@dataclass(frozen=True)
class ReadQuestion:
    """A question as the readout sees it, without its target."""

    record_id: Any
    id: str
    type: QuestionType
    options: tuple[str, ...]
    """Option ids in logit order."""
    reads: tuple[tuple[int, int], ...]
    """``[start, end)`` row positions its logits are read from."""
    record: tuple[int, int]
    """``[start, end)`` row positions of its record."""


class SystemOneStrategy(ABC):
    """How a decision model sees a request and how its head reads each question's logits. Registered in
    ``SYSTEM_ONE_STRATEGIES`` by architecture name. Must be deterministic, so a resumed run reads the same samples."""

    def __init__(self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig):
        self.tokenizer = tokenizer
        self.config = config

    @abstractmethod
    def render_request(self, request: SystemOneRequest, max_tokens: int) -> RenderedRequest | None:
        """The request's rendering, or None to skip the record. A rendering longer than ``max_tokens`` is skipped."""

    @abstractmethod
    def option_logits(
        self,
        model: nn.Module,
        hidden_states: Float[Tensor, "seq hidden"],
        input_ids: Int[Tensor, "seq"],
        questions: list[ReadQuestion],
    ) -> Float[Tensor, "options"]:
        """Every question's option logits, concatenated in ``questions`` order, each in its ``options`` order.

        Called once per micro-batch inside the forward, on the packed row's final hidden states. Run the head once
        for the whole micro-batch, through its modules: their weights may be FSDP shards.
        Call the same modules on every micro-batch (otherwise ranks hang); the model class freezes parameters this
        never uses."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _find_violation(rendered: RenderedRequest, request: SystemOneRequest, vocab_size: int) -> str | None:
    required, optional = RenderedRequest.__required_keys__, RenderedRequest.__optional_keys__
    if not isinstance(rendered, dict) or not required <= set(rendered) <= required | optional:
        got = sorted(rendered) if isinstance(rendered, dict) else type(rendered).__name__
        return f"expected a dict with input_ids, seq_lens, read_spans and optionally options, got {got}"

    input_ids, seq_lens, read_spans = rendered["input_ids"], rendered["seq_lens"], rendered["read_spans"]
    for key in ("input_ids", "seq_lens", "read_spans"):
        if not isinstance(rendered[key], list):
            return f"{key} must be a list, got {type(rendered[key]).__name__}"
    if not all(_is_int(token) for token in input_ids):
        return f"input_ids must hold ints, got {sorted({type(token).__name__ for token in input_ids})}"
    if any(not 0 <= token < vocab_size for token in input_ids):
        return f"input_ids must be token ids in [0, {vocab_size}), got ids from {min(input_ids)} to {max(input_ids)}"
    if not seq_lens or any(not _is_int(seq_len) or seq_len <= 0 for seq_len in seq_lens):
        return f"seq_lens must hold at least one segment, each a positive int, got {seq_lens}"
    if sum(seq_lens) != len(input_ids):
        return f"seq_lens sum to {sum(seq_lens)}, but there are {len(input_ids)} input_ids"

    if len(read_spans) != len(request.questions):
        return f"read_spans has {len(read_spans)} entries for {len(request.questions)} questions"
    segment_ends = list(accumulate(seq_lens))
    previous_end = 0
    for question, spans in zip(request.questions, read_spans):
        if not isinstance(spans, (list, tuple)) or not spans:
            return f"question {question.id!r}: read_spans must be a non-empty list of spans, got {spans!r}"
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 2 or not all(_is_int(i) for i in span):
                return f"question {question.id!r}: read span {span!r} must be a (start, end) pair of ints"
            start, end = span
            if not previous_end <= start < end <= len(input_ids):
                return (
                    f"question {question.id!r}: read span {span!r} must satisfy {previous_end} <= start < end <= "
                    f"{len(input_ids)} (spans are non-empty, within the record, disjoint and increasing)"
                )
            if bisect_right(segment_ends, start) != bisect_left(segment_ends, end):
                return f"question {question.id!r}: read span {span!r} crosses a segment end, at one of {segment_ends}"
            previous_end = end

    if "options" in rendered:
        options = rendered["options"]
        if not isinstance(options, list) or len(options) != len(request.questions):
            return (
                f"options must list the option ids of each of the {len(request.questions)} questions, got {options!r}"
            )
        for question, order in zip(request.questions, options):
            # Options are distinct, so an order of the same length holding each of them is a permutation
            same_length = isinstance(order, (list, tuple)) and len(order) == len(question.options)
            if not same_length or not all(option in order for option in question.options):
                return f"question {question.id!r}: options {order!r} must be a permutation of {question.options}"
    return None


def check_rendered_request(rendered: RenderedRequest, request: SystemOneRequest, vocab_size: int) -> None:
    """Raise ``ValueError`` unless the strategy's rendering keeps the ``RenderedRequest`` contract for this request."""
    violation = _find_violation(rendered, request, vocab_size)
    if violation is not None:
        raise ValueError(f"record {request.id!r}: the strategy broke the RenderedRequest contract: {violation}")


def build_sample(rendered: RenderedRequest, request: SystemOneRequest, targets: list[list[float] | None]) -> Sample:
    """The training sample of a checked rendering. ``targets`` are in ``Question.options`` order."""
    num_tokens = len(rendered["input_ids"])
    loss_mask = [False] * num_tokens
    orders = rendered.get("options", [question.options for question in request.questions])
    questions = []
    for question, spans, order, target in zip(request.questions, rendered["read_spans"], orders, targets, strict=True):
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
        "input_ids": rendered["input_ids"],
        "position_ids": [position for seq_len in rendered["seq_lens"] for position in range(seq_len)],
        "seq_lens": rendered["seq_lens"],
        "loss_mask": loss_mask,
        # Targets are per question, in sample_data
        "target_ids": [0] * num_tokens,
        "mm_kwargs": None,
        "mm_token_type_ids": None,
        "sample_data": {"id": request.id, "questions": questions},
    }


def render_sample(
    strategy: SystemOneStrategy,
    request: SystemOneRequest,
    seq_len: int,
    targets: list[list[float] | None] | None = None,
) -> Sample | None:
    """The record's sample, or None when the strategy skips it or it is longer than ``seq_len``."""
    rendered = strategy.render_request(request, seq_len)
    if rendered is None:
        get_logger().debug(f"Skipping record {request.id!r}: the strategy returned None")
        return None
    # Before the length check, so a broken rendering raises instead of being skipped
    check_rendered_request(rendered, request, strategy.config.get_text_config().vocab_size)
    if len(rendered["input_ids"]) > seq_len:
        get_logger().warning(
            f"Skipping record {request.id!r}: its rendering has {len(rendered['input_ids'])} tokens, more than "
            f"seq_len ({seq_len}). Records are never truncated."
        )
        return None
    return build_sample(rendered, request, targets or [None] * len(request.questions))


class SystemOneDataset(HFDataset):
    """A HF dataset of System One records, one sample per record with a labelled question."""

    def __init__(
        self,
        dataset: Dataset,
        strategy: SystemOneStrategy,
        columns: SystemOneColumnsConfig = SystemOneColumnsConfig(),
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        max_epochs: int | None = None,
    ):
        super().__init__(dataset, shuffle=shuffle, seed=seed, non_dp_size=non_dp_size, max_epochs=max_epochs)
        self.strategy = strategy
        self.columns = columns
        self.seq_len = seq_len

        names = dataset.column_names
        # state and questions are required; the other columns only when renamed
        for field in ("id", "state", "questions", "label", "soft_label"):
            column = getattr(columns, field)
            if (field in ("state", "questions") or column != field) and column not in names:
                raise ValueError(f"data.columns.{field} is {column!r}, but the dataset has only {names}")
        # A label column that is null in every row has the null type
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
        # Not structs: a struct row gains the other rows' keys as nulls
        for field in ("state", "questions", *label_fields):
            column = getattr(columns, field)
            feature = dataset.features[column]
            if not (isinstance(feature, Value) and feature.dtype in ("string", "large_string", "string_view")):
                raise ValueError(
                    f"data.columns.{field} is {column!r}, a column of type {feature}, but System One columns hold "
                    "JSON text: store each value with json.dumps (a text state as a JSON string)"
                )

    def check_records(self) -> None:
        """Parse this data rank's share of the rows once, so that a malformed record stops the run at startup."""
        rows = self.dataset.shard(self.data_world_size, self.data_rank, contiguous=False)
        names = dict.fromkeys([*self.columns.model_dump().values(), "__subset", "__split", "__index"])
        rows = rows.select_columns([name for name in names if name in rows.column_names])
        for batch in rows.iter(batch_size=1024):
            for values in zip(*batch.values()):
                build_targets(*parse_record(dict(zip(batch, values)), self.columns))

    def _process(self, example: dict) -> dict | None:
        request, labels = parse_record(example, self.columns)
        targets = build_targets(request, labels)
        if all(target is None for target in targets):
            self.logger.warning(f"Skipping record {request.id!r}: none of its questions has a label")
            return None
        return render_sample(self.strategy, request, self.seq_len, targets)


def setup_system_one_dataset(
    strategy: SystemOneStrategy,
    config: SystemOneDataConfig,
    raw_dataset: Dataset,
    *,
    non_dp_size: int = 1,
    max_epochs: int | None = None,
) -> SystemOneDataset:
    return SystemOneDataset(
        raw_dataset,
        strategy,
        columns=config.columns,
        shuffle=config.shuffle,
        seed=config.seed,
        seq_len=config.seq_len,
        non_dp_size=non_dp_size,
        max_epochs=max_epochs,
    )
