"""System One data: the model's strategy renders each request into tokens, and prime-rl builds the sample around them."""

from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from itertools import accumulate
from typing import NotRequired, TypedDict

from datasets import Dataset, Value
from torch import Tensor
from transformers import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import SystemOneColumnsConfig, SystemOneDataConfig
from prime_rl.trainer.data import Sample
from prime_rl.trainer.sft.data import HFDataset
from prime_rl.trainer.sft.loss import LossInputs
from prime_rl.trainer.sft.system_one.record import SystemOneRequest, build_targets, parse_record
from prime_rl.utils.logger import get_logger


class RenderedRequest(TypedDict):
    """What a strategy renders for one request: its tokens, and where the model's head reads each question.
    prime-rl checks it and derives the positions, the loss mask and the sample's ``sample_data`` from it."""

    input_ids: list[int]
    seq_lens: list[int]
    """Attention segments, each > 0, summing to ``len(input_ids)``. Positions restart at 0 in each."""
    read_spans: list[list[tuple[int, int]]]
    """Per question, in request order: the ``[start, end)`` token spans the head reads, at least one. Spans are
    non-empty, each within one attention segment, disjoint and increasing across the whole record."""
    options: NotRequired[list[list[str]]]
    """Per question: its option ids in the order of the head's logits, when that is not the order of
    ``Question.options``. Targets are reordered to match."""


class SystemOneStrategy(ABC):
    """Base class of System One strategies: how a decision model sees a request (``render_request``) and reads each
    question's logits (``option_logits``). The model class names its strategy with the classmethod
    ``system_one_strategy`` and keeps the head's weights.

    Built once in the main process as ``Strategy(tokenizer, config)``, before dataloader workers fork, for training and
    validation; workers share what ``__init__`` builds. It never sees labels, and must be deterministic, so a resumed
    run reads the same samples."""

    def __init__(self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig):
        self.tokenizer = tokenizer
        """The run's tokenizer, so token ids match the model's embeddings. Any chat template is the strategy's own."""
        self.config = config
        """The model's config, with what its loader attached."""

    @abstractmethod
    def render_request(self, request: SystemOneRequest, max_tokens: int) -> RenderedRequest | None:
        """The request's tokens, attention segments and read spans, or None to skip the record. ``max_tokens`` is
        ``data.seq_len``: a longer rendering is skipped, never truncated."""

    @classmethod
    @abstractmethod
    def option_logits(cls, inputs: LossInputs) -> list[Tensor]:
        """The logits of one record's questions: one 1-D tensor per question, in record order, with one logit per
        option in the order of the question's ``options``. They are the logits the model trains on: a checkpoint's own
        temperature, which its server applies, is part of these logits; a calibration fitted after training is not.

        ``inputs`` is the record as one sample: ``hidden_states`` are the backbone's final (post-norm) hidden states of
        the record's tokens alone, ``(tokens, hidden)``, ``loss_mask`` is true on its read tokens, and ``sample_data``
        holds the record's ``id`` and its questions' ``id``, ``type``, ``options`` and ``reads`` (spans of the
        record's tokens), never a ``target``.

        Called inside the model's forward pass: read settings from ``inputs.model.config``, and call the model's
        modules rather than reading their weights, on inputs in their weights' dtype. The logits may have any float
        dtype."""


def system_one_strategy_cls(model_cls: type) -> type[SystemOneStrategy]:
    """The strategy that ``model_cls`` declares with ``system_one_strategy``."""
    if not callable(getattr(model_cls, "system_one_strategy", None)):
        raise ValueError(
            f"System One data trains a decision model, whose class declares system_one_strategy, but "
            f"{model_cls.__name__} does not. For a class of your own ([model.implementation]), define the classmethod "
            "system_one_strategy; for a d3 or Clef-Flash release, check the side file beside its config.json."
        )
    strategy_cls = model_cls.system_one_strategy()
    if not (isinstance(strategy_cls, type) and issubclass(strategy_cls, SystemOneStrategy)):
        raise TypeError(
            f"{model_cls.__name__}.system_one_strategy() must return a SystemOneStrategy subclass, got {strategy_cls!r}"
        )
    return strategy_cls


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _find_violation(rendered: RenderedRequest, request: SystemOneRequest) -> str | None:
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
            # The canonical options are distinct, so an order of their length that holds each of them is a permutation.
            same_length = isinstance(order, (list, tuple)) and len(order) == len(question.options)
            if not same_length or not all(option in order for option in question.options):
                return f"question {question.id!r}: options {order!r} must be a permutation of {question.options}"
    return None


def check_rendered_request(rendered: RenderedRequest, request: SystemOneRequest) -> None:
    """Raise ``ValueError`` unless the strategy's rendering keeps the ``RenderedRequest`` contract for this request."""
    violation = _find_violation(rendered, request)
    if violation is not None:
        raise ValueError(f"record {request.id!r}: the strategy broke the RenderedRequest contract: {violation}")


def build_sample(rendered: RenderedRequest, request: SystemOneRequest, targets: list[list[float] | None]) -> Sample:
    """A checked rendering and the record's targets (in canonical option order) -> the training sample.

    The loss mask marks the read tokens, and ``sample_data`` carries what a loss needs per question: its options,
    its read spans and its target, in the strategy's option order.
    """
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
        # Targets are per question, in sample_data: there are no token targets and no causal shift.
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
    """The record's sample, or None when the strategy declines the request or its rendering has more than ``seq_len``
    tokens. Without ``targets`` its questions carry none."""
    rendered = strategy.render_request(request, seq_len)
    if rendered is None:
        get_logger().debug(f"Skipping record {request.id!r}: the strategy returned None")
        return None
    # Checked before the length, so that a broken rendering of an over-long record raises instead of being skipped.
    check_rendered_request(rendered, request)
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
