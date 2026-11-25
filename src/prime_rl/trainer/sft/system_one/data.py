"""System One data: the model's strategy renders each record into a training sample."""

from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import ClassVar, NotRequired, TypedDict

import numpy as np
from datasets import Dataset, Value
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers import PretrainedConfig, ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.data import Sample
from prime_rl.trainer.sft.data import HFRowSource
from prime_rl.trainer.sft.system_one.record import QuestionType, SystemOneRequest, parse_record
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
    mm_kwargs: NotRequired[dict[str, Tensor]]
    """The images' inputs to the model's forward, in prompt order, for a request with images."""
    mm_token_type_ids: NotRequired[list[int]]
    """With ``mm_kwargs``: per token, 1 for an image's token, else 0."""


@dataclass(frozen=True)
class ReadQuestion:
    """A question as the readout sees it, without its target."""

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

    modalities: ClassVar[frozenset[str]] = frozenset()
    """The media ``render_request`` renders with ``[model.vlm]``, in vLLM's names. Only "image" is supported."""

    def __init__(
        self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig, processor: ProcessorMixin | None = None
    ):
        """``processor`` is the model's, given with ``[model.vlm]`` to a strategy that reads images."""
        self.tokenizer = tokenizer
        self.config = config
        self.processor = processor

    @abstractmethod
    def render_request(self, request: SystemOneRequest, max_tokens: int) -> RenderedRequest | None:
        """The request's rendering, or None to skip the record. The trainer never truncates a rendering: one longer
        than ``max_tokens`` is skipped, so a strategy may shorten its own content to fit."""

    @abstractmethod
    def option_logits(
        self,
        model: nn.Module,
        hidden_states: Float[Tensor, "seq hidden"],
        input_ids: Int[Tensor, "seq"],
        questions: list[ReadQuestion],
    ) -> Float[Tensor, "options"]:
        """Every question's option logits, concatenated in ``questions`` order, each in its ``options`` order. Runs
        once per micro-batch inside the forward: call the same head modules, once each, on every micro-batch, rather
        than reading their weights; the model class freezes parameters this never uses."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _find_violation(
    rendered: RenderedRequest, request: SystemOneRequest, vocab_size: int, image_token_id: int | None
) -> str | None:
    required, optional = RenderedRequest.__required_keys__, RenderedRequest.__optional_keys__
    if not isinstance(rendered, dict) or not required <= set(rendered) <= required | optional:
        got = sorted(rendered) if isinstance(rendered, dict) else type(rendered).__name__
        return (
            "expected a dict with input_ids, seq_lens, read_spans and optionally options, mm_kwargs and "
            f"mm_token_type_ids, got {got}"
        )

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

    if ("mm_kwargs" in rendered) != ("mm_token_type_ids" in rendered):
        return "mm_kwargs and mm_token_type_ids go together"
    if ("mm_kwargs" in rendered) != bool(request.images):
        return f"mm_kwargs must be given for a request with images and only then: it has {len(request.images)}"
    if "mm_kwargs" in rendered:
        if any("video" in key for key in rendered["mm_kwargs"]):
            return f"video inputs are not supported, got {sorted(rendered['mm_kwargs'])}"
        # Without pixel_values the model would train without the images
        missing = {"pixel_values", "image_grid_thw"} - set(rendered["mm_kwargs"])
        if missing:
            return f"mm_kwargs lacks {sorted(missing)}"
        # More than the record's are fine: a prompt per question shows them in each
        num_images = len(rendered["mm_kwargs"]["image_grid_thw"])
        if num_images < len(request.images):
            return f"{num_images} images rendered of the record's {len(request.images)}"
        token_types = rendered["mm_token_type_ids"]
        if len(token_types) != len(input_ids):
            return f"{len(token_types)} mm_token_type_ids for {len(input_ids)} input_ids"
        if not set(token_types) <= {0, 1} or 1 not in token_types:
            return f"mm_token_type_ids must be 1 at the images' tokens and 0 elsewhere, got {sorted(set(token_types))}"
        # The model puts features at its image token only, and would leave the others out
        others = {token for token, token_type in zip(input_ids, token_types) if token_type == 1} - {image_token_id}
        if others:
            return f"an image's tokens must be the model's image token {image_token_id}, got {sorted(others)}"
    return None


def check_rendered_request(
    rendered: RenderedRequest, request: SystemOneRequest, vocab_size: int, image_token_id: int | None = None
) -> None:
    """Raise ``ValueError`` unless the strategy's rendering keeps the ``RenderedRequest`` contract for this request.
    ``image_token_id`` is the model's, for a request with images."""
    violation = _find_violation(rendered, request, vocab_size, image_token_id)
    if violation is not None:
        raise ValueError(f"record {request.id!r}: the strategy broke the RenderedRequest contract: {violation}")


def build_sample(rendered: RenderedRequest, request: SystemOneRequest, targets: list[list[float] | None]) -> Sample:
    """The training sample of a checked rendering. ``targets`` are in ``Question.options`` order."""
    num_tokens = len(rendered["input_ids"])
    loss_mask = np.zeros(num_tokens, dtype=bool)
    orders = rendered.get("options", [question.options for question in request.questions])
    questions = []
    for question, spans, order, target in zip(request.questions, rendered["read_spans"], orders, targets, strict=True):
        for start, end in spans:
            loss_mask[start:end] = True
        if target is not None:
            target = [target[question.options.index(option)] for option in order]
        questions.append(
            {
                "type": question.type,
                "options": list(order),
                "reads": [list(span) for span in spans],
                "target": target,
            }
        )
    return {
        "input_ids": np.array(rendered["input_ids"], dtype=np.int64),
        "position_ids": np.concatenate([np.arange(seq_len) for seq_len in rendered["seq_lens"]]),
        "seq_lens": rendered["seq_lens"],
        "loss_mask": loss_mask,
        # Targets are per question, in sample_data
        "target_ids": np.zeros(num_tokens, dtype=np.int64),
        "mm_kwargs": rendered.get("mm_kwargs"),
        "mm_token_type_ids": np.array(rendered["mm_token_type_ids"], dtype=np.int64)
        if "mm_token_type_ids" in rendered
        else None,
        "sample_data": {"id": request.id, "questions": questions},
    }


def has_stray_image_token(rendered: RenderedRequest, image_token_id: int) -> bool:
    """Whether the rendering has the image token where it has no image, as when a record's text holds it. The model
    would put an image's features there."""
    input_ids = np.asarray(rendered["input_ids"])
    is_image = np.asarray(rendered.get("mm_token_type_ids", np.zeros_like(input_ids))) == 1
    return bool(((input_ids == image_token_id) & ~is_image).any())


def check_images(request: SystemOneRequest, takes_images: bool) -> None:
    """Raise if the record has images that the run would leave out."""
    if request.images and not takes_images:
        raise ValueError(
            f"record {request.id!r} has images, which need [model.vlm] and a model whose strategy reads images; to "
            "train without them, drop or rename the column"
        )


def render_sample(
    strategy: SystemOneStrategy,
    request: SystemOneRequest,
    seq_len: int,
    targets: list[list[float] | None] | None = None,
) -> Sample | None:
    """The record's sample, or None when the strategy skips it or it is longer than ``seq_len``."""
    check_images(request, strategy.processor is not None)
    try:
        rendered = strategy.render_request(request, seq_len)
    except ValueError as error:
        raise ValueError(f"record {request.id!r}: {error}") from error
    if rendered is None:
        get_logger().debug(f"Skipping record {request.id!r}: the strategy returned None")
        return None
    # Before the length check, so a broken rendering raises instead of being skipped
    image_token_id = None if strategy.processor is None else strategy.config.image_token_id
    check_rendered_request(rendered, request, strategy.config.get_text_config().vocab_size, image_token_id)
    if image_token_id is not None and has_stray_image_token(rendered, image_token_id):
        get_logger().warning(
            f"Skipping record {request.id!r}: its rendering has the model's image token outside its images, as when "
            "its text holds it"
        )
        return None
    if len(rendered["input_ids"]) > seq_len:
        get_logger().warning(
            f"Skipping record {request.id!r}: its rendering has {len(rendered['input_ids'])} tokens, more than "
            f"seq_len ({seq_len}): the trainer never truncates a rendering."
        )
        return None
    return build_sample(rendered, request, targets or [None] * len(request.questions))


def check_columns(dataset: Dataset, columns: SystemOneColumnsConfig) -> None:
    """Raise unless the dataset has the record's columns, as string columns of JSON text."""
    names = dataset.column_names
    # state and questions are required; the other columns only when renamed
    for field in ("id", "state", "questions", "label", "soft_label", "images"):
        column = getattr(columns, field)
        if (field in ("state", "questions") or column != field) and column not in names:
            raise ValueError(f"data.columns.{field} is {column!r}, but the dataset has only {names}")
    # A column that is null in every row has the null type
    given = [
        field
        for field in ("label", "soft_label", "images")
        if getattr(columns, field) in names and dataset.features[getattr(columns, field)] != Value("null")
    ]
    label_fields = [field for field in given if field != "images"]
    if not label_fields:
        raise ValueError(
            f"System One data needs a label column, {columns.label!r} (data.columns.label) or "
            f"{columns.soft_label!r} (data.columns.soft_label), that is not null in every row, but the dataset "
            f"has only {names}"
        )
    # Not structs: a struct row gains the other rows' keys as nulls
    for field in ("state", "questions", *given):
        column = getattr(columns, field)
        feature = dataset.features[column]
        if not (isinstance(feature, Value) and feature.dtype in ("string", "large_string", "string_view")):
            hint = "a text state as a JSON string"
            if field == "images":
                hint = "a list of base64 data URLs; drop or rename the column if it is not for the model"
            raise ValueError(
                f"data.columns.{field} is {column!r}, a column of type {feature}, but System One columns hold "
                f"JSON text: store each value with json.dumps ({hint})"
            )


def check_records(
    dataset: Dataset,
    columns: SystemOneColumnsConfig,
    rank: int = 0,
    world_size: int = 1,
    source: str | None = None,
    split: str | None = None,
    takes_images: bool = False,
) -> None:
    """Parse every ``world_size``-th row from ``rank`` of the split ``split`` of the source ``source`` once, so that a
    malformed record, or one with images when the run does not take them, stops the run at startup."""
    check_columns(dataset, columns)
    if takes_images and rank == 0 and columns.images not in dataset.column_names:
        get_logger().warning(
            f"Source {source!r}, split {split!r} has no {columns.images!r} column (data.columns.images), so its "
            f"records have no images: its columns are {dataset.column_names}"
        )
    rows = dataset.shard(world_size, rank, contiguous=False)
    names = dict.fromkeys([*columns.model_dump().values(), "__index"])
    rows = rows.select_columns([name for name in names if name in rows.column_names])
    # Few rows at a time, as a row may hold images
    for batch in rows.iter(batch_size=32):
        for values in zip(*batch.values()):
            request, _ = parse_record(dict(zip(batch, values)), columns, source, split)
            check_images(request, takes_images)


class SystemOneSource(HFRowSource):
    """Rows of a HF dataset of System One records, one sample per record with a labelled question."""

    def __init__(
        self,
        dataset: Dataset,
        strategy: SystemOneStrategy,
        name: str = "train",
        split: str | None = None,
        columns: SystemOneColumnsConfig = SystemOneColumnsConfig(),
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
    ):
        super().__init__(dataset, name=name, shuffle=shuffle, seed=seed)
        self.strategy = strategy
        self.split = split
        self.columns = columns
        self.seq_len = seq_len
        check_columns(dataset, columns)

    def _process(self, example: dict) -> Sample | None:
        request, targets = parse_record(example, self.columns, self.name, self.split)
        if all(target is None for target in targets):
            self.logger.warning(f"Skipping record {request.id!r}: none of its questions has a label")
            return None
        return render_sample(self.strategy, request, self.seq_len, targets)
