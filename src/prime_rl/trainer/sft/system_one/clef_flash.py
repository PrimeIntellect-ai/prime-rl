# Ported from vLLM v0.34.0 (Apache-2.0, Copyright contributors to the vLLM project), with vLLM's names:
# - vllm/entrypoints/openai/decisions/question_types.py: StructuredDecisionError, LABELS, Option, Question (lines 17-38),
#   QuestionType's name and label_set (41-44), build_question (101-131) and the parse_options of ChoiceQuestion,
#   NoulQuestion and ScoreQuestion (146-162, 178-199, 219-234).
# - vllm/entrypoints/generate/structured_decisions/serving.py: parse_questions (lines 27-53).
# - vllm/entrypoints/openai/decisions/strategies.py: DecisionLimits (lines 83-89); JointSchemaStrategy (841-1068): its
#   constants, __init__, limits, _render, _tokens, _options and encode; ClefFlashStrategy (1138-1190): its MAX_LENGTH,
#   its image placeholder and _render_media for images, with the model's image processor in place of vLLM's renderer.
#   Videos are not ported.
# - vllm/model_executor/models/clef_joint_schema.py: parse_joint_schema's spans (lines 222-229) and
#   JointSchemaPooler.forward's lexical embeddings (lines 287-297), read from the LM head as
#   vllm/model_executor/models/clef_flash.py's _embed_lexical reads them (lines 43-44).
# Left out, as only vLLM's other strategies and its responses use them: Question.labels, DecisionLimits.labeled and
# render_state.
"""Clef-Flash's System One strategy: vLLM's ``ClefFlashStrategy`` prompt, read by the joint schema head."""

import json
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prime_rl.trainer.models.qwen3_5.clef_joint_schema import RecordSchema, schema_indices, segment_mean
from prime_rl.trainer.sft.system_one import record
from prime_rl.trainer.sft.system_one.data import ReadQuestion, RenderedRequest, SystemOneStrategy
from prime_rl.trainer.sft.system_one.images import load_image
from prime_rl.utils.logger import get_logger


class StructuredDecisionError(ValueError):
    """A request vLLM refuses; training skips its record."""


# A choice labels its options A to Z in order, so it has at most 26.
LABELS = tuple(string.ascii_uppercase)


@dataclass(frozen=True)
class Option:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    type: "QuestionType"
    instructions: str
    options: tuple[Option, ...]


class QuestionType(ABC):
    name: ClassVar[str]
    # vLLM's labels of a question's options, in option order. A question has at most as many options.
    label_set: ClassVar[tuple[str, ...]] = LABELS

    @abstractmethod
    def parse_options(self, qid: str, criteria: Any) -> list[Option]: ...


class ChoiceQuestion(QuestionType):
    """Pick one option. ``criteria`` maps each option name to a description,
    or to null."""

    name = "choice"

    def parse_options(self, qid: str, criteria: Any) -> list[Option]:
        if not isinstance(criteria, dict) or not criteria:
            raise StructuredDecisionError(
                f"question {qid!r}: choice criteria must map option names to a description or null"
            )
        return [Option(str(name), None if desc is None else str(desc)) for name, desc in criteria.items()]


class NoulQuestion(QuestionType):
    """Yes or no. ``criteria`` may describe what true and false mean."""

    name = "noul"
    label_set = ("yes", "no")

    def parse_options(self, qid: str, criteria: Any) -> list[Option]:
        if criteria is not None and (not isinstance(criteria, dict) or not criteria.keys() <= {"true", "false"}):
            raise StructuredDecisionError(f"question {qid!r}: noul criteria must be an object with true and false")
        criteria = criteria or {}
        true = criteria.get("true")
        false = criteria.get("false")
        return [
            Option("yes", None if true is None else str(true)),
            Option("no", None if false is None else str(false)),
        ]


class ScoreQuestion(QuestionType):
    """Rate the state on an ordered scale. ``criteria`` is the ordered list of
    level names, from the first to the last level of the scale."""

    name = "score"
    # 0-indexed like the answer, so a level's label is its score.
    label_set = tuple(string.digits)

    def parse_options(self, qid: str, criteria: Any) -> list[Option]:
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise StructuredDecisionError(f"question {qid!r}: score criteria must be an ordered list of levels")
        return [Option(str(level)) for level in criteria]


QUESTION_TYPES: dict[str, QuestionType] = {cls.name: cls() for cls in (ChoiceQuestion, NoulQuestion, ScoreQuestion)}


def build_question(qid: str, type_name: str, instructions: Any, criteria: Any, max_options: int) -> Question:
    if not qid:
        raise StructuredDecisionError("question ids must be non-empty")
    qtype = QUESTION_TYPES[type_name]
    options = qtype.parse_options(qid, criteria)
    if not options:
        raise StructuredDecisionError(f"question {qid!r}: needs at least one option")
    names = [o.name for o in options]
    if len(set(names)) != len(names):
        raise StructuredDecisionError(f"question {qid!r}: duplicate option names")
    limit = min(max_options, len(qtype.label_set))
    if len(options) > limit:
        raise StructuredDecisionError(f"question {qid!r}: at most {limit} options for this model")
    if not isinstance(instructions, str):
        instructions = "" if instructions is None else str(instructions)
    return Question(id=qid, type=qtype, instructions=instructions, options=tuple(options))


@dataclass(frozen=True)
class DecisionLimits:
    max_questions: int
    max_options: int


def parse_questions(questions: tuple[record.Question, ...], limits: DecisionLimits) -> list[Question]:
    if not questions:
        raise StructuredDecisionError("questions: needs at least one question")
    if len(questions) > limits.max_questions:
        raise StructuredDecisionError(f"questions: at most {limits.max_questions} for this model")
    return [build_question(q.id, q.type, q.instructions, q.criteria, limits.max_options) for q in questions]


class ClefFlashStrategy(SystemOneStrategy):
    """vLLM's ClefFlashStrategy without its softmax: the state, then every question with its options, in one prompt
    and one attention segment. A question's read spans are its instructions, then its options in the head's order."""

    SYSTEM_PROMPT = (
        "Read the complete state and schema. Decide every field jointly. Each "
        "answer must be exactly one of that field's allowed options."
    )
    MAX_LENGTH = 16384
    IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
    takes_images = True
    TYPE_IDS = {"noul": 0, "choice": 1, "score": 2}
    NOUL_CRITERIA = {
        "true": "The proposition is true or the answer is yes.",
        "false": "The proposition is false or the answer is no.",
    }

    def __init__(self, tokenizer, config, processor=None):
        super().__init__(tokenizer, config, processor)
        if processor is not None and self._tokens(self.IMAGE_PLACEHOLDER).count(config.image_token_id) != 1:
            raise ValueError(f"the tokenizer's <|image_pad|> is not the model's image token {config.image_token_id}")
        # vLLM's bound on a server that reads at least MAX_LENGTH tokens
        self.max_length = self.MAX_LENGTH
        self.prefix = self._tokens(f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
        self.suffix = self._tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
        )

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))

    @staticmethod
    def _render(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _options(self, q: Question) -> list[tuple[str, Any]]:
        """Each option's id and description, in the model's order: choices sorted by name, noul true then false,
        score levels by index."""
        if q.type.name == "noul":
            return [
                (key, self.NOUL_CRITERIA[key] if o.description is None else o.description)
                for key, o in zip(("true", "false"), q.options)
            ]
        if q.type.name == "choice":
            return sorted(((o.name, o.description) for o in q.options), key=lambda option: option[0])
        return [(str(i), o.name) for i, o in enumerate(q.options)]

    def _render_images(self, images: tuple[str, ...]) -> tuple[list[int], dict[str, Tensor]]:
        """The system prompt and the images, with the images' placeholders expanded, and the images' inputs to the
        model's forward."""
        image_processor = self.processor.image_processor
        try:
            processed = image_processor(images=[load_image(url) for url in images], return_tensors="pt")
        except ValueError as error:
            raise StructuredDecisionError(f"media: {error}") from error
        mm_kwargs = {key: torch.as_tensor(processed[key]).contiguous() for key in ("pixel_values", "image_grid_thw")}
        lengths = iter((mm_kwargs["image_grid_thw"].prod(-1) // image_processor.merge_size**2).tolist())
        prefix = list(self.prefix)
        for token in self._tokens(self.IMAGE_PLACEHOLDER * len(images) + "\n"):
            prefix += [token] * (next(lengths) if token == self.config.image_token_id else 1)
        return prefix, mm_kwargs

    def encode(
        self, questions: list[Question], state: str, prefix: list[int]
    ) -> tuple[list[int], dict[str, list[Any]], list[list[str]], int]:
        """The prompt, the head's schema with spans counted from the prompt's end, each question's option ids in the
        head's order, and the number of tokens cut from the state's end. ``prefix`` is what precedes the state: the
        system prompt, and the images if there are any."""
        schema_ids = self._tokens("\n\nSCHEMA FIELDS:\n")
        question_spans, option_spans, option_ids = [], [], []
        for n, q in enumerate(questions, start=1):
            schema_ids += self._tokens(f"\nFIELD {n}\nID: {q.id}\nTYPE: {q.type.name}\nINSTRUCTION: ")
            start = len(schema_ids)
            schema_ids += self._tokens(q.instructions or q.id)
            question_spans.append([start, len(schema_ids)])
            schema_ids += self._tokens("\nALLOWED OPTIONS:\n")
            options = self._options(q)
            for m, (option_id, description) in enumerate(options, start=1):
                schema_ids += self._tokens(f"OPTION {m}: ")
                start = len(schema_ids)
                semantics = {"option_id": option_id}
                if description is not None:
                    semantics["description"] = description
                schema_ids += self._tokens(self._render(semantics))
                option_spans.append([start, len(schema_ids)])
                schema_ids += self._tokens("\n")
            schema_ids += self._tokens("END FIELD\n")
            option_ids.append([option_id for option_id, _ in options])

        fixed = len(prefix) + len(schema_ids) + len(self.suffix)
        if fixed > self.max_length:
            raise StructuredDecisionError(
                f"the questions take {fixed} tokens, more than the {self.max_length} this model reads"
            )
        state_ids = self._tokens(state)
        cut = max(0, fixed + len(state_ids) - self.max_length)
        prompt = prefix + state_ids[: self.max_length - fixed] + schema_ids + self.suffix
        # The schema ends len(self.suffix) tokens before the prompt does.
        end = len(schema_ids) + len(self.suffix)
        schema: dict[str, list[Any]] = {
            "question_spans": [[s - end, e - end] for s, e in question_spans],
            "option_spans": [[s - end, e - end] for s, e in option_spans],
            "option_counts": [len(ids) for ids in option_ids],
        }
        return prompt, schema, option_ids, cut

    def render_request(self, request: record.SystemOneRequest, max_tokens: int) -> RenderedRequest | None:
        """vLLM's prompt, with the state cut to fit ``MAX_LENGTH`` tokens as vLLM cuts it. A request vLLM refuses is
        skipped. Not bounded by ``max_tokens``: a longer rendering is skipped, and only a kept one's cut is logged."""
        try:
            questions = parse_questions(request.questions, self.limits())
            prefix, mm_kwargs = self._render_images(request.images) if request.images else (self.prefix, None)
            prompt, schema, options, cut = self.encode(questions, self._render(request.state), prefix)
        except StructuredDecisionError as error:
            get_logger().warning(f"Skipping record {request.id!r}: vLLM refuses it: {error}")
            return None
        # The model puts image features at every image token of a row, vLLM only at the placeholders'
        image_token = self.config.image_token_id
        if self.processor is not None and prompt.count(image_token) != prefix.count(image_token):
            get_logger().warning(f"Skipping record {request.id!r}: its text holds the image token")
            return None
        if cut and self.max_length <= max_tokens:
            get_logger().warning(
                f"Record {request.id!r}: the last {cut} tokens of its state are cut to fit {self.max_length}"
            )
        prompt_len = len(prompt)
        question_spans = [(start + prompt_len, end + prompt_len) for start, end in schema["question_spans"]]
        option_spans = iter([(start + prompt_len, end + prompt_len) for start, end in schema["option_spans"]])
        read_spans = [
            [question_span, *(next(option_spans) for _ in range(count))]
            for question_span, count in zip(question_spans, schema["option_counts"])
        ]
        rendered = RenderedRequest(input_ids=prompt, seq_lens=[prompt_len], read_spans=read_spans, options=options)
        if mm_kwargs is not None:
            rendered["mm_kwargs"] = mm_kwargs
            rendered["mm_token_type_ids"] = [int(token == image_token) for token in prompt]
        return rendered

    def option_logits(
        self, model: nn.Module, hidden_states: Tensor, input_ids: Tensor, questions: list[ReadQuestion]
    ) -> Tensor:
        """The joint schema head, run once on all of the micro-batch's records, with the lexical prior read from the
        LM head's rows."""
        records = []
        for (start, end), group in groupby(questions, key=lambda question: question.record):
            group = list(group)
            spans = [q.reads[0] for q in group] + [span for q in group for span in q.reads[1:]]
            types = [self.TYPE_IDS[q.type] for q in group]
            records.append(RecordSchema(start, end, types, spans, [len(q.reads) - 1 for q in group]))
        assert len({schema.start for schema in records}) == len(records), "a record's questions must be adjacent"
        schema, lexical_positions, lexical_span_ids = schema_indices(records, len(hidden_states), hidden_states.device)
        # The full weight: an untied LM head shares the final norm's FSDP unit, which stays gathered through the forward
        lexical = segment_mean(
            F.embedding(input_ids[lexical_positions], model.lm_head.weight),
            lexical_span_ids,
            schema.span_lengths[len(questions) :],
        )
        return model.pooler.head(hidden_states, lexical, schema).float()
