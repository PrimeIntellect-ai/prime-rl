# Ported from vLLM v0.34.0 (Apache-2.0, Copyright contributors to the vLLM project), with vLLM's names:
# - vllm/entrypoints/openai/decisions/question_types.py: StructuredDecisionError, LABELS, Option, Question (lines 17-38),
#   QuestionType's name and label_set (41-44), build_question (101-131) and the parse_options of ChoiceQuestion,
#   NoulQuestion and ScoreQuestion (146-162, 178-199, 219-234).
# - vllm/entrypoints/generate/structured_decisions/serving.py: parse_questions (lines 27-53).
# - vllm/entrypoints/openai/decisions/strategies.py: DecisionLimits (lines 83-89); JointSchemaStrategy (841-1068): its
#   constants, __init__, limits, render_state, _render, _tokens, _options and encode; ClefFlashStrategy's MAX_LENGTH
#   (1138-1142). Text only: the media (modalities, PLACEHOLDERS, _render_media) is not ported.
# - vllm/model_executor/models/clef_joint_schema.py: parse_joint_schema's spans (lines 222-229) and
#   JointSchemaPooler.forward's lexical embeddings (lines 287-297), read from the LM head as
#   vllm/model_executor/models/clef_flash.py's _embed_lexical reads them (lines 43-44).
"""Clef-Flash's System One strategy: vLLM's ``ClefFlashStrategy`` prompt, read by the joint schema head."""

import json
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby
from typing import Any, ClassVar

import torch.nn.functional as F
from torch import Tensor, nn

from prime_rl.trainer.models.qwen3_5.clef_joint_schema import RecordSchema, schema_indices, segment_mean
from prime_rl.trainer.sft.system_one import record
from prime_rl.trainer.sft.system_one.data import ReadQuestion, RenderedRequest, SystemOneStrategy
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
    labels: tuple[str, ...]


class QuestionType(ABC):
    name: ClassVar[str]
    # The labels of a question's options, in option order.
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


def build_question(
    qid: str,
    type_name: str,
    instructions: Any,
    criteria: Any,
    max_options: int,
    labeled: bool = True,
) -> Question:
    if not qid:
        raise StructuredDecisionError("question ids must be non-empty")
    qtype = QUESTION_TYPES[type_name]
    options = qtype.parse_options(qid, criteria)
    if not options:
        raise StructuredDecisionError(f"question {qid!r}: needs at least one option")
    names = [o.name for o in options]
    if len(set(names)) != len(names):
        raise StructuredDecisionError(f"question {qid!r}: duplicate option names")
    limit = min(max_options, len(qtype.label_set)) if labeled else max_options
    if len(options) > limit:
        raise StructuredDecisionError(f"question {qid!r}: at most {limit} options for this model")
    if not isinstance(instructions, str):
        instructions = "" if instructions is None else str(instructions)
    return Question(
        id=qid,
        type=qtype,
        instructions=instructions,
        options=tuple(options),
        labels=qtype.label_set[: len(options)],
    )


@dataclass(frozen=True)
class DecisionLimits:
    max_questions: int
    max_options: int
    # Whether a question has at most as many options as its type has labels.
    labeled: bool = True


def parse_questions(questions: tuple[record.Question, ...], limits: DecisionLimits) -> list[Question]:
    if not questions:
        raise StructuredDecisionError("questions: needs at least one question")
    if len(questions) > limits.max_questions:
        raise StructuredDecisionError(f"questions: at most {limits.max_questions} for this model")
    return [
        build_question(q.id, q.type, q.instructions, q.criteria, limits.max_options, labeled=limits.labeled)
        for q in questions
    ]


class ClefFlashStrategy(SystemOneStrategy):
    """vLLM's ClefFlashStrategy without its softmax: the state, then every question with its options, in one prompt
    and one attention segment. A question's read spans are its instructions, then its options in the head's order."""

    SYSTEM_PROMPT = (
        "Read the complete state and schema. Decide every field jointly. Each "
        "answer must be exactly one of that field's allowed options."
    )
    MAX_LENGTH = 16384
    TYPE_IDS = {"noul": 0, "choice": 1, "score": 2}
    NOUL_CRITERIA = {
        "true": "The proposition is true or the answer is yes.",
        "false": "The proposition is false or the answer is no.",
    }

    def __init__(self, tokenizer, config):
        super().__init__(tokenizer, config)
        # vLLM's bound on a server that reads at least MAX_LENGTH tokens
        self.max_length = self.MAX_LENGTH
        self.prefix = self._tokens(f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
        self.suffix = self._tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
        )

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))

    def render_state(self, state: Any) -> str:
        return self._render(state)

    @staticmethod
    def _render(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _options(self, q: Question) -> list[tuple[int, str, Any]]:
        """Each option's index in ``q.options``, id and description, in the
        model's order: choices sorted by name, noul true then false, score
        levels by index."""
        if q.type.name == "noul":
            return [
                (
                    i,
                    key,
                    self.NOUL_CRITERIA[key] if o.description is None else o.description,
                )
                for i, (key, o) in enumerate(zip(("true", "false"), q.options))
            ]
        if q.type.name == "choice":
            return sorted(
                ((i, o.name, o.description) for i, o in enumerate(q.options)),
                key=lambda option: option[1],
            )
        if q.type.name == "score":
            return [(i, str(i), o.name) for i, o in enumerate(q.options)]
        raise StructuredDecisionError(f"question {q.id!r}: this model does not answer {q.type.name!r} questions")

    def encode(
        self, questions: list[Question], state: str, prefix: list[int] | None = None
    ) -> tuple[list[int], dict[str, list[Any]], list[list[int]]]:
        """The prompt, the head's schema with spans counted from the prompt's end, and each question's option indices
        in the head's order."""
        prefix = self.prefix if prefix is None else prefix
        schema_ids = self._tokens("\n\nSCHEMA FIELDS:\n")
        question_spans, option_spans, orders = [], [], []
        for n, q in enumerate(questions, start=1):
            schema_ids += self._tokens(f"\nFIELD {n}\nID: {q.id}\nTYPE: {q.type.name}\nINSTRUCTION: ")
            start = len(schema_ids)
            schema_ids += self._tokens(q.instructions or q.id)
            question_spans.append([start, len(schema_ids)])
            schema_ids += self._tokens("\nALLOWED OPTIONS:\n")
            options = self._options(q)
            for m, (_, option_id, description) in enumerate(options, start=1):
                schema_ids += self._tokens(f"OPTION {m}: ")
                start = len(schema_ids)
                semantics = {"option_id": option_id}
                if description is not None:
                    semantics["description"] = description
                schema_ids += self._tokens(self._render(semantics))
                option_spans.append([start, len(schema_ids)])
                schema_ids += self._tokens("\n")
            schema_ids += self._tokens("END FIELD\n")
            orders.append([i for i, _, _ in options])

        fixed = len(prefix) + len(schema_ids) + len(self.suffix)
        if fixed > self.max_length:
            raise StructuredDecisionError(
                f"the questions take {fixed} tokens, more than the {self.max_length} this model reads"
            )
        state_ids = self._tokens(state)[: self.max_length - fixed]
        prompt = prefix + state_ids + schema_ids + self.suffix
        # The schema ends len(self.suffix) tokens before the prompt does.
        end = len(schema_ids) + len(self.suffix)
        schema: dict[str, list[Any]] = {
            "question_types": [self.TYPE_IDS[q.type.name] for q in questions],
            "question_spans": [[s - end, e - end] for s, e in question_spans],
            "option_spans": [[s - end, e - end] for s, e in option_spans],
            "option_counts": [len(order) for order in orders],
        }
        return prompt, schema, orders

    def render_request(self, request: record.SystemOneRequest, max_tokens: int) -> RenderedRequest | None:
        """vLLM's prompt, with the state cut to fit ``MAX_LENGTH`` tokens as vLLM cuts it. A request vLLM refuses is
        skipped. Not bounded by ``max_tokens``: a longer rendering is skipped, and only a kept one's cut is logged."""
        try:
            questions = parse_questions(request.questions, self.limits())
            state = self.render_state(request.state)
            prompt, schema, _ = self.encode(questions, state)
        except StructuredDecisionError as error:
            get_logger().warning(f"Skipping record {request.id!r}: vLLM refuses it: {error}")
            return None
        if len(prompt) == self.max_length <= max_tokens:
            cut = len(self.encode(questions, "")[0]) + len(self._tokens(state)) - self.max_length
            if cut > 0:
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
        options = [[option_id for _, option_id, _ in self._options(q)] for q in questions]
        return RenderedRequest(input_ids=prompt, seq_lens=[prompt_len], read_spans=read_spans, options=options)

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
