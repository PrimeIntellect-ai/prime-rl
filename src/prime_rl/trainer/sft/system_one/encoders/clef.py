"""Clef-Flash's joint schema prompt, as vLLM 0.33's ``ClefFlashStrategy`` builds it (``[data.encoder]``,
``import_path = "prime_rl.trainer.sft.system_one.encoders.clef.ClefFlashEncoder"``)."""

import json
from typing import Any

from prime_rl.trainer.sft.system_one.data import EncodedRecord, SystemOneEncoder
from prime_rl.trainer.sft.system_one.encoders import find_refusal
from prime_rl.trainer.sft.system_one.record import Question, SystemOneRecord
from prime_rl.utils.logger import get_logger

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each "
    "answer must be exactly one of that field's allowed options."
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
MAX_LENGTH = 16384
MAX_CHOICE_OPTIONS = 26
NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def head_options(question: Question) -> list[tuple[str, str | None]]:
    """Each option's id and description in the head's order: noul ``true``, ``false``; choices sorted by id; score
    levels ascending."""
    if question.type == "noul":
        criteria = question.criteria or {}
        return [(key, NOUL_CRITERIA[key] if criteria.get(key) is None else criteria[key]) for key in ("true", "false")]
    if question.type == "choice":
        return sorted(zip(question.options, question.descriptions), key=lambda option: option[0])
    return list(zip(question.options, question.criteria))


class ClefFlashEncoder(SystemOneEncoder):
    """The whole record is one prompt and one attention segment: the state, then every question's schema. A
    question's reads are its instruction, then one span per option in the head's order. A state that does not fit in
    ``min(16384, max_tokens)`` tokens is cut, as vLLM cuts it at ``min(16384, max_model_len)``."""

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def refusal(self, record: SystemOneRecord) -> str | None:
        """Why vLLM's ClefFlashStrategy would refuse the record's questions, or None."""
        if (reason := find_refusal(record)) is not None:
            return reason
        for question in record.questions:
            if len(question.options) > MAX_CHOICE_OPTIONS:
                return f"question {question.id!r}: {len(question.options)} options, more than {MAX_CHOICE_OPTIONS}"
        return None

    def __call__(self, record: SystemOneRecord, max_tokens: int) -> EncodedRecord | None:
        if (reason := self.refusal(record)) is not None:
            get_logger().warning(f"Skipping record {record.id!r}: {reason}")
            return None

        prefix, suffix = self._tokens(PREFIX), self._tokens(SUFFIX)
        schema_ids = self._tokens("\n\nSCHEMA FIELDS:\n")
        reads, options = [], []
        for n, question in enumerate(record.questions, start=1):
            schema_ids += self._tokens(f"\nFIELD {n}\nID: {question.id}\nTYPE: {question.type}\nINSTRUCTION: ")
            start = len(schema_ids)
            schema_ids += self._tokens(question.instructions or question.id)
            spans = [(start, len(schema_ids))]
            schema_ids += self._tokens("\nALLOWED OPTIONS:\n")
            order = head_options(question)
            for m, (option_id, description) in enumerate(order, start=1):
                schema_ids += self._tokens(f"OPTION {m}: ")
                start = len(schema_ids)
                semantics = {"option_id": option_id}
                if description is not None:
                    semantics["description"] = description
                schema_ids += self._tokens(render(semantics))
                spans.append((start, len(schema_ids)))
                schema_ids += self._tokens("\n")
            schema_ids += self._tokens("END FIELD\n")
            reads.append(spans)
            options.append([option_id for option_id, _ in order])

        max_length = min(MAX_LENGTH, max_tokens)
        fixed = len(prefix) + len(schema_ids) + len(suffix)
        if fixed > max_length:
            get_logger().warning(
                f"Skipping record {record.id!r}: its questions take {fixed} tokens, more than {max_length}"
            )
            return None
        state_ids = self._tokens(render(record.state))[: max_length - fixed]
        offset = len(prefix) + len(state_ids)
        return EncodedRecord(
            input_ids=prefix + state_ids + schema_ids + suffix,
            seq_lens=[fixed + len(state_ids)],
            reads=[[(start + offset, end + offset) for start, end in spans] for spans in reads],
            options=options,
        )
