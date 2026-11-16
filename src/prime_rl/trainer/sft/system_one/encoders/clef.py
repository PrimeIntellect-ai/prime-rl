# Ported from vLLM 0.33.0, vllm/entrypoints/openai/decisions/strategies.py (Apache-2.0, Copyright contributors to the
# vLLM project), JointSchemaStrategy and ClefFlashStrategy: the constants (lines 854-857, 861-864, 1142), self.prefix
# and self.suffix (__init__, 881-888), _render (899-905), _tokens (907-908), _options (998-1020) and encode
# (1022-1068), with vLLM's names. limits (890-891) is refusal, a skip; media (_render_media, PLACEHOLDERS) is not ported.
"""Clef-Flash's joint schema prompt, as vLLM 0.33's ``ClefFlashStrategy`` builds it (``[data.encoder]``,
``import_path = "prime_rl.trainer.sft.system_one.encoders.clef.ClefFlashEncoder"``)."""

import json
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.system_one.data import EncodedRecord, SystemOneEncoder
from prime_rl.trainer.sft.system_one.encoders import find_refusal
from prime_rl.trainer.sft.system_one.record import Question, SystemOneRecord
from prime_rl.utils.logger import get_logger


class ClefFlashEncoder(SystemOneEncoder):
    """The prompt half of vLLM's ClefFlashStrategy. The whole record is one prompt and one attention segment: the
    state, then every question's schema. A question's reads are its instruction, then one span per option in the
    head's order. A state that does not fit in ``min(16384, max_tokens)`` tokens is cut, as vLLM cuts it at
    ``min(16384, max_model_len)``."""

    SYSTEM_PROMPT = (
        "Read the complete state and schema. Decide every field jointly. Each "
        "answer must be exactly one of that field's allowed options."
    )
    MAX_LENGTH = 16384
    MAX_OPTIONS = 26
    NOUL_CRITERIA = {
        "true": "The proposition is true or the answer is yes.",
        "false": "The proposition is false or the answer is no.",
    }

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)
        self.prefix = self._tokens(f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
        self.suffix = self._tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
        )

    @staticmethod
    def _render(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _options(self, q: Question) -> list[tuple[str, Any]]:
        """Each option's id and description in the head's order: noul ``true``, ``false``; choices sorted by id; score
        levels ascending. Unlike vLLM's, without each option's index in the question's options."""
        if q.type == "noul":
            criteria = q.criteria or {}
            return [
                (key, self.NOUL_CRITERIA[key] if criteria.get(key) is None else criteria[key])
                for key in ("true", "false")
            ]
        if q.type == "choice":
            return sorted(zip(q.options, q.descriptions), key=lambda option: option[0])
        return list(zip(q.options, q.criteria))

    def encode(
        self, questions: tuple[Question, ...], state: str, max_length: int
    ) -> tuple[list[int], list[list[tuple[int, int]]], list[list[str]], int] | None:
        """The prompt, each question's reads, each question's option ids in the head's order and the number of state
        tokens cut, or None when the questions alone take more than ``max_length`` tokens, where vLLM raises. Unlike
        vLLM's, the spans count from the prompt's start, and each question's are its instruction's then its
        options'."""
        schema_ids = self._tokens("\n\nSCHEMA FIELDS:\n")
        reads, orders = [], []
        for n, q in enumerate(questions, start=1):
            schema_ids += self._tokens(f"\nFIELD {n}\nID: {q.id}\nTYPE: {q.type}\nINSTRUCTION: ")
            start = len(schema_ids)
            schema_ids += self._tokens(q.instructions or q.id)
            spans = [(start, len(schema_ids))]
            schema_ids += self._tokens("\nALLOWED OPTIONS:\n")
            options = self._options(q)
            for m, (option_id, description) in enumerate(options, start=1):
                schema_ids += self._tokens(f"OPTION {m}: ")
                start = len(schema_ids)
                semantics = {"option_id": option_id}
                if description is not None:
                    semantics["description"] = description
                schema_ids += self._tokens(self._render(semantics))
                spans.append((start, len(schema_ids)))
                schema_ids += self._tokens("\n")
            schema_ids += self._tokens("END FIELD\n")
            reads.append(spans)
            orders.append([option_id for option_id, _ in options])

        fixed = len(self.prefix) + len(schema_ids) + len(self.suffix)
        if fixed > max_length:
            return None
        state_ids = self._tokens(state)
        cut = max(len(state_ids) - (max_length - fixed), 0)
        state_ids = state_ids[: max_length - fixed]
        prompt = self.prefix + state_ids + schema_ids + self.suffix
        offset = len(self.prefix) + len(state_ids)
        return prompt, [[(s + offset, e + offset) for s, e in spans] for spans in reads], orders, cut

    def refusal(self, record: SystemOneRecord) -> str | None:
        """Why the record cannot be encoded as vLLM's ClefFlashStrategy reads it, or None."""
        if (reason := find_refusal(record)) is not None:
            return reason
        for question in record.questions:
            if len(question.options) > self.MAX_OPTIONS:
                return f"question {question.id!r}: {len(question.options)} options, more than {self.MAX_OPTIONS}"
        return None

    def __call__(self, record: SystemOneRecord, max_tokens: int) -> EncodedRecord | None:
        if (reason := self.refusal(record)) is not None:
            get_logger().warning(f"Skipping record {record.id!r}: {reason}")
            return None
        max_length = min(self.MAX_LENGTH, max_tokens)
        if (encoded := self.encode(record.questions, self._render(record.state), max_length)) is None:
            get_logger().warning(f"Skipping record {record.id!r}: its questions take more than {max_length} tokens")
            return None
        prompt, reads, options, cut = encoded
        if cut:
            get_logger().warning(
                f"Record {record.id!r}: cutting the last {cut} tokens of its state, so that its prompt fits in "
                f"{max_length} tokens"
            )
        return EncodedRecord(input_ids=prompt, seq_lens=[len(prompt)], reads=reads, options=options)
