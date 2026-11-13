"""Open-Jev's encoder (``[data.encoder]``, ``import_path = "decision_models.open_jev.encoder.OpenJevEncoder"``).

The prompt format (``render``, ``compile_question`` and ``candidate_prompts``) is Open-Jev's, from ``jev/api.py`` of
https://github.com/Zefan-Cai/Open-Jev at commit bd41188 (MIT, Copyright (c) 2026 Open-Jev contributors; the notice is
in ../README.md). The released head and the Yes/No initialisation depend on the exact text.
"""

import json
from typing import Any

from renderers import Qwen35RendererConfig, create_renderer

from prime_rl.trainer.sft.system_one.data import EncodedRecord, SystemOneEncoder
from prime_rl.trainer.sft.system_one.record import Question, SystemOneRecord


def render(value: Any) -> str:
    if not isinstance(value, (str, dict, list)):
        raise ValueError(f"Open-Jev renders text, objects and arrays, got {value!r}")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def compile_question(question: Question) -> tuple[str, list[str]]:
    """The question's text and its answers as the model reads them. A noul question's descriptions join the text; a
    choice answer is ``name: description``; a score answer is its level's description."""
    text, descriptions = render(question.instructions), dict(zip(question.options, question.descriptions))
    if question.type == "noul":
        for option, word in (("true", "Yes"), ("false", "No")):
            if option in (question.criteria or {}):
                text += f"\n{word} means: {render(descriptions[option])}"
        return text, ["no", "yes"]
    if question.type == "choice":
        return text, [name if d is None else f"{name}: {render(d)}" for name, d in descriptions.items()]
    return text, [render(level) for level in question.descriptions]


def candidate_prompts(state: Any, question: Question) -> list[str]:
    """One yes/no prompt for a noul question, and one per answer for a choice or score question."""
    text, answers = compile_question(question)
    prefix = f"Context:\n{render(state)}\n\nQuestion: {text}\n"
    if question.type == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
    return [
        prefix + f"Proposed answer: {answer}\nIs this proposed answer correct? Answer Yes or No." for answer in answers
    ]


class OpenJevEncoder(SystemOneEncoder):
    """Every candidate prompt is a user turn of its own attention segment, read at its last token (the end of the
    generation prompt, with thinking off)."""

    renderer = None

    def __call__(self, record: SystemOneRecord, max_tokens: int) -> EncodedRecord:
        if self.renderer is None:
            self.renderer = create_renderer(self.tokenizer, Qwen35RendererConfig(enable_thinking=False))
        input_ids, seq_lens, reads = [], [], []
        for question in record.questions:
            spans = []
            for prompt in candidate_prompts(record.state, question):
                ids = self.renderer.render_ids([{"role": "user", "content": prompt}], add_generation_prompt=True)
                input_ids += ids
                seq_lens.append(len(ids))
                spans.append((len(input_ids) - 1, len(input_ids)))
            reads.append(spans)
        return EncodedRecord(input_ids=input_ids, seq_lens=seq_lens, reads=reads)
