# Ported from vLLM main at fb159244e6fdd0b2ecd9d62855cae9a2d282ae87 (Apache-2.0, Copyright contributors to the vLLM
# project), with vLLM's names; v0.34.0 has neither slot_prompt.py nor SlotLMStrategy:
# - vllm/entrypoints/openai/decisions/slot_prompt.py: all of it (lines 10-68).
# - vllm/entrypoints/openai/decisions/strategies.py: SlotLMStrategy (lines 1460-1632): its MAX_OPTIONS, the label and
#   prefix checks of __init__, limits, _generation_prompt, _prompt, _tokens, _prefix_text, and read's prompts and prefix
#   length (1579-1588).
# - vllm/entrypoints/generate/structured_decisions/serving.py: parse_questions' limit (lines 33-36), with the
#   max_options check of question_types.py's build_question for unlabelled limits. parse_record makes their other
#   checks, and at most 10 score levels (MAX_SCORE_LEVELS).
# vLLM reads the instructions and criteria as a request gives them (Question.raw), as a record holds them. Text only.
# Left out, as only a server uses them: the fp32 head check (lm_head_logits computes in fp32), the decisions API's
# questions, max_model_len, the sampling parameters and the prefix cache.
"""A slot decision model's System One strategy: vLLM's ``SlotLMStrategy`` prompts and prefixes, read from the LM head."""

import json
from typing import Any

from transformers import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5_slot import get_n_slots
from prime_rl.trainer.sft.system_one import record
from prime_rl.trainer.sft.system_one.clef_flash import LABELS, DecisionLimits, StructuredDecisionError
from prime_rl.trainer.sft.system_one.data import ReadQuestion, RenderedRequest
from prime_rl.trainer.sft.system_one.next_token import NextTokenStrategy, reply_label_ids
from prime_rl.utils.logger import get_logger

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# Two-letter labels that are not one token in the Qwen3.5 vocabulary.
_NOT_ONE_TOKEN = frozenset("BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ".split())
_PAIRS = tuple(a + b for a in LABELS for b in LABELS if a + b not in _NOT_ONE_TOKEN)
# Slot i labels option i: A to Z, then the two-letter labels in order.
SLOT_LABELS = LABELS + _PAIRS[: 256 - len(LABELS)]
DEFAULT_INSTRUCTIONS = {
    "choice": "Which option applies to the state?",
    "noul": "Is the statement true, or is the answer to the question yes?",
    "score": "Which level describes the state?",
}
NOUL_DEFAULTS = {
    "true": "the statement is true / the answer is yes",
    "false": "the statement is false / the answer is no",
}
# Placed where a suffix goes to find where the shared prefix ends.
PREFIX_MARK = "QEV_PREFIX_MARK"


def render_state(value: Any, strip: bool = False) -> str:
    """Strings as they are (stripped with ``strip``), others as indented JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip() if strip else value
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_suffix(type_name: str, instructions: Any, criteria: Any) -> str:
    """The question, its options under slot labels and the answer line."""
    if type_name == "noul":
        criteria = criteria or {}
        names = ["yes", "no"]
        descriptions = [d if criteria.get(k) is None else criteria[k] for k, d in NOUL_DEFAULTS.items()]
    elif type_name == "choice":
        names, descriptions = list(criteria), list(criteria.values())
    else:
        names, descriptions = [f"level {i}" for i in range(len(criteria))], criteria
    lines = []
    for label, name, desc in zip(SLOT_LABELS, names, descriptions):
        text = render_state(desc, strip=True)
        lines.append(f"{label}. {name}: {text}" if text else f"{label}. {name}")
    header = "Rate the state:" if type_name == "score" else "Question:"
    question = render_state(instructions, strip=True) or DEFAULT_INSTRUCTIONS[type_name]
    kind = "letter" if len(names) <= len(LABELS) else "label"
    options, labels = "\n".join(lines), ", ".join(SLOT_LABELS[: len(names)])
    return f"{header} {question}\n\nOptions:\n{options}\n\nAnswer with one {kind}: {labels}."


class SlotLMStrategy(NextTokenStrategy):
    """vLLM's SlotLMStrategy without its softmax. Each question is one prompt in an attention segment of its own, split
    in two blocks: the prefix, up to the end of the state, and the question. Its logits are the LM head's, at the
    prompt's last token, for the slot labels of its options."""

    # The most options the checkpoint was trained to read in one question.
    MAX_OPTIONS = 255

    def __init__(self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig):
        super().__init__(tokenizer, config)
        # Each slot label's own token must also be the reply's first token.
        labels = SLOT_LABELS[: get_n_slots(config)]
        _, self.slot_ids = reply_label_ids(tokenizer, self._generation_prompt(), labels)
        if any(self._tokens(label) != [i] for label, i in zip(labels, self.slot_ids)):
            raise ValueError("the slot labels' tokens differ at the start of the reply")
        # The prefix and each suffix are tokenized on their own, which needs the suffix to start a pre-token after the
        # prefix.
        prefix = self._prefix_text("x")
        for type_name in ("choice", "score"):
            suffix = render_suffix(type_name, None, {"x": None})
            tail = self._prompt("x", suffix)[len(prefix) :]
            if self._tokens(prefix) + self._tokens(tail) != self._tokens(prefix + tail):
                raise ValueError("the slot prompt's suffix merges with its prefix")

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=min(len(self.slot_ids), self.MAX_OPTIONS))

    def _generation_prompt(self) -> list[int]:
        return self._tokens(self._prompt("x", "y"))

    def _prompt(self, state: str, suffix: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<state>\n{state}\n</state>\n\n{suffix}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prefix_text(self, state: str) -> str:
        """The prompt up to where the suffix starts."""
        marked = self._prompt(state, PREFIX_MARK)
        # The last mark is the suffix's: the state may contain the mark too.
        return marked[: marked.rindex(PREFIX_MARK)]

    def _check_limits(self, request: record.SystemOneRequest) -> None:
        limits = self.limits()
        if len(request.questions) > limits.max_questions:
            raise StructuredDecisionError(f"questions: at most {limits.max_questions} for this model")
        for question in request.questions:
            if len(question.options) > limits.max_options:
                raise StructuredDecisionError(
                    f"question {question.id!r}: at most {limits.max_options} options for this model"
                )

    def render_request(self, request: record.SystemOneRequest, max_tokens: int) -> RenderedRequest | None:
        """vLLM's prompt of each question, each an attention segment with vLLM's prefix, read at its last token. A
        request vLLM refuses is skipped. Not bounded by ``max_tokens``: a longer rendering is skipped."""
        try:
            self._check_limits(request)
        except StructuredDecisionError as error:
            get_logger().warning(f"Skipping record {request.id!r}: vLLM refuses it: {error}")
            return None
        state = render_state(request.state)
        # The state is most of every prompt, so it is tokenized once.
        prefix_text = self._prefix_text(state)
        prefix = self._tokens(prefix_text)
        input_ids: list[int] = []
        seq_lens, read_spans = [], []
        for question in request.questions:
            prompt = self._prompt(state, render_suffix(question.type, question.instructions, question.criteria))
            assert prompt.startswith(prefix_text)
            prompt_ids = prefix + self._tokens(prompt[len(prefix_text) :])
            input_ids += prompt_ids
            seq_lens.append(len(prompt_ids))
            read_spans.append([(len(input_ids) - 1, len(input_ids))])
        return RenderedRequest(
            input_ids=input_ids,
            seq_lens=seq_lens,
            read_spans=read_spans,
            # A noul's slots are yes, then no
            options=[["true", "false"] if q.type == "noul" else q.options for q in request.questions],
            prefix_lens=[len(prefix)] * len(seq_lens),
        )

    def _label_ids(self, question: ReadQuestion) -> list[int]:
        """The tokens of the slots of the question's options, in logit order."""
        return self.slot_ids[: len(question.options)]
