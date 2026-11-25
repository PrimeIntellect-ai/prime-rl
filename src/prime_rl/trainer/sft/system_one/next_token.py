# Ported from vLLM v0.34.0 (Apache-2.0, Copyright contributors to the vLLM project), with vLLM's names:
# - vllm/entrypoints/openai/decisions/strategies.py: state_text (lines 141-142), prompt_tail (212-216),
#   reply_label_ids (219-238), type_label_ids (241-250) and NextTokenStrategy (263-510): its __init__,
#   _generation_prompt, _content and limits. _render applies the tokenizer's chat template with the chat options of
#   ReadPromptRequest.build_chat_params (119-131): the generation prompt, with thinking off.
# - vllm/entrypoints/openai/decisions/question_types.py: QuestionType's reply_instruction, reply_line, body and prompt
#   (lines 45, 58-79) with NoulQuestion's and ScoreQuestion's own (184, 201-204, 227), as functions of clef_flash.py's
#   Question, whose port leaves them and Question.labels out. The rest of that file and parse_questions are
#   clef_flash.py's.
# Text only, and a record has no instructions of its own: no image and no system message (vLLM's top-level
# instructions). Left out, as only a server uses them: read, _question_read, the sampling parameters and
# _warm_shared_prefix. Left out too: _prompts' check that a prompt ends with the tail. A server renders a read with
# the request's chat options and its renderer's template, which may differ from the probe's; here _render renders
# both alike, so the check could only fail with a chat template that puts no added token after the user turn's
# content, and Qwen3.5's puts <|im_end|> there.
"""A stock Qwen3.5's System One strategy: vLLM's ``NextTokenStrategy`` prompts, read from the LM head."""

import json
from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn
from transformers import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.system_one import record
from prime_rl.trainer.sft.system_one.clef_flash import (
    LABELS,
    QUESTION_TYPES,
    DecisionLimits,
    Option,
    Question,
    StructuredDecisionError,
    parse_questions,
)
from prime_rl.trainer.sft.system_one.data import ReadQuestion, RenderedRequest, SystemOneStrategy
from prime_rl.trainer.sft.system_one.lm_head import lm_head_logits
from prime_rl.utils.logger import get_logger

REPLY_INSTRUCTIONS = {
    "choice": "Answer with the letter of one option only.",
    "noul": "Answer with yes or no only.",
    "score": "Answer with the number of one level only.",
}


def reply_line(question: Question, label: str, option: Option) -> str:
    if question.type.name == "noul":
        # A side without a description still shows its label. Without the label in the prompt, models answer "Yes"
        # or "No" instead.
        return f"{label}: {option.description}" if option.description else label
    return f"{label}: {option.name} - {option.description}" if option.description else f"{label}: {option.name}"


def body(question: Question) -> str:
    """The options and the reply instruction, after the question's heading."""
    labels = question.type.label_set[: len(question.options)]
    lines = [reply_line(question, label, o) for label, o in zip(labels, question.options)]
    lines.append(REPLY_INSTRUCTIONS[question.type.name])
    return "\n".join(lines)


def prompt(question: Question) -> str:
    """The question as the model reads it, after the state."""
    if not question.instructions:
        return body(question)
    return f"Question: {question.instructions}\n{body(question)}"


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def prompt_tail(tokenizer: PreTrainedTokenizer, prompt_ids: Sequence[int]) -> list[int]:
    """The prompt's ids after its last added token."""
    added = set(tokenizer.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
    return list(prompt_ids[start:])


def reply_label_ids(
    tokenizer: PreTrainedTokenizer, prompt_ids: Sequence[int], labels: Sequence[str] = LABELS
) -> tuple[list[int], list[int]]:
    """The prompt's ids after its last added token, and the token each of ``labels`` adds after them as the first
    token of the reply. Raises ValueError if a label is not one distinct token there."""
    tail = prompt_tail(tokenizer, prompt_ids)
    text = tokenizer.decode(tail)
    ids: list[int] = []
    for label in labels:
        extended = tokenizer.encode(text + label, add_special_tokens=False)
        if extended[:-1] != tail or extended[-1] in ids:
            raise ValueError(f"label {label!r} is not one distinct token after this model's chat prompt")
        ids.append(extended[-1])
    return tail, ids


def type_label_ids(tokenizer: PreTrainedTokenizer, prompt_ids: Sequence[int]) -> dict[str, int]:
    """The reply's first token for each label of every question type. Raises ValueError as reply_label_ids does."""
    label_ids: dict[str, int] = {}
    for qtype in QUESTION_TYPES.values():
        _, ids = reply_label_ids(tokenizer, prompt_ids, qtype.label_set)
        label_ids.update(zip(qtype.label_set, ids))
    return label_ids


class NextTokenStrategy(SystemOneStrategy):
    """vLLM's NextTokenStrategy without its softmax. Each question is one prompt in an attention segment of its own:
    the state, then the question with its labelled options. Its logits are the LM head's, at the prompt's last token,
    for the tokens of its options' labels."""

    def __init__(self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig):
        super().__init__(tokenizer, config)
        # Every prompt ends with the same generation prompt, so a label's token is the same for every question.
        self.label_ids = type_label_ids(tokenizer, self._generation_prompt())

    def _render(self, content: str) -> list[int]:
        """A user turn as a chat prompt's ids, up to the generation prompt, with thinking off."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _generation_prompt(self) -> list[int]:
        """A chat prompt's ids up to the generation prompt, with thinking off."""
        return self._render("x")

    def _content(self, state: str, question: Question) -> str:
        """The user turn: the state, then the question."""
        return f"{state}\n\n{prompt(question)}"

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))

    def render_request(self, request: record.SystemOneRequest, max_tokens: int) -> RenderedRequest | None:
        """vLLM's prompt of each question, each an attention segment read at its last token. A request vLLM refuses is
        skipped. Not bounded by ``max_tokens``: a longer rendering is skipped."""
        try:
            questions = parse_questions(request.questions, self.limits())
        except StructuredDecisionError as error:
            get_logger().warning(f"Skipping record {request.id!r}: vLLM refuses it: {error}")
            return None
        state = state_text(request.state)
        input_ids: list[int] = []
        seq_lens, read_spans = [], []
        for question in questions:
            prompt_ids = self._render(self._content(state, question))
            input_ids += prompt_ids
            seq_lens.append(len(prompt_ids))
            read_spans.append([(len(input_ids) - 1, len(input_ids))])
        # vLLM's labels of a noul are yes, then no
        options = [["true", "false"] if q.type == "noul" else q.options for q in request.questions]
        return RenderedRequest(input_ids=input_ids, seq_lens=seq_lens, read_spans=read_spans, options=options)

    def _label_ids(self, question: ReadQuestion) -> list[int]:
        """The tokens of the labels of the question's options, in logit order."""
        return [self.label_ids[label] for label in QUESTION_TYPES[question.type].label_set[: len(question.options)]]

    def option_logits(
        self, model: nn.Module, hidden_states: Tensor, input_ids: Tensor, questions: list[ReadQuestion]
    ) -> Tensor:
        """Each question's logits of its labels' tokens at its prompt's last token, from the LM head's rows of those
        tokens, as vLLM reads their logprobs for the reply's first token."""
        positions = [question.reads[0][0] for question in questions]
        return lm_head_logits(model, hidden_states, positions, [self._label_ids(question) for question in questions])
