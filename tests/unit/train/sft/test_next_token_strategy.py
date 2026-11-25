"""A stock Qwen3.5's System One strategy renders vLLM v0.34.0's NextTokenStrategy prompts and reads the LM head."""

import json
import string
from itertools import accumulate
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset
from torch import nn
from transformers import PretrainedConfig, PreTrainedTokenizerFast

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.model import ExtraConfigSelection
from prime_rl.trainer.models.qwen3_5.modeling_clef_flash import ClefFlashForJointSchema
from prime_rl.trainer.sft.system_one import next_token
from prime_rl.trainer.sft.system_one.clef_flash import ClefFlashStrategy
from prime_rl.trainer.sft.system_one.data import RenderedRequest, SystemOneSource, render_sample
from prime_rl.trainer.sft.system_one.loss import build_decision_loss_inputs
from prime_rl.trainer.sft.system_one.next_token import NextTokenStrategy
from prime_rl.trainer.sft.system_one.record import parse_record
from prime_rl.trainer.sft.system_one.strategies import get_system_one_strategy_cls
from tests.unit.train.models.qwen3_5.clef_flash_release import HIDDEN, VOCAB, tiny_config, tokenizer

QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Is a service down?", "criteria": {"true": "A service is down."}},
    "team": {"type": "choice", "instructions": "Who handles it?", "criteria": {"technical": "Bugs", "billing": None}},
    "urgency": {"type": "score", "criteria": ["Can wait", "Today"]},
}
STATE = "Checkout fails."
# vLLM v0.34.0's user turn for each of these questions, written out (vllm/entrypoints/openai/decisions/strategies.py,
# lines 321-323, and question_types.py, lines 58-79 and 201-204).
CONTENTS = [
    "Checkout fails.\n\nQuestion: Is a service down?\nyes: A service is down.\nno\nAnswer with yes or no only.",
    "Checkout fails.\n\nQuestion: Who handles it?\nA: technical - Bugs\nB: billing\n"
    "Answer with the letter of one option only.",
    "Checkout fails.\n\n0: Can wait\n1: Today\nAnswer with the number of one level only.",
]
# ChatML turns. With thinking off the prompt ends with an empty think block, as Qwen3.5's does in vLLM's tests
# (tests/entrypoints/generate/structured_decisions/test_slot_strategy.py).
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)
# "yes" and "no" as one token each, as a noul's labels must be
MERGES = (("y", "e"), ("ye", "s"), ("n", "o"))


def chat(content: str) -> str:
    """CHAT_TEMPLATE's prompt for a user turn, with thinking off."""
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def chat_tokenizer(merges: tuple[tuple[str, str], ...] = MERGES) -> PreTrainedTokenizerFast:
    tokens = tokenizer(merges)
    tokens.chat_template = CHAT_TEMPLATE
    return tokens


def row(state=STATE, questions=QUESTIONS, label=None) -> dict:
    return {
        "id": "r",
        "state": json.dumps(state),
        "questions": json.dumps(questions),
        "label": json.dumps(label or {"outage": True}),
    }


def request(state=STATE, questions=QUESTIONS):
    return parse_record({**row(state, questions), "label": None}, SystemOneColumnsConfig())[0]


def prompts(rendered: RenderedRequest) -> list[list[int]]:
    """The rendering's attention segments."""
    ends = accumulate(rendered["seq_lens"])
    return [rendered["input_ids"][end - length : end] for end, length in zip(ends, rendered["seq_lens"])]


@pytest.fixture(scope="module")
def strategy() -> NextTokenStrategy:
    return NextTokenStrategy(chat_tokenizer(), tiny_config())


def test_renders_vllms_prompt_of_each_question_in_a_segment_of_its_own(strategy):
    rendered = strategy.render_request(request(), max_tokens=4096)

    texts = [strategy.tokenizer.decode(prompt) for prompt in prompts(rendered)]
    assert texts == [chat(content) for content in CONTENTS]
    ends = list(accumulate(rendered["seq_lens"]))
    assert ends[-1] == len(rendered["input_ids"])
    assert rendered["read_spans"] == [[(end - 1, end)] for end in ends]
    assert rendered["options"] == [["true", "false"], ["technical", "billing"], ["0", "1"]]


def test_targets_follow_vllms_labels_and_positions_restart_in_each_segment(strategy):
    labelled = row(label={"outage": True, "team": "billing", "urgency": 0})
    sample = SystemOneSource(Dataset.from_list([labelled]), strategy, seq_len=4096)._process(labelled)
    questions = sample["sample_data"]["questions"]
    assert [question["options"] for question in questions] == [["true", "false"], ["technical", "billing"], ["0", "1"]]
    assert [question["target"] for question in questions] == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    ends = list(accumulate(sample["seq_lens"]))
    assert [question["reads"] for question in questions] == [[[end - 1, end]] for end in ends]
    assert sample["position_ids"].tolist() == [position for length in sample["seq_lens"] for position in range(length)]
    assert np.flatnonzero(sample["loss_mask"]).tolist() == [end - 1 for end in ends]


def test_a_state_that_is_not_text_is_written_as_vllm_writes_it(strategy):
    state = {"note": "café", "seen": [1, 2]}
    rendered = strategy.render_request(request(state, {"q": {"type": "noul"}}), max_tokens=4096)
    content = '{"note": "café", "seen": [1, 2]}\n\nyes\nno\nAnswer with yes or no only.'
    assert strategy.tokenizer.decode(rendered["input_ids"]) == chat(content)


def test_instructions_that_are_not_text_are_written_as_vllm_writes_them(strategy):
    questions = {"q": {"type": "noul", "instructions": {"ask": "down?"}}}
    rendered = strategy.render_request(request(questions=questions), max_tokens=4096)
    content = "Checkout fails.\n\nQuestion: {'ask': 'down?'}\nyes\nno\nAnswer with yes or no only."
    assert strategy.tokenizer.decode(rendered["input_ids"]) == chat(content)


def test_a_labels_token_is_the_first_token_of_the_reply(strategy):
    labels = [*string.ascii_uppercase, "yes", "no", *string.digits]
    assert list(strategy.label_ids) == labels
    assert strategy.tokenizer.convert_ids_to_tokens(list(strategy.label_ids.values())) == labels
    # One token per byte: "yes" is three tokens
    with pytest.raises(ValueError, match="label 'yes' is not one distinct token after this model's chat prompt"):
        NextTokenStrategy(chat_tokenizer(merges=()), tiny_config())


def test_skips_what_vllm_refuses(strategy, monkeypatch):
    warnings = []
    monkeypatch.setattr(next_token, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    many = {f"q{i}": {"type": "noul"} for i in range(65)}
    assert strategy.render_request(request(questions=many), max_tokens=4096) is None
    assert strategy.render_request(request(questions=dict(list(many.items())[:64])), max_tokens=4096) is not None

    choices = {chr(ord("a") + i) * 2: None for i in range(26)}
    assert strategy.render_request(request(questions={"c": {"type": "choice", "criteria": choices}}), 4096) is not None
    choices["zz2"] = None
    assert strategy.render_request(request(questions={"c": {"type": "choice", "criteria": choices}}), 4096) is None
    assert warnings == [
        "Skipping record 'r': vLLM refuses it: questions: at most 64 for this model",
        "Skipping record 'r': vLLM refuses it: question 'c': at most 26 options for this model",
    ]


def test_seq_len_skips_a_longer_rendering(strategy):
    full = strategy.render_request(request(), max_tokens=4096)
    assert strategy.render_request(request(), max_tokens=1) == full
    assert render_sample(strategy, request(), len(full["input_ids"])) is not None
    assert render_sample(strategy, request(), len(full["input_ids"]) - 1) is None


def test_option_logits_are_the_lm_heads_at_the_last_token_of_each_prompt(strategy):
    labelled = row(label={"outage": True, "team": "billing", "urgency": 0})
    sample = SystemOneSource(Dataset.from_list([labelled]), strategy, seq_len=4096)._process(labelled)
    # The record twice in one row: the second's reads are at row positions, after the first
    num_tokens = len(sample["input_ids"])
    input_ids = torch.from_numpy(sample["input_ids"]).repeat(2)
    questions = build_decision_loss_inputs([num_tokens] * 2, [sample["sample_data"]] * 2).questions
    torch.manual_seed(0)
    model = nn.Module()
    model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    hidden_states = torch.randn(len(input_ids), HIDDEN, requires_grad=True)

    logits = strategy.option_logits(model, hidden_states, input_ids, questions)

    last = [start + end - 1 for start in (0, num_tokens) for end in accumulate(sample["seq_lens"])]
    label_ids = [[strategy.label_ids[label] for label in labels] for labels in (["yes", "no"], ["A", "B"], ["0", "1"])]
    with torch.no_grad():
        full = model.lm_head(hidden_states)
    assert logits.dtype == torch.float32
    expected = torch.cat([full[position, ids] for position, ids in zip(last, label_ids * 2, strict=True)])
    torch.testing.assert_close(logits, expected)

    # The gradient reaches the LM head's rows of the six labels and the six read tokens, and nothing else
    logits.sum().backward()
    assert model.lm_head.weight.grad.abs().sum(-1).nonzero().flatten().tolist() == sorted(sum(label_ids, []))
    assert hidden_states.grad.abs().sum(-1).nonzero().flatten().tolist() == last


def test_a_stock_qwen3_5_checkpoint_finds_the_strategy():
    assert get_system_one_strategy_cls(tiny_config()) is NextTokenStrategy
    # A Clef-Flash release has the same config.json: its strategy is found by the selected class's name
    selection = ExtraConfigSelection(ClefFlashForJointSchema, {})
    assert get_system_one_strategy_cls(tiny_config(), selection) is ClefFlashStrategy
    # As in vLLM, a text-only checkpoint has no strategy
    with pytest.raises(ValueError, match=r"architectures \['Qwen3_5ForCausalLM'\] do not start with one"):
        get_system_one_strategy_cls(PretrainedConfig(architectures=["Qwen3_5ForCausalLM"]))


def test_renders_what_vllm_renders(strategy):
    strategies = pytest.importorskip("vllm.entrypoints.openai.decisions.strategies")
    from vllm.entrypoints.openai.decisions.question_types import build_question

    tokens = strategy.tokenizer
    context = SimpleNamespace(
        online_renderer=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokens)),
        engine_client=SimpleNamespace(model_config=SimpleNamespace(supports_multimodal_inputs=False)),
        chat_template=None,
        default_chat_template_kwargs={},
    )
    vllm_strategy = strategies.NextTokenStrategy(context)
    assert strategy._generation_prompt() == vllm_strategy._generation_prompt()
    assert strategy.label_ids == vllm_strategy.label_ids
    # A server renders a read through its renderer (preprocess_chat). vLLM's prompts here are the tokenizer's template
    # with the chat options of a read whose request sets none: the generation prompt, with thinking off
    chat_kwargs = strategies.ReadPromptRequest().build_chat_params(None, "auto").get_apply_chat_template_kwargs()

    limits = vllm_strategy.limits()
    choices = {"zeta": "Last", "alpha": None, "mid": "Middle"}
    records = [
        (STATE, QUESTIONS),
        (
            {"ticket": "Checkout fails."},
            {"c": {"type": "choice", "instructions": {"ask": "who?"}, "criteria": choices}},
        ),
        ({"logs": ["café", 1.5, None]}, {"n": {"type": "noul"}, "s": {"type": "score", "criteria": ["a", "b", "c"]}}),
    ]
    for state, questions in records:
        vllm_questions = [
            build_question(qid, q["type"], q.get("instructions"), q.get("criteria"), limits.max_options)
            for qid, q in questions.items()
        ]
        contents = [vllm_strategy._content(vllm_strategy.render_state(state), q) for q in vllm_questions]
        vllm_prompts = [
            list(tokens.apply_chat_template([{"role": "user", "content": content}], **chat_kwargs))
            for content in contents
        ]
        # vLLM's own check of each prompt, in _prompts
        assert all(prompt[len(prompt) - len(vllm_strategy.tail) :] == vllm_strategy.tail for prompt in vllm_prompts)

        assert prompts(strategy.render_request(request(state, questions), max_tokens=4096)) == vllm_prompts
        sample = render_sample(strategy, request(state, questions), 4096)
        read_questions = build_decision_loss_inputs([len(sample["input_ids"])], [sample["sample_data"]]).questions
        label_groups = [[[token] for token in strategy._label_ids(question)] for question in read_questions]
        assert label_groups == [vllm_strategy._label_groups(q) for q in vllm_questions]
