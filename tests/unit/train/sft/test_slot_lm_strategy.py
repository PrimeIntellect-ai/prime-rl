"""A slot decision model's System One strategy renders vLLM main's SlotLMStrategy prompts and prefixes and reads the LM
head."""

import json
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from torch import nn

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.model import ExtraConfigSelection
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForSlotDecision
from prime_rl.trainer.sft.system_one import slot_lm
from prime_rl.trainer.sft.system_one.data import RenderedRequest, SystemOneSource, render_sample
from prime_rl.trainer.sft.system_one.loss import build_decision_loss_inputs
from prime_rl.trainer.sft.system_one.record import parse_record
from prime_rl.trainer.sft.system_one.slot_lm import SLOT_LABELS, SlotLMStrategy
from prime_rl.trainer.sft.system_one.strategies import get_system_one_strategy_cls
from tests.unit.train.models.qwen3_5.clef_flash_release import HIDDEN, VOCAB, tiny_config
from tests.unit.train.models.qwen3_5.lm_head_release import MERGES, MODELING_OVERRIDES, chat_tokenizer

# vLLM's test of the strategy (tests/entrypoints/generate/structured_decisions/test_slot_strategy.py at fb159244): the
# state and questions that qev, the checkpoints' reference runtime, probes, and qev's prompts for them.
PROBE_STATE = {
    "ticket": "My card was charged twice for the same order and support has not replied in 3 days.",
    "customer": {"tenure_months": 14, "vip": False},
}
PROBE_QUESTIONS = {
    "a": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "b": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "charges and refunds", "technical": "bugs", "sales": "pricing"},
    },
    "c": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["calm", "irritated", "angry"],
    },
}
SYSTEM_TURN = (
    "<|im_start|>system\nApply the question to the state. Choose exactly one of the listed options. Respond with only "
    "its uppercase letter, with no explanation or reasoning.<|im_end|>\n"
)
PREFIX = (
    SYSTEM_TURN + '<|im_start|>user\n<state>\n{\n  "ticket": "My card was charged twice for the same order and support '
    'has not replied in 3 days.",\n  "customer": {\n    "tenure_months": 14,\n    "vip": false\n  }\n}\n</state>\n\n'
)
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QEV_PROMPTS = [
    PREFIX + "Question: Is the customer asking for money back?\n\nOptions:\n"
    "A. yes: the statement is true / the answer is yes\n"
    "B. no: the statement is false / the answer is no\n\n"
    "Answer with one letter: A, B." + TAIL,
    PREFIX + "Question: Which team should handle this?\n\nOptions:\n"
    "A. billing: charges and refunds\nB. technical: bugs\nC. sales: pricing\n\n"
    "Answer with one letter: A, B, C." + TAIL,
    PREFIX + "Rate the state: How frustrated is the customer?\n\nOptions:\n"
    "A. level 0: calm\nB. level 1: irritated\nC. level 2: angry\n\n"
    "Answer with one letter: A, B, C." + TAIL,
]


def slot_config(n_slots: int = 26):
    config = tiny_config()
    config.modeling_overrides = {**MODELING_OVERRIDES, "n_slots": n_slots}
    return config


def request(state=PROBE_STATE, questions=PROBE_QUESTIONS):
    row = {"id": "r", "state": json.dumps(state), "questions": json.dumps(questions)}
    return parse_record(row, SystemOneColumnsConfig())[0]


def prompts(rendered: RenderedRequest) -> list[list[int]]:
    """The rendering's attention segments."""
    ends = accumulate(rendered["seq_lens"])
    return [rendered["input_ids"][end - length : end] for end, length in zip(ends, rendered["seq_lens"])]


@pytest.fixture(scope="module")
def strategy() -> SlotLMStrategy:
    return SlotLMStrategy(chat_tokenizer(), slot_config())


def test_renders_the_reference_runtimes_prompts_with_the_state_as_prefix(strategy):
    rendered = strategy.render_request(request(), max_tokens=4096)

    decode = strategy.tokenizer.decode
    assert [decode(prompt) for prompt in prompts(rendered)] == QEV_PROMPTS
    assert [decode(p[:n]) for p, n in zip(prompts(rendered), rendered["prefix_lens"], strict=True)] == [PREFIX] * 3
    ends = list(accumulate(rendered["seq_lens"]))
    assert ends[-1] == len(rendered["input_ids"])
    assert rendered["read_spans"] == [[(end - 1, end)] for end in ends]
    assert rendered["options"] == [["true", "false"], ["billing", "technical", "sales"], ["0", "1", "2"]]


def test_defaults_descriptions_and_a_text_state(strategy):
    questions = {
        "n": {"type": "noul", "instructions": "  ", "criteria": {"false": " Nothing is down. "}},
        "c": {"type": "choice", "instructions": {"ask": "who?"}, "criteria": {"x": None, "y": {"id": 1}, "z": ""}},
        "s": {"type": "score", "criteria": ["low", {"at": "café"}]},
    }
    state = f"  {slot_lm.PREFIX_MARK} "
    rendered = strategy.render_request(request(state, questions), max_tokens=4096)
    prefix = f"{SYSTEM_TURN}<|im_start|>user\n<state>\n{state}\n</state>\n\n"
    suffixes = [
        "Question: Is the statement true, or is the answer to the question yes?\n\nOptions:\n"
        "A. yes: the statement is true / the answer is yes\nB. no: Nothing is down.\n\nAnswer with one letter: A, B.",
        'Question: {\n  "ask": "who?"\n}\n\nOptions:\nA. x\nB. y: {\n  "id": 1\n}\nC. z\n\n'
        "Answer with one letter: A, B, C.",
        'Rate the state: Which level describes the state?\n\nOptions:\nA. level 0: low\nB. level 1: {\n  "at": "café"\n}'
        "\n\nAnswer with one letter: A, B.",
    ]
    decode = strategy.tokenizer.decode
    assert [decode(prompt) for prompt in prompts(rendered)] == [prefix + suffix + TAIL for suffix in suffixes]
    assert rendered["prefix_lens"] == [len(strategy._tokens(prefix))] * 3


def test_slot_labels():
    assert len(SLOT_LABELS) == len(set(SLOT_LABELS)) == 256
    assert SLOT_LABELS[:26] == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") and SLOT_LABELS[26:28] == ("AA", "AB")
    assert SLOT_LABELS[-1] == "JU"


def test_more_than_26_options_use_two_letter_labels():
    strategy = SlotLMStrategy(chat_tokenizer((*MERGES, ("A", "A"), ("A", "B"))), slot_config(28))
    choices = {f"option {i}": None for i in range(28)}
    rendered = strategy.render_request(request("s", {"q": {"type": "choice", "criteria": choices}}), max_tokens=4096)
    text = strategy.tokenizer.decode(rendered["input_ids"])
    assert "\nZ. option 25\nAA. option 26\nAB. option 27\n\nAnswer with one label: A, B, C," in text
    assert strategy.tokenizer.convert_ids_to_tokens(strategy.slot_ids) == list(SLOT_LABELS[:28])


def test_refuses_a_tokenizer_the_prompt_does_not_fit():
    # No token for AA
    with pytest.raises(ValueError, match="label 'AA' is not one distinct token after this model's chat prompt"):
        SlotLMStrategy(chat_tokenizer(), slot_config(27))
    # The prefix's last newline and the suffix's Qu are one token
    with pytest.raises(ValueError, match="the slot prompt's suffix merges with its prefix"):
        SlotLMStrategy(chat_tokenizer((*MERGES, ("Q", "u"), ("Ċ", "Qu"))), slot_config())


def test_skips_what_vllm_refuses(strategy, monkeypatch):
    warnings = []
    monkeypatch.setattr(slot_lm, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    many = {f"q{i}": {"type": "noul"} for i in range(65)}
    assert strategy.render_request(request(questions=many), max_tokens=4096) is None
    assert strategy.render_request(request(questions=dict(list(many.items())[:64])), max_tokens=4096) is not None
    choices = {f"o{i}": None for i in range(27)}
    assert strategy.render_request(request(questions={"c": {"type": "choice", "criteria": choices}}), 4096) is None
    del choices["o26"]
    assert strategy.render_request(request(questions={"c": {"type": "choice", "criteria": choices}}), 4096) is not None
    assert warnings == [
        "Skipping record 'r': vLLM refuses it: questions: at most 64 for this model",
        "Skipping record 'r': vLLM refuses it: question 'c': at most 26 options for this model",
    ]


def test_a_question_has_at_most_255_options(strategy, monkeypatch):
    monkeypatch.setattr(strategy, "slot_ids", list(range(256)))
    assert strategy.limits().max_options == 255


def test_option_logits_are_the_lm_heads_for_the_slots_of_the_options(strategy):
    row = {
        "id": "r",
        "state": json.dumps(PROBE_STATE),
        "questions": json.dumps(PROBE_QUESTIONS),
        "label": json.dumps({"a": True, "b": "sales", "c": 0}),
    }
    sample = SystemOneSource(Dataset.from_list([row]), strategy, seq_len=4096)._process(row)
    questions = sample["sample_data"]["questions"]
    assert [question["target"] for question in questions] == [[1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    inputs = build_decision_loss_inputs([len(sample["input_ids"])], [sample["sample_data"]])
    torch.manual_seed(0)
    model = nn.Module()
    model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    hidden_states = torch.randn(len(sample["input_ids"]), HIDDEN)

    logits = strategy.option_logits(model, hidden_states, torch.from_numpy(sample["input_ids"]), inputs.questions)

    slots = strategy.tokenizer.convert_tokens_to_ids(["A", "B", "C"])
    full = model.lm_head(hidden_states)
    ends = accumulate(sample["seq_lens"])
    expected = [full[end - 1, slots[:n]] for end, n in zip(ends, (2, 3, 3), strict=True)]
    torch.testing.assert_close(logits, torch.cat(expected))


def test_a_slot_checkpoint_finds_the_strategy():
    selection = ExtraConfigSelection(Qwen3_5ForSlotDecision, MODELING_OVERRIDES)
    assert get_system_one_strategy_cls(slot_config(), selection) is SlotLMStrategy


def test_renders_what_vllm_renders(strategy):
    """Skipped until the installed vLLM has the strategy, which v0.34.0 does not."""
    pytest.importorskip("vllm.entrypoints.openai.decisions.slot_prompt")
    from vllm.entrypoints.openai.decisions import slot_prompt, strategies
    from vllm.entrypoints.openai.decisions.question_types import build_question

    assert SLOT_LABELS == slot_prompt.SLOT_LABELS
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(modeling_overrides=MODELING_OVERRIDES),
        head_dtype=torch.float32,
        max_model_len=4096,
        supports_multimodal_inputs=False,
    )
    context = SimpleNamespace(
        online_renderer=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: strategy.tokenizer)),
        engine_client=SimpleNamespace(model_config=model_config),
        chat_template=None,
        default_chat_template_kwargs={},
    )
    vllm_strategy = strategies.SlotLMStrategy(context)
    limits = vllm_strategy.limits()
    assert strategy.slot_ids == vllm_strategy.slot_ids
    assert strategy.limits().max_questions == limits.max_questions
    assert strategy.limits().max_options == limits.max_options

    records = [
        (PROBE_STATE, PROBE_QUESTIONS),
        (
            "text",
            {"n": {"type": "noul", "criteria": {"true": " Yes. "}}, "s": {"type": "score", "criteria": ["a", "b"]}},
        ),
        ({"logs": ["café", 1.5, None]}, {"c": {"type": "choice", "criteria": {"z": {"id": 1}, "a": None}}}),
    ]
    for state, questions in records:
        # As vLLM's read builds its prompts
        state_text = vllm_strategy.render_state(state)
        prefix_text = vllm_strategy._prefix_text(state_text)
        prefix = vllm_strategy._tokens(prefix_text)
        vllm_prompts = []
        for qid, q in questions.items():
            question = build_question(
                qid, q["type"], q.get("instructions"), q.get("criteria"), limits.max_options, labeled=limits.labeled
            )
            prompt = vllm_strategy._prompt(state_text, vllm_strategy._suffix(question))
            vllm_prompts.append(prefix + vllm_strategy._tokens(prompt[len(prefix_text) :]))
        rendered = strategy.render_request(request(state, questions), max_tokens=4096)
        assert prompts(rendered) == vllm_prompts
        assert rendered["prefix_lens"] == [len(prefix)] * len(vllm_prompts)
        assert render_sample(strategy, request(state, questions), 4096)["sample_data"]["prefix_lens"] == [
            len(prefix)
        ] * len(vllm_prompts)
