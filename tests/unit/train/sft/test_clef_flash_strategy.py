"""Clef-Flash's System One strategy renders vLLM v0.34.0's ClefFlashStrategy prompt."""

import json
from types import SimpleNamespace

import pytest
from datasets import Dataset

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.sft.system_one import clef_flash
from prime_rl.trainer.sft.system_one.clef_flash import ClefFlashStrategy
from prime_rl.trainer.sft.system_one.data import SystemOneSource, render_sample
from prime_rl.trainer.sft.system_one.record import parse_record
from tests.unit.train.models.qwen3_5.clef_flash_release import tiny_config, tokenizer

QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Is a service down?", "criteria": {"true": "A service is down."}},
    "team": {"type": "choice", "instructions": "Who handles it?", "criteria": {"technical": "Bugs", "billing": None}},
    "urgency": {"type": "score", "criteria": ["Can wait", "Today"]},
}
STATE = {"ticket": "Checkout fails.", "plan": "pro"}
# vLLM v0.34.0's prompt for this request, written out (vllm/entrypoints/openai/decisions/strategies.py, lines 854-888
# and 998-1068).
PROMPT = (
    "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly "
    "one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n"
    '{"plan":"pro","ticket":"Checkout fails."}'
    "\n\nSCHEMA FIELDS:\n"
    "\nFIELD 1\nID: outage\nTYPE: noul\nINSTRUCTION: Is a service down?\nALLOWED OPTIONS:\n"
    'OPTION 1: {"description":"A service is down.","option_id":"true"}\n'
    'OPTION 2: {"description":"The proposition is false or the answer is no.","option_id":"false"}\n'
    "END FIELD\n"
    "\nFIELD 2\nID: team\nTYPE: choice\nINSTRUCTION: Who handles it?\nALLOWED OPTIONS:\n"
    'OPTION 1: {"option_id":"billing"}\n'
    'OPTION 2: {"description":"Bugs","option_id":"technical"}\n'
    "END FIELD\n"
    "\nFIELD 3\nID: urgency\nTYPE: score\nINSTRUCTION: urgency\nALLOWED OPTIONS:\n"
    'OPTION 1: {"description":"Can wait","option_id":"0"}\n'
    'OPTION 2: {"description":"Today","option_id":"1"}\n'
    "END FIELD\n"
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
)


def row(state=STATE, questions=QUESTIONS, label=None) -> dict:
    return {
        "id": "r",
        "state": json.dumps(state),
        "questions": json.dumps(questions),
        "label": json.dumps(label or {"outage": True}),
    }


def request(state=STATE, questions=QUESTIONS):
    return parse_record({**row(state, questions), "label": None}, SystemOneColumnsConfig())[0]


@pytest.fixture(scope="module")
def strategy() -> ClefFlashStrategy:
    return ClefFlashStrategy(tokenizer(), tiny_config())


def bounded(max_length: int, monkeypatch) -> ClefFlashStrategy:
    """The strategy with ``MAX_LENGTH = max_length``."""
    monkeypatch.setattr(ClefFlashStrategy, "MAX_LENGTH", max_length)
    return ClefFlashStrategy(tokenizer(), tiny_config())


def test_renders_vllms_prompt_with_the_heads_spans(strategy):
    rendered = strategy.render_request(request(), max_tokens=4096)
    decode = strategy.tokenizer.decode

    assert decode(rendered["input_ids"]) == PROMPT
    assert rendered["seq_lens"] == [len(rendered["input_ids"])]
    assert rendered["options"] == [["true", "false"], ["billing", "technical"], ["0", "1"]]
    texts = [[decode(rendered["input_ids"][start:end]) for start, end in spans] for spans in rendered["read_spans"]]
    assert texts == [
        [
            "Is a service down?",
            '{"description":"A service is down.","option_id":"true"}',
            '{"description":"The proposition is false or the answer is no.","option_id":"false"}',
        ],
        ["Who handles it?", '{"option_id":"billing"}', '{"description":"Bugs","option_id":"technical"}'],
        ["urgency", '{"description":"Can wait","option_id":"0"}', '{"description":"Today","option_id":"1"}'],
    ]


def test_targets_follow_the_heads_option_order(strategy):
    labelled = row(label={"outage": True, "team": "technical", "urgency": 0})
    sample = SystemOneSource(Dataset.from_list([labelled]), strategy, seq_len=4096)._process(labelled)
    questions = sample["sample_data"]["questions"]
    assert [question["options"] for question in questions] == [["true", "false"], ["billing", "technical"], ["0", "1"]]
    assert [question["target"] for question in questions] == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]


def test_a_text_state_is_read_as_is(strategy):
    rendered = strategy.render_request(request(state="Checkout fails."), max_tokens=4096)
    assert "STATE:\nCheckout fails.\n\nSCHEMA FIELDS:" in strategy.tokenizer.decode(rendered["input_ids"])


def test_the_state_is_cut_to_fit_max_length_as_vllm_cuts_it(strategy, monkeypatch):
    full = strategy.render_request(request(), max_tokens=4096)
    state_tokens = len(strategy.tokenizer.encode(json.dumps(STATE, separators=(",", ":"), sort_keys=True)))
    fixed = len(full["input_ids"]) - state_tokens

    warnings = []
    monkeypatch.setattr(clef_flash, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    cut = bounded(fixed + 5, monkeypatch).render_request(request(), max_tokens=4096)
    assert len(cut["input_ids"]) == fixed + 5
    assert warnings == [f"Record 'r': the last {state_tokens - 5} tokens of its state are cut to fit {fixed + 5}"]
    assert strategy.render_request(request(), max_tokens=4096) and len(warnings) == 1
    assert bounded(fixed + 5, monkeypatch).render_request(request(), max_tokens=fixed + 4) and len(warnings) == 1
    assert '<|im_start|>user\nSTATE:\n{"pla\n\nSCHEMA FIELDS:' in strategy.tokenizer.decode(cut["input_ids"])
    shift = state_tokens - 5
    assert cut["read_spans"] == [[(s - shift, e - shift) for s, e in spans] for spans in full["read_spans"]]


def test_seq_len_skips_a_longer_rendering_instead_of_cutting_it(strategy):
    full = strategy.render_request(request(), max_tokens=4096)
    assert strategy.render_request(request(), max_tokens=len(full["input_ids"]) - 1) == full
    assert render_sample(strategy, request(), len(full["input_ids"]) - 1) is None


def test_skips_a_request_whose_questions_do_not_fit(strategy, monkeypatch):
    full = strategy.render_request(request(state=""), max_tokens=4096)
    assert bounded(len(full["input_ids"]), monkeypatch).render_request(request(state=""), 4096) is not None
    assert bounded(len(full["input_ids"]) - 1, monkeypatch).render_request(request(state=""), 4096) is None


def test_skips_what_vllm_refuses(strategy, monkeypatch):
    # One token per byte: 64 questions need more than Clef-Flash's 16384
    long = bounded(65536, monkeypatch)
    many = {f"q{i}": {"type": "noul"} for i in range(65)}
    assert long.render_request(request(questions=many), max_tokens=65536) is None
    assert long.render_request(request(questions=dict(list(many.items())[:64])), max_tokens=65536) is not None

    choices = {chr(ord("a") + i) * 2: None for i in range(26)}
    assert strategy.render_request(request(questions={"c": {"type": "choice", "criteria": choices}}), 4096) is not None
    choices["zz2"] = None
    assert strategy.render_request(request(questions={"c": {"type": "choice", "criteria": choices}}), 4096) is None


def test_instructions_that_are_not_text_are_written_as_vllm_writes_them(strategy):
    questions = {"q": {"type": "noul", "instructions": {"ask": "down?"}}}
    rendered = strategy.render_request(request(questions=questions), max_tokens=4096)
    start, end = rendered["read_spans"][0][0]
    assert strategy.tokenizer.decode(rendered["input_ids"][start:end]) == "{'ask': 'down?'}"


def test_renders_what_vllm_renders(strategy):
    strategies = pytest.importorskip("vllm.entrypoints.openai.decisions.strategies")
    from vllm.entrypoints.openai.decisions.question_types import build_question

    renderer = SimpleNamespace(get_tokenizer=lambda: strategy.tokenizer)
    context = SimpleNamespace(
        online_renderer=SimpleNamespace(renderer=renderer),
        engine_client=SimpleNamespace(model_config=SimpleNamespace(max_model_len=ClefFlashStrategy.MAX_LENGTH)),
    )
    vllm_strategy = strategies.ClefFlashStrategy(context)
    limits = vllm_strategy.limits()
    choices = {"zeta": "Last", "alpha": None, "mid": "Middle"}
    records = [
        (STATE, QUESTIONS),
        ("Checkout fails.", {"c": {"type": "choice", "instructions": {"ask": "who?"}, "criteria": choices}}),
        ({"logs": "x" * 20000}, {"n": {"type": "noul"}, "s": {"type": "score", "criteria": ["a", "b", "c"]}}),
    ]
    for state, questions in records:
        vllm_questions = [
            build_question(qid, q["type"], q.get("instructions"), q.get("criteria"), limits.max_options)
            for qid, q in questions.items()
        ]
        prompt, schema, _ = vllm_strategy.encode(vllm_questions, vllm_strategy.render_state(state))
        rendered = strategy.render_request(request(state, questions), max_tokens=4096)

        assert rendered["input_ids"] == prompt
        n = len(prompt)
        option_spans = iter([(s + n, e + n) for s, e in schema["option_spans"]])
        assert rendered["read_spans"] == [
            [(s + n, e + n), *(next(option_spans) for _ in range(count))]
            for (s, e), count in zip(schema["question_spans"], schema["option_counts"])
        ]
        assert rendered["options"] == [[option for _, option, _ in vllm_strategy._options(q)] for q in vllm_questions]
