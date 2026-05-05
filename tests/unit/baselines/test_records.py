from types import SimpleNamespace

import pytest

from prime_rl.baselines.records import flatten_output, parse_answer, token_summary_row


class _Parser:
    def parse_answer(self, completion):
        return completion[0]["content"].split()[-1]


class _StringParser:
    def parse_answer(self, completion):
        return completion.split()[-1]


class _ExplodingParser:
    def parse_answer(self, completion):
        raise RuntimeError("parser exploded")


def test_flatten_output_extracts_answer_tokens_and_timing():
    output = {
        "example_id": "ex1",
        "task": "toy",
        "answer": "4",
        "completion": [{"role": "assistant", "content": "answer 4"}],
        "reward": 1.0,
        "timing": {"generation_ms": 10.0, "scoring_ms": 2.0, "total_ms": 12.0},
        "is_completed": True,
        "is_truncated": False,
        "judge_response": "CORRECT",
        "token_usage": {"input_tokens": 7, "output_tokens": 3},
        "metrics": {"format": 1.0},
    }

    row = flatten_output(
        output=output,
        env=SimpleNamespace(parser=_Parser()),
        run_id="run",
        env_id="env",
        protocol="single_shot",
        dataset="toy",
        model="model",
        seed=3,
        trial_index=0,
        success_threshold=1.0,
    )

    assert row["parsed_answer"] == "4"
    assert row["response"] == "answer 4"
    assert row["correct"] is True
    assert row["input_tokens"] == 7
    assert row["output_tokens"] == 3
    assert row["total_ms"] == 12.0
    assert row["judge_response"] == "CORRECT"


def test_flatten_output_preserves_string_completion_response():
    output = {
        "example_id": "ex1",
        "task": "toy",
        "answer": "4",
        "completion": " answer 4 ",
        "reward": 1.0,
    }

    row = flatten_output(
        output=output,
        env=SimpleNamespace(parser=_StringParser()),
        run_id="run",
        env_id="env",
        protocol="single_shot",
        dataset="toy",
        model="model",
        seed=3,
        trial_index=0,
        success_threshold=1.0,
    )

    assert row["parsed_answer"] == "4"
    assert row["response"] == "answer 4"


def test_parse_answer_propagates_parser_errors():
    with pytest.raises(RuntimeError, match="parser exploded"):
        parse_answer(SimpleNamespace(parser=_ExplodingParser()), [{"role": "assistant", "content": "answer 4"}])


def test_flatten_output_uses_judge_decision_for_hard_correct_with_soft_reward():
    output = {
        "example_id": "ex1",
        "task": "toy",
        "answer": "4",
        "completion": [{"role": "assistant", "content": "answer 4"}],
        "reward": 2 / 3,
        "timing": {"generation_ms": 10.0, "scoring_ms": 2.0, "total_ms": 12.0},
        "is_completed": True,
        "is_truncated": False,
        "token_usage": {"input_tokens": 7, "output_tokens": 3},
        "judge_decision_last": {
            "p_correct": 2 / 3,
            "reward": 2 / 3,
            "hard_response": "CORRECT",
            "support": {"policy": {"threshold": 0.5}},
        },
    }

    row = flatten_output(
        output=output,
        env=SimpleNamespace(parser=_Parser()),
        run_id="run",
        env_id="env",
        protocol="single_shot",
        dataset="toy",
        model="model",
        seed=3,
        trial_index=0,
        success_threshold=1.0,
    )

    assert row["correct"] is True
    assert row["posterior_correct"] == 2 / 3


def test_token_summary_row_matches_nanodebate_shape():
    rows = [
        {"input_tokens": 2, "output_tokens": 5},
        {"input_tokens": 3, "output_tokens": 7},
    ]

    summary = token_summary_row(rows, model="org/model", protocol="single_shot", dataset="ds", seed=9)

    assert summary["model_name"] == "org/model"
    assert summary["model_short"] == "model"
    assert summary["input_tokens"] == 5
    assert summary["output_tokens"] == 12
    assert summary["total_tokens"] == 17
