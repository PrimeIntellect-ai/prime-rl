import json
import math

import pytest
import torch
from datasets import Dataset
from torch import nn

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.sft.loss import LossInputs
from prime_rl.trainer.sft.system_one.data import (
    RenderedRequest,
    SystemOneDataset,
    SystemOneStrategy,
    check_rendered_request,
    system_one_strategy_cls,
)
from prime_rl.trainer.sft.system_one.loss import decision_loss
from prime_rl.trainer.sft.system_one.record import build_targets, parse_record

COLUMNS = SystemOneColumnsConfig()
QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Is a service down?", "criteria": {"true": "A service is down."}},
    "team": {"type": "choice", "instructions": "Who handles it?", "criteria": {"technical": "Bugs", "billing": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
}
LABEL = {"outage": True, "urgency": 2}
SOFT_LABEL = {"team": {"billing": 0.25, "technical": 0.75}}


def row(questions=QUESTIONS, label=None, soft_label=None):
    return {
        "id": "r",
        "state": json.dumps("Checkout fails."),
        "questions": json.dumps(questions),
        "label": None if label is None else json.dumps(label),
        "soft_label": None if soft_label is None else json.dumps(soft_label),
    }


ROW = row(label=LABEL, soft_label=SOFT_LABEL)


def test_parse_record_and_build_targets():
    request, labels = parse_record(ROW, COLUMNS)
    assert (request.id, request.state) == ("r", "Checkout fails.")
    assert [q.options for q in request.questions] == [["false", "true"], ["technical", "billing"], ["0", "1", "2"]]
    assert [q.descriptions for q in request.questions] == [
        [None, "A service is down."],
        ["Bugs", None],
        ["Can wait", "This week", "Today"],
    ]
    assert [(given.hard, given.soft) for given in labels] == [(1, None), (None, [0.75, 0.25]), (2, None)]
    assert build_targets(request, labels) == [[0.0, 1.0], [0.75, 0.25], [0.0, 0.0, 1.0]]


def test_soft_label_wins_and_unlabelled_questions_have_no_target():
    label = {"outage": False, "team": None}  # a null label is no label
    soft_label = {"outage": {"false": 0.25, "true": 0.75}, "urgency": {"0": 0.33, "1": 0.33, "2": 0.33}}
    request, labels = parse_record(row(label=label, soft_label=soft_label), COLUMNS)
    assert labels[0].hard == 0
    targets = build_targets(request, labels)
    assert targets[:2] == [[0.25, 0.75], None]
    assert targets[2] == pytest.approx([1 / 3] * 3)  # a sum within 0.05 of 1 is rescaled


def test_soft_label_rounded_to_two_decimals_is_rescaled():
    # [0.015] * 7 + [0.125, 0.355, 0.415] rounded to two decimals: a sum of 0.95
    probabilities = [0.01] * 7 + [0.12, 0.35, 0.41]
    options = [str(index) for index in range(10)]
    questions = {"q": {"type": "choice", "criteria": dict.fromkeys(options)}}
    _, labels = parse_record(row(questions, soft_label={"q": dict(zip(options, probabilities))}), COLUMNS)
    assert labels[0].soft == pytest.approx([p / 0.95 for p in probabilities])


@pytest.mark.parametrize(
    "criteria, expected",
    [({"true": None}, None), ({"true": "Down.", "false": None}, {"true": "Down."}), (None, None)],
)
def test_null_noul_descriptions_are_absent(criteria, expected):
    question = {"type": "noul", "instructions": "Is a service down?", "criteria": criteria}
    request, _ = parse_record(row({"outage": question}), COLUMNS)
    assert request.questions[0].criteria == expected


def test_instructions_are_optional():
    questions = {"absent": {"type": "noul"}, "null": {"type": "noul", "instructions": None}}
    request, _ = parse_record(row(questions), COLUMNS)
    assert [question.instructions for question in request.questions] == ["", ""]


def test_label_columns_may_be_absent_or_null():
    without_soft_label = {key: value for key, value in row(label=LABEL).items() if key != "soft_label"}
    request, labels = parse_record(without_soft_label, COLUMNS)
    assert build_targets(request, labels) == [[0.0, 1.0], None, [0.0, 0.0, 1.0]]
    request, labels = parse_record(row(), COLUMNS)
    assert build_targets(request, labels) == [None, None, None]


def test_columns_can_be_renamed_and_the_id_falls_back_to_the_split_and_row_index():
    renamed = {"__split": "train", "__index": 7, "input": json.dumps({"ticket": 7}), "questions": json.dumps(QUESTIONS)}
    renamed["probs"] = json.dumps(SOFT_LABEL)
    columns = SystemOneColumnsConfig(state="input", soft_label="probs")
    request, labels = parse_record(renamed, columns)
    assert (request.id, request.state) == ("train/7", {"ticket": 7})
    assert labels[1].soft == [0.75, 0.25]
    with pytest.raises(ValueError, match="record 'train/7': question 'team'"):
        parse_record({**renamed, "probs": json.dumps({"team": {"technical": 1.0}})}, columns)


@pytest.mark.parametrize(
    "questions, label, soft_label",
    [
        ({}, None, None),  # no questions
        ({"q": "noul"}, None, None),  # a definition that is not an object
        ({"q": {"type": "rank", "instructions": "?"}}, None, None),  # unknown type
        ({"q": {"type": "noul", "instructions": "?", "criteria": {"yes": "y"}}}, None, None),  # noul criteria keys
        ({"q": {"type": "choice", "instructions": "?", "criteria": {}}}, None, None),  # no options
        ({"q": {"type": "choice", "instructions": "?", "criteria": {"": None}}}, None, None),  # an empty option id
        ({"q": {"type": "score", "instructions": "?", "criteria": ["low", ""]}}, None, None),  # an empty level
        ({"q": {"type": "score", "instructions": "?", "criteria": [1, "1"]}}, None, None),  # the same level twice
        ({"": {"type": "noul"}}, None, None),  # an empty question id
        ({"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}, None, None),  # one level
        ({"q": {"type": "score", "instructions": "?", "criteria": ["level"] * 11}}, None, None),  # too many levels
        (QUESTIONS, {"team": "sales"}, None),  # a label that is not an option
        (QUESTIONS, {"team": True}, None),  # true/false is a noul label only
        (QUESTIONS, {"urgency": 3}, None),  # a level out of range
        (QUESTIONS, {"urgency": True}, None),  # a bool is not a level
        (QUESTIONS, {"other": True}, None),  # an unknown question
        (QUESTIONS, ["outage"], None),  # labels that are not an object
        (QUESTIONS, None, {"other": {"false": 1.0, "true": 0.0}}),  # an unknown question
        (QUESTIONS, None, {"team": {"technical": 1.0}}),  # a missing option
        (QUESTIONS, None, {"team": {"technical": 0.7, "billing": 0.7}}),  # not a distribution
        (QUESTIONS, None, {"team": {"technical": 0.5, "billing": 0.44}}),  # a sum more than 0.05 from 1
        (QUESTIONS, None, {"team": {"technical": 1.5, "billing": -0.5}}),  # a negative probability
        (QUESTIONS, None, {"team": {"technical": "0.5", "billing": 0.5}}),  # not numbers
    ],
)
def test_rejects_malformed_records(questions, label, soft_label):
    with pytest.raises(ValueError, match="record 'r'"):
        parse_record(row(questions, label, soft_label), COLUMNS)


@pytest.mark.parametrize(
    "column, value",
    [
        ("state", "Checkout fails."),  # plain text is not JSON text
        ("state", {"ticket": 7}),  # an object that is not encoded
        ("state", None),
        ("questions", None),
        ("label", "{"),
        ("soft_label", SOFT_LABEL),
    ],
)
def test_rejects_columns_that_are_not_json_text(column, value):
    with pytest.raises(ValueError, match=f"record 'r': column '{column}'"):
        parse_record({**row(label=LABEL), column: value}, COLUMNS)


class StubStrategy(SystemOneStrategy):
    """Ten tokens in two segments, read at one span per question or two, with noul and choice options reordered."""

    def __init__(self, tokenizer, config, read_spans=None):
        super().__init__(tokenizer, config)
        self.read_spans = read_spans

    def render_request(self, request, max_tokens):
        return RenderedRequest(
            input_ids=list(range(10, 20)),
            seq_lens=[4, 6],
            read_spans=self.read_spans,
            options=[["true", "false"], ["billing", "technical"], ["0", "1", "2"]],
        )


def _dataset(read_spans) -> SystemOneDataset:
    return SystemOneDataset(Dataset.from_list([ROW]), StubStrategy(None, None, read_spans))


def test_process_builds_sample_in_strategy_option_order():
    sample = _dataset([[(1, 2)], [(2, 4)], [(6, 7), (8, 10)]])._process(ROW)

    assert sample == {
        "input_ids": list(range(10, 20)),
        "position_ids": [0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
        "seq_lens": [4, 6],
        "loss_mask": [False, True, True, True, False, False, True, False, True, True],
        "target_ids": [0] * 10,
        "mm_kwargs": None,
        "mm_token_type_ids": None,
        "sample_data": {
            "id": "r",
            "questions": [
                {"id": "outage", "type": "noul", "options": ["true", "false"], "reads": [[1, 2]], "target": [1.0, 0.0]},
                {
                    "id": "team",
                    "type": "choice",
                    "options": ["billing", "technical"],
                    "reads": [[2, 4]],
                    "target": [0.25, 0.75],
                },
                {
                    "id": "urgency",
                    "type": "score",
                    "options": ["0", "1", "2"],
                    "reads": [[6, 7], [8, 10]],
                    "target": [0.0, 0.0, 1.0],
                },
            ],
        },
    }


@pytest.mark.parametrize(
    "input_ids, seq_lens, read_spans",
    [
        (list(range(10)), [4, 6], [[(1, 3)], [(2, 4)], [(6, 7)]]),  # overlapping spans
        (list(range(10)), [4, 6], [[(1, 2)], [(3, 5)], [(6, 7)]]),  # a span across two segments
        ([True, *range(9)], [4, 6], [[(1, 2)], [(2, 4)], [(6, 7)]]),  # a bool token
        (list(range(10)), [True, 3, 6], [[(1, 2)], [(2, 4)], [(6, 7)]]),  # a bool segment length
        (list(range(10)), [4, 6], [[(True, 2)], [(2, 4)], [(6, 7)]]),  # a bool span start
    ],
)
def test_check_rendered_request_rejects_broken_renderings(input_ids, seq_lens, read_spans):
    request, _ = parse_record(ROW, COLUMNS)
    rendered = RenderedRequest(input_ids=input_ids, seq_lens=seq_lens, read_spans=read_spans)
    with pytest.raises(ValueError, match="record 'r': the strategy broke the RenderedRequest contract"):
        check_rendered_request(rendered, request)


def test_process_skips_record_without_labels():
    assert _dataset([[(1, 2)], [(2, 4)], [(6, 7)]])._process(row()) is None


class SkippingStrategy(SystemOneStrategy):
    def render_request(self, request, max_tokens):
        return None


def test_raises_when_every_record_is_skipped():
    dataset = SystemOneDataset(Dataset.from_list([ROW] * 3), SkippingStrategy(None, None))
    with pytest.raises(RuntimeError, match="skipped 7 rows in a row"):
        next(iter(dataset))


class StubDecisionModel:
    """A readout that returns the logits of the given distributions, in float64 so that the loss has to cast them, and
    keeps the batch it was called with."""

    def __init__(self, probs):
        self.logits = [torch.tensor(p, dtype=torch.float64).log().requires_grad_() for p in probs]

    @classmethod
    def system_one_strategy(cls):
        return StubStrategy

    def decision_logits(self, hidden_states, batch):
        self.batch = batch
        return self.logits


def test_system_one_strategy_cls_requires_a_declared_strategy():
    assert system_one_strategy_cls(StubDecisionModel) is StubStrategy
    with pytest.raises(ValueError, match="object does not"):
        system_one_strategy_cls(object)


def _decision_inputs(model) -> LossInputs:
    """A row of two records. In the second, ``outage`` has no label and the target of ``team`` is tied."""
    second = {**row(label={"urgency": 0}, soft_label={"team": {"billing": 0.5, "technical": 0.5}}), "id": "s"}
    dataset = _dataset([[(1, 2)], [(2, 4)], [(6, 7), (8, 10)]])
    records = [dataset._process(example)["sample_data"] for example in (ROW, second)]
    return LossInputs(model=model, hidden_states=torch.zeros(1, 20, 4), batch={"sample_data": records})


def test_decision_loss_sums_over_labelled_questions():
    # Targets: [1, 0], [0.25, 0.75], [0, 0, 1] in record r; none, [0.5, 0.5], [1, 0, 0] in record s.
    model = StubDecisionModel([[0.8, 0.2], [0.75, 0.25], [0.25, 0.25, 0.5], [0.9, 0.1], [0.2, 0.8], [0.5, 0.25, 0.25]])
    outputs = decision_loss(_decision_inputs(model), brier_weight=0.1)

    ce = sum(
        [
            -math.log(0.8),
            -(0.25 * math.log(0.75) + 0.75 * math.log(0.25)),
            -math.log(0.5),
            -(0.5 * math.log(0.2) + 0.5 * math.log(0.8)),
            -math.log(0.5),
        ]
    )
    brier = 0.08 + 0.5 + 0.375 + 0.18 + 0.375
    assert outputs.count == 5 and isinstance(outputs.count, int)
    torch.testing.assert_close(outputs.loss, torch.tensor(ce + 0.1 * brier))
    # Accuracy: team is wrong in r, and right in s at either option of its tied target.
    expected = {"ce": torch.tensor(ce), "brier": torch.tensor(brier), "accuracy": torch.tensor(4.0)}
    assert outputs.metrics.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(outputs.metrics[name], value)

    # The readout gets the records without the targets.
    questions = [question for record in model.batch["sample_data"] for question in record["questions"]]
    assert [record["id"] for record in model.batch["sample_data"]] == ["r", "s"]
    assert questions[1] == {"id": "team", "type": "choice", "options": ["billing", "technical"], "reads": [[2, 4]]}
    assert all("target" not in question for question in questions)


def test_decision_loss_rejects_broken_readouts():
    probs = [[0.8, 0.2], [0.75, 0.25], [0.25, 0.25, 0.5], [0.9, 0.1], [0.2, 0.8], [0.5, 0.25, 0.25]]
    with pytest.raises(ValueError, match=r"5 logit vectors for the 6 questions of records \['r', 's'\]"):
        decision_loss(_decision_inputs(StubDecisionModel(probs[:-1])))
    with pytest.raises(ValueError, match=r"record 's', question 'team': .* shape \(3,\) for 2 options"):
        decision_loss(_decision_inputs(StubDecisionModel([*probs[:4], [0.2, 0.3, 0.5], probs[5]])))
    with pytest.raises(TypeError, match=r"decision_logits.*Linear.*\[model.implementation\]"):
        decision_loss(_decision_inputs(nn.Linear(4, 2)))


def test_decision_loss_without_labelled_questions_backpropagates_zero():
    # A probability of 0 is a logit of -inf.
    model = StubDecisionModel([[1.0, 0.0], [0.75, 0.25], [0.0, 0.0, 1.0], [0.9, 0.1], [0.2, 0.8], [0.5, 0.25, 0.25]])
    inputs = _decision_inputs(model)
    for record in inputs.batch["sample_data"]:
        for question in record["questions"]:
            question["target"] = None
    outputs = decision_loss(inputs)
    outputs.loss.backward()

    assert outputs.count == 0 and outputs.loss.item() == 0.0
    assert all(torch.equal(logits.grad, torch.zeros_like(logits)) for logits in model.logits)


def test_decision_loss_is_finite_at_minus_inf_on_options_with_target_0():
    # Probability 0 on false in r's outage and on 0 and 1 in r's urgency, whose targets are 0 there.
    model = StubDecisionModel([[1.0, 0.0], [0.75, 0.25], [0.0, 0.0, 1.0], [0.9, 0.1], [0.2, 0.8], [0.5, 0.25, 0.25]])
    outputs = decision_loss(_decision_inputs(model), brier_weight=0.1)
    outputs.loss.backward()

    ce = -(0.25 * math.log(0.75) + 0.75 * math.log(0.25)) - (0.5 * math.log(0.2) + 0.5 * math.log(0.8)) - math.log(0.5)
    brier = 0.5 + 0.18 + 0.375
    torch.testing.assert_close(outputs.loss, torch.tensor(ce + 0.1 * brier))
    assert all(torch.isfinite(logits.grad).all() for logits in model.logits)
