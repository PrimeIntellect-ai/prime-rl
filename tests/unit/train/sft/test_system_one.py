import json

import pytest

from prime_rl.configs.sft import SystemOneColumnsConfig
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


@pytest.mark.parametrize("num_options, too_far", [(30, 0.028), (40, 0.031)])
def test_soft_label_sum_tolerance_grows_with_the_options(num_options, too_far):
    # A uniform label rounded to two decimals: 0.03 per option, a sum of 0.9 or 1.2, within 0.005 per option
    options = [str(index) for index in range(num_options)]
    questions = {"q": {"type": "choice", "criteria": dict.fromkeys(options)}}
    _, labels = parse_record(row(questions, soft_label={"q": dict.fromkeys(options, 0.03)}), COLUMNS)
    assert labels[0].soft == pytest.approx([1 / num_options] * num_options)
    with pytest.raises(ValueError, match=f"within {0.005 * num_options:g}"):
        parse_record(row(questions, soft_label={"q": dict.fromkeys(options, too_far)}), COLUMNS)


def test_score_labels_may_be_whole_floats():
    _, labels = parse_record(row(label={"urgency": 2.0}), COLUMNS)
    assert labels[2].hard == 2


@pytest.mark.parametrize(
    "column, value, error",
    [
        ("state", "[" * 100_000 + "]" * 100_000, RecursionError),
        ("soft_label", json.dumps({"team": {"technical": 10**400, "billing": 0}}), OverflowError),
    ],
)
def test_other_parse_errors_name_the_record(column, value, error):
    with pytest.raises(error) as raised:
        parse_record({**ROW, column: value}, COLUMNS)
    assert raised.value.__notes__ == ["record 'r'"]


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


def test_a_null_id_falls_back_like_an_absent_one():
    request, _ = parse_record({**ROW, "id": None, "__split": "train", "__index": 7}, COLUMNS)
    assert request.id == "train/7"


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
        (QUESTIONS, {"urgency": 1.5}, None),  # a level that is not whole
        (QUESTIONS, {"urgency": 3.0}, None),  # a whole level out of range
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
