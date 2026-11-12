import json

import pytest
from datasets import Dataset

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.sft.system_one.data import EncodedRecord, SystemOneDataset, SystemOneEncoder
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
    record, labels = parse_record(ROW, COLUMNS)
    assert (record.id, record.state) == ("r", "Checkout fails.")
    assert [q.options for q in record.questions] == [["false", "true"], ["technical", "billing"], ["0", "1", "2"]]
    assert [q.descriptions for q in record.questions] == [
        [None, "A service is down."],
        ["Bugs", None],
        ["Can wait", "This week", "Today"],
    ]
    assert [(given.hard, given.soft) for given in labels] == [(1, None), (None, [0.75, 0.25]), (2, None)]
    assert build_targets(record, labels) == [[0.0, 1.0], [0.75, 0.25], [0.0, 0.0, 1.0]]


def test_soft_label_wins_and_unlabelled_questions_have_no_target():
    label = {"outage": False, "team": None}  # a null label is no label
    soft_label = {"outage": {"false": 0.25, "true": 0.75}}
    record, labels = parse_record(row(label=label, soft_label=soft_label), COLUMNS)
    assert labels[0].hard == 0
    assert build_targets(record, labels) == [[0.25, 0.75], None, None]


def test_label_columns_may_be_absent_or_null():
    without_soft_label = {key: value for key, value in row(label=LABEL).items() if key != "soft_label"}
    record, labels = parse_record(without_soft_label, COLUMNS)
    assert build_targets(record, labels) == [[0.0, 1.0], None, [0.0, 0.0, 1.0]]
    record, labels = parse_record(row(), COLUMNS)
    assert build_targets(record, labels) == [None, None, None]


def test_columns_can_be_renamed_and_the_id_falls_back_to_the_row_index():
    renamed = {"__index": 7, "input": json.dumps({"ticket": 7}), "questions": json.dumps(QUESTIONS)}
    renamed["probs"] = json.dumps(SOFT_LABEL)
    columns = SystemOneColumnsConfig(state="input", soft_label="probs")
    record, labels = parse_record(renamed, columns)
    assert (record.id, record.state) == (7, {"ticket": 7})
    assert labels[1].soft == [0.75, 0.25]
    with pytest.raises(ValueError, match="record 7: question 'team'"):
        parse_record({**renamed, "probs": json.dumps({"team": {"technical": 1.0}})}, columns)


@pytest.mark.parametrize(
    "questions, label, soft_label",
    [
        ({}, None, None),  # no questions
        ({"q": "noul"}, None, None),  # a definition that is not an object
        ({"q": {"type": "rank", "instructions": "?"}}, None, None),  # unknown type
        ({"q": {"type": "noul"}}, None, None),  # no instructions
        ({"q": {"type": "noul", "instructions": "?", "criteria": {"yes": "y"}}}, None, None),  # noul criteria keys
        ({"q": {"type": "choice", "instructions": "?", "criteria": {}}}, None, None),  # no options
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


class StubEncoder(SystemOneEncoder):
    """Ten tokens in two segments, read at one span per question or two, with noul and choice options reordered."""

    def __init__(self, tokenizer, reads):
        super().__init__(tokenizer)
        self.reads = reads

    def __call__(self, record, max_tokens):
        return EncodedRecord(
            input_ids=list(range(10, 20)),
            seq_lens=[4, 6],
            reads=self.reads,
            options=[["true", "false"], ["billing", "technical"], ["0", "1", "2"]],
        )


def _dataset(reads) -> SystemOneDataset:
    return SystemOneDataset(Dataset.from_list([ROW]), StubEncoder(None, reads))


def test_process_builds_sample_in_encoder_option_order():
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
            "num_tokens": 10,
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


def test_process_raises_on_overlapping_read_spans():
    with pytest.raises(ValueError, match="record 'r': the encoder broke the EncodedRecord contract"):
        _dataset([[(1, 3)], [(2, 4)], [(6, 7)]])._process(ROW)


def test_process_skips_record_without_labels():
    assert _dataset([[(1, 2)], [(2, 4)], [(6, 7)]])._process(row()) is None
