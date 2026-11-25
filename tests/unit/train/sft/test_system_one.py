import json
import math
import os
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from torch import nn
from transformers import PretrainedConfig

import prime_rl.trainer.parallel_dims as parallel_dims_module
from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.data import CatDataset
from prime_rl.trainer.model import ExtraConfigSelection, forward, reshard_module, setup_fsdp
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.sft.data import SFTDataset
from prime_rl.trainer.sft.system_one import strategies
from prime_rl.trainer.sft.system_one.data import (
    ReadQuestion,
    RenderedRequest,
    SystemOneSource,
    SystemOneStrategy,
    check_records,
    check_rendered_request,
)
from prime_rl.trainer.sft.system_one.lm_head import lm_head_logits
from prime_rl.trainer.sft.system_one.loss import (
    DecisionLossInputs,
    SystemOneBatch,
    build_decision_loss_inputs,
    build_prefix_lens,
    decision_loss,
    system_one_collate,
)
from prime_rl.trainer.sft.system_one.record import parse_record
from prime_rl.trainer.sft.system_one.strategies import get_system_one_strategy_cls

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


def test_parse_record():
    request, targets = parse_record(ROW, COLUMNS)
    assert (request.id, request.state) == ("r", "Checkout fails.")
    assert [q.options for q in request.questions] == [["false", "true"], ["technical", "billing"], ["0", "1", "2"]]
    assert targets == [[0.0, 1.0], [0.75, 0.25], [0.0, 0.0, 1.0]]


def test_soft_label_wins_and_unlabelled_questions_have_no_target():
    label = {"outage": False, "team": None}  # a null label is no label
    soft_label = {"outage": {"false": 0.25, "true": 0.75}, "urgency": {"0": 0.33, "1": 0.33, "2": 0.33}}
    _, targets = parse_record(row(label=label, soft_label=soft_label), COLUMNS)
    assert targets[:2] == [[0.25, 0.75], None]
    assert targets[2] == pytest.approx([1 / 3] * 3)  # a sum within 0.05 of 1 is rescaled


def test_soft_label_rounded_to_two_decimals_is_rescaled():
    # [0.015] * 7 + [0.125, 0.355, 0.415] rounded to two decimals: a sum of 0.95
    probabilities = [0.01] * 7 + [0.12, 0.35, 0.41]
    options = [str(index) for index in range(10)]
    questions = {"q": {"type": "choice", "criteria": dict.fromkeys(options)}}
    _, targets = parse_record(row(questions, soft_label={"q": dict(zip(options, probabilities))}), COLUMNS)
    assert targets[0] == pytest.approx([p / 0.95 for p in probabilities])


@pytest.mark.parametrize("num_options, too_far", [(30, 0.028), (40, 0.031)])
def test_soft_label_sum_tolerance_grows_with_the_options(num_options, too_far):
    # A uniform label rounded to two decimals: 0.03 per option, a sum of 0.9 or 1.2, within 0.005 per option
    options = [str(index) for index in range(num_options)]
    questions = {"q": {"type": "choice", "criteria": dict.fromkeys(options)}}
    _, targets = parse_record(row(questions, soft_label={"q": dict.fromkeys(options, 0.03)}), COLUMNS)
    assert targets[0] == pytest.approx([1 / num_options] * num_options)
    with pytest.raises(ValueError, match=f"within {0.005 * num_options:g}"):
        parse_record(row(questions, soft_label={"q": dict.fromkeys(options, too_far)}), COLUMNS)


def test_score_labels_may_be_whole_floats():
    _, targets = parse_record(row(label={"urgency": 2.0}), COLUMNS)
    assert targets[2] == [0.0, 0.0, 1.0]


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
    assert parse_record(without_soft_label, COLUMNS)[1] == [[0.0, 1.0], None, [0.0, 0.0, 1.0]]
    assert parse_record(row(), COLUMNS)[1] == [None, None, None]


def test_columns_can_be_renamed_and_the_id_falls_back_to_the_source_split_and_row_index():
    renamed = {"__index": 7, "input": json.dumps({"ticket": 7}), "questions": json.dumps(QUESTIONS)}
    renamed["probs"] = json.dumps(SOFT_LABEL)
    columns = SystemOneColumnsConfig(state="input", soft_label="probs")
    request, targets = parse_record(renamed, columns, "records", "train")
    assert (request.id, request.state) == ("records/train/7", {"ticket": 7})
    assert targets[1] == [0.75, 0.25]
    with pytest.raises(ValueError, match="record 'records/train/7': question 'team'"):
        parse_record({**renamed, "probs": json.dumps({"team": {"technical": 1.0}})}, columns, "records", "train")


def test_a_null_id_falls_back_like_an_absent_one():
    request, _ = parse_record({**ROW, "id": None, "__index": 7}, COLUMNS, "records", "train")
    assert request.id == "records/train/7"


MANY_OPTIONS = {"q": {"type": "choice", "criteria": dict.fromkeys(map(str, range(200)))}}


@pytest.mark.parametrize(
    "questions, label, soft_label",
    [
        ({}, None, None),  # no questions
        ({"q": "noul"}, None, None),  # a definition that is not an object
        ({"q": {"type": "rank", "instructions": "?"}}, None, None),  # unknown type
        ({"q": {"type": "noul", "instruction": "?"}}, None, None),  # an unknown key
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
        (MANY_OPTIONS, None, {"q": dict.fromkeys(MANY_OPTIONS["q"]["criteria"], 0)}),  # all zeros, within 1 of 1
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


CONFIG = PretrainedConfig(vocab_size=32)


class StubStrategy(SystemOneStrategy):
    """Ten tokens in two segments, one or two read spans per question, noul and choice options reordered."""

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

    def option_logits(self, model, hidden_states, input_ids, questions):
        model.seen.append((hidden_states, input_ids, questions))
        return model.logits


READ_SPANS = [[(1, 2)], [(2, 4)], [(6, 7), (8, 10)]]


def _source(read_spans=READ_SPANS) -> SystemOneSource:
    return SystemOneSource(Dataset.from_list([ROW]), StubStrategy(None, CONFIG, read_spans))


def test_process_builds_sample_in_strategy_option_order():
    sample = _source()._process(ROW)

    assert {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in sample.items()} == {
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
                {"type": "noul", "options": ["true", "false"], "reads": [[1, 2]], "target": [1.0, 0.0]},
                {"type": "choice", "options": ["billing", "technical"], "reads": [[2, 4]], "target": [0.25, 0.75]},
                {
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
        ([32, *range(9)], [4, 6], [[(1, 2)], [(2, 4)], [(6, 7)]]),  # a token id past the vocabulary
        ([-1, *range(9)], [4, 6], [[(1, 2)], [(2, 4)], [(6, 7)]]),  # a negative token id
    ],
)
def test_check_rendered_request_rejects_broken_renderings(input_ids, seq_lens, read_spans):
    request, _ = parse_record(ROW, COLUMNS)
    rendered = RenderedRequest(input_ids=input_ids, seq_lens=seq_lens, read_spans=read_spans)
    with pytest.raises(ValueError, match="record 'r': the strategy broke the RenderedRequest contract"):
        check_rendered_request(rendered, request, vocab_size=32)


class TwoBlockStrategy(StubStrategy):
    """StubStrategy with a prefix in each segment; the second segment is all prefix."""

    prefix_lens = [1, 6]

    def render_request(self, request, max_tokens):
        return {**super().render_request(request, max_tokens), "prefix_lens": self.prefix_lens}


@pytest.mark.parametrize("prefix_lens", [[1], [0, 6], [5, 6], [True, 6], (1, 6), [1, 6, 1]])
def test_check_rendered_request_rejects_broken_prefix_lens(monkeypatch, prefix_lens):
    monkeypatch.setattr(TwoBlockStrategy, "prefix_lens", prefix_lens)
    source = SystemOneSource(Dataset.from_list([ROW]), TwoBlockStrategy(None, CONFIG, READ_SPANS))
    with pytest.raises(ValueError, match="prefix_lens must hold an int from 1 to its length per segment of"):
        source._process(ROW)


def test_process_skips_record_without_labels():
    assert _source()._process(row()) is None


class SkippingStrategy(StubStrategy):
    def render_request(self, request, max_tokens):
        return None


def test_raises_when_every_record_is_skipped():
    source = SystemOneSource(Dataset.from_list([ROW] * 3), SkippingStrategy(None, CONFIG))
    with pytest.raises(RuntimeError, match="skipped 7 rows in a row"):
        next(iter(SFTDataset([source])))


def test_check_records_parses_the_data_rank_share_of_the_rows():
    dataset = Dataset.from_list([ROW, {**ROW, "id": "bad", "label": "{"}, ROW])
    with pytest.raises(ValueError, match="record 'bad': column 'label'"):
        check_records(dataset, COLUMNS)
    check_records(dataset, COLUMNS, rank=0, world_size=2)  # rows 0 and 2
    # A record without an id is named by its source, split and row index
    unnamed = Dataset.from_list([{**ROW, "id": None, "label": "{", "__index": 0}])
    with pytest.raises(ValueError, match="record 'records/validation/0': column 'label'"):
        check_records(unnamed, COLUMNS, source="records", split="validation")


def test_get_system_one_strategy_cls_looks_up_the_architecture(monkeypatch):
    table = {"StubDecisionModel": StubStrategy, "SkippingDecisionModel": SkippingStrategy}
    monkeypatch.setattr(strategies, "SYSTEM_ONE_STRATEGIES", table)
    assert get_system_one_strategy_cls(PretrainedConfig(architectures=["StubDecisionModel"])) is StubStrategy
    # Only the first architecture counts
    config = PretrainedConfig(architectures=["SkippingDecisionModel", "StubDecisionModel"])
    assert get_system_one_strategy_cls(config) is SkippingStrategy
    with pytest.raises(
        ValueError,
        match=r"architectures \['Qwen3ForCausalLM', 'StubDecisionModel'\] do not start with one. "
        r"Strategies are registered for \['SkippingDecisionModel', 'StubDecisionModel'\]",
    ):
        get_system_one_strategy_cls(PretrainedConfig(architectures=["Qwen3ForCausalLM", "StubDecisionModel"]))
    with pytest.raises(ValueError, match=r"architectures \[\] do not start with one"):
        get_system_one_strategy_cls(PretrainedConfig())
    # A class selected by an extra config file or key is looked up by its name
    stock = PretrainedConfig(architectures=["Qwen3ForCausalLM"])
    selection = ExtraConfigSelection(type("StubDecisionModel", (), {}), {})
    assert get_system_one_strategy_cls(stock, selection) is StubStrategy


class StubDecisionModel:
    """Its logits are the log of the given distributions, flat, in float64 so the loss has to cast them."""

    def __init__(self, probs=()):
        self.logits = torch.tensor([p for dist in probs for p in dist], dtype=torch.float64).log().requires_grad_()
        self.seen = []


STRATEGY = StubStrategy(None, CONFIG)
# Record s: outage has no label, the target of team is tied, and urgency's is [1, 0, 0].
SECOND = {**row(label={"urgency": 0}, soft_label={"team": {"billing": 0.5, "technical": 0.5}}), "id": "s"}
PROBS = [[0.8, 0.2], [0.75, 0.25], [0.25, 0.25, 0.5], [0.9, 0.1], [0.2, 0.8], [0.5, 0.25, 0.25]]
INPUT_IDS = torch.arange(24)[None]


def _micro_batch(source: SystemOneSource, rows: list[dict]) -> SystemOneBatch:
    """The rows' records packed into one micro-batch of seq_len 12 and micro_batch_size 2, padded to 24 tokens."""
    samples = [source._process(example) for example in rows]
    (packed,) = CatDataset(samples, seq_len=24, pad_to_multiple_of=24)
    return system_one_collate([packed])


def _decision_loss_inputs(rows: list[dict] = [ROW, SECOND]) -> DecisionLossInputs:
    return _micro_batch(_source(), rows)["decision_loss_inputs"].to("cpu")


def test_a_row_has_a_prefix_per_segment_and_its_padding_is_one_block():
    source = SystemOneSource(Dataset.from_list([ROW]), TwoBlockStrategy(None, CONFIG, READ_SPANS))
    batch = _micro_batch(source, [ROW, SECOND])
    assert batch["seq_lens"].tolist() == [4, 6, 4, 6, 4]
    assert batch["prefix_lens"].tolist() == [1, 6, 1, 6, 4]
    # Without prefixes, attention is the model's own
    assert _micro_batch(_source(), [ROW, SECOND])["prefix_lens"] is None
    with pytest.raises(ValueError, match="for every request or for none"):
        build_prefix_lens([4, 6, 4, 6], [{"prefix_lens": [1, 6]}, {}])


def test_build_decision_loss_inputs_maps_reads_into_the_row():
    batch = _micro_batch(_source(), [ROW, SECOND])
    assert batch["seq_lens"].tolist() == [4, 6, 4, 6, 4]  # the last segment is the padding
    assert batch["sample_lens"].tolist() == [10, 10]
    assert batch["sample_data"] is None  # replaced by the decision-loss inputs
    inputs = batch["decision_loss_inputs"]

    reads = [((1, 2),), ((2, 4),), ((6, 7), (8, 10)), ((11, 12),), ((12, 14),), ((16, 17), (18, 20))]
    assert [question.reads for question in inputs.questions] == reads
    read_positions = [position for spans in reads for start, end in spans for position in range(start, end)]
    assert read_positions == batch["loss_mask"][0].nonzero().squeeze(-1).tolist()
    assert [question.record for question in inputs.questions] == [(0, 10)] * 3 + [(10, 20)] * 3
    assert inputs.questions[4] == ReadQuestion("choice", ("billing", "technical"), ((12, 14),), (10, 20))
    # Questions of 2, 2, 3, 2, 2 and 3 options in a grid 3 wide
    assert inputs.option_slots.tolist() == [0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17]
    assert inputs.labelled.tolist() == [0, 1, 2, 4, 5]
    expected = [[1, 0, 0], [0.25, 0.75, 0], [0, 0, 1], [0.5, 0.5, 0], [1, 0, 0]]
    torch.testing.assert_close(inputs.targets, torch.tensor(expected, dtype=torch.float32))
    assert inputs.num_labelled == 5


def test_decision_loss_sums_over_labelled_questions():
    # Targets: [1, 0], [0.25, 0.75], [0, 0, 1] in record r; none, [0.5, 0.5], [1, 0, 0] in record s.
    model = StubDecisionModel(PROBS)
    hidden_states = torch.randn(1, 24, 4)
    inputs = _decision_loss_inputs()
    out = decision_loss(model, hidden_states, strategy=STRATEGY, inputs=inputs, input_ids=INPUT_IDS, brier_weight=0.1)

    ce = [
        -math.log(0.8),
        -(0.25 * math.log(0.75) + 0.75 * math.log(0.25)),
        -math.log(0.5),
        -(0.5 * math.log(0.2) + 0.5 * math.log(0.8)),
        -math.log(0.5),
    ]
    brier = [0.08, 0.5, 0.375, 0.18, 0.375]
    torch.testing.assert_close(out["loss"], torch.tensor(sum(ce) + 0.1 * sum(brier)))
    # One value per labelled question. Accuracy: team is wrong in r, and right in s at either option of its tied target.
    expected = {"ce": ce, "brier": brier, "accuracy": [1.0, 0.0, 1.0, 1.0, 1.0]}
    for name, values in expected.items():
        torch.testing.assert_close(out[name], torch.tensor(values))

    # One readout for the micro-batch, on its packed row, without the targets.
    ((states, input_ids, questions),) = model.seen
    torch.testing.assert_close(states, hidden_states[0])
    assert torch.equal(input_ids, INPUT_IDS[0])
    assert questions == inputs.questions


@pytest.mark.parametrize(
    "logits, got",
    [
        (torch.zeros(13), r"torch.float32 of shape \(13,\)"),
        (torch.zeros(2, 7), r"torch.float32 of shape \(2, 7\)"),
        (torch.zeros(14, dtype=torch.long), r"torch.int64 of shape \(14,\)"),
        ([torch.zeros(2)] * 7, "list"),
    ],
)
def test_decision_loss_rejects_broken_readouts(logits, got):
    model = StubDecisionModel()
    model.logits = logits
    with pytest.raises(
        ValueError,
        match=r"StubStrategy.option_logits must return a 1-D float tensor with the 14 option logits of the "
        rf"micro-batch's 6 questions, got {got}",
    ):
        inputs = _decision_loss_inputs()
        decision_loss(
            model, torch.zeros(1, 24, 4), strategy=STRATEGY, inputs=inputs, input_ids=INPUT_IDS, brier_weight=0
        )


def test_decision_loss_without_labelled_questions_backpropagates_zero():
    samples = [_source()._process(example) for example in (ROW, SECOND)]
    for sample in samples:
        for question in sample["sample_data"]["questions"]:
            question["target"] = None
    sample_lens = [len(sample["input_ids"]) for sample in samples]
    inputs = build_decision_loss_inputs(sample_lens, [sample["sample_data"] for sample in samples])
    # A probability of 0 is a logit of -inf.
    model = StubDecisionModel([[1.0, 0.0], [0.75, 0.25], [0.0, 0.0, 1.0], [0.9, 0.1], [0.2, 0.8], [0.5, 0.25, 0.25]])
    out = decision_loss(
        model, torch.zeros(1, 24, 4), strategy=STRATEGY, inputs=inputs, input_ids=INPUT_IDS, brier_weight=0.1
    )
    out["loss"].backward()

    assert inputs.num_labelled == 0 and out["loss"].item() == 0.0
    assert all(out[name].numel() == 0 for name in ("ce", "brier", "accuracy"))
    assert torch.equal(model.logits.grad, torch.zeros_like(model.logits))
    assert len(model.seen) == 1  # the head still runs, as on every micro-batch


def test_decision_loss_is_finite_at_minus_inf_on_options_with_target_0():
    # Probability 0 on false in r's outage and on 0 and 1 in r's urgency, whose targets are 0 there.
    model = StubDecisionModel([[1.0, 0.0], [0.75, 0.25], [0.0, 0.0, 1.0], [0.9, 0.1], [0.2, 0.8], [0.5, 0.25, 0.25]])
    inputs = _decision_loss_inputs()
    out = decision_loss(
        model, torch.zeros(1, 24, 4), strategy=STRATEGY, inputs=inputs, input_ids=INPUT_IDS, brier_weight=0.1
    )
    loss = out["loss"]
    loss.backward()

    ce = -(0.25 * math.log(0.75) + 0.75 * math.log(0.25)) - (0.5 * math.log(0.2) + 0.5 * math.log(0.8)) - math.log(0.5)
    brier = 0.5 + 0.18 + 0.375
    torch.testing.assert_close(loss, torch.tensor(ce + 0.1 * brier))
    assert torch.isfinite(model.logits.grad).all()


class TokenStrategy(StubStrategy):
    """StubStrategy with the record's own tokens, read as the mean of the decision head's logits over the reads."""

    def render_request(self, request, max_tokens):
        first = 10 if request.id == "r" else 20
        return {**super().render_request(request, max_tokens), "input_ids": list(range(first, first + 10))}

    def option_logits(self, model, hidden_states, input_ids, questions):
        logits = model.decision_head(hidden_states)  # once for the micro-batch
        return torch.cat(
            [
                torch.cat([logits[start:end] for start, end in question.reads]).mean(0)[: len(question.options)]
                for question in questions
            ]
        )


class SegmentDecisionModel(nn.Module):
    """A decision model whose backbone attends both ways within each segment."""

    def __init__(self, vocab: int = 32, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.position = nn.Embedding(16, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.model = nn.Module()  # inject_prime_lm_head expects a backbone module at model.model
        self.model.forward = self.backbone
        self.lm_head = nn.Linear(hidden, vocab, bias=False).requires_grad_(False)  # the readout never uses it
        self.decision_head = nn.Linear(hidden, 3)
        inject_prime_lm_head(self)

    def backbone(self, input_ids, position_ids, seq_lens, **kwargs):
        states = (self.embed(input_ids) + self.position(position_ids))[0]
        states = torch.cat([segment + segment.mean(0) for segment in states.split(seq_lens.tolist())])
        return SimpleNamespace(last_hidden_state=self.norm(states)[None])


def _reference_loss(model, sample: dict, brier_weight: float) -> torch.Tensor:
    """A record's decision loss computed alone, without packing, in plain torch per labelled question."""
    hidden_states = model.model(
        torch.from_numpy(sample["input_ids"])[None],
        torch.from_numpy(sample["position_ids"])[None],
        torch.tensor(sample["seq_lens"]),
    ).last_hidden_state[0]
    loss = torch.zeros(())
    for question in sample["sample_data"]["questions"]:
        if question["target"] is None:
            continue
        reads = torch.cat([hidden_states[start:end] for start, end in question["reads"]])
        p = model.decision_head(reads).mean(0)[: len(question["options"])].softmax(-1)
        target = torch.tensor(question["target"])
        loss = loss - (target * p.log()).sum() + brier_weight * ((p - target) ** 2).sum()
    return loss


def test_decision_loss_equals_a_per_record_reference():
    torch.manual_seed(0)
    model = SegmentDecisionModel()
    strategy = TokenStrategy(None, CONFIG, READ_SPANS)
    source = SystemOneSource(Dataset.from_list([ROW]), strategy)
    batch = _micro_batch(source, [ROW, SECOND])
    inputs = batch["decision_loss_inputs"]
    head_calls = []
    model.decision_head.register_forward_hook(lambda *args: head_calls.append(1))

    def packed_loss(seq_lens: torch.Tensor) -> torch.Tensor:
        loss_fn = partial(
            decision_loss, model, strategy=strategy, inputs=inputs, input_ids=batch["input_ids"], brier_weight=0.1
        )
        out = forward(model, batch["input_ids"], batch["position_ids"], seq_lens=seq_lens, hidden_states_fn=loss_fn)
        return out["loss"]

    loss = packed_loss(batch["seq_lens"])
    loss.backward()
    assert len(head_calls) == 1  # one readout for the micro-batch's two records
    grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    assert "lm_head.weight" not in grads
    model.zero_grad()

    samples = [source._process(example) for example in (ROW, SECOND)]
    reference = sum(_reference_loss(model, sample, brier_weight=0.1) for sample in samples)
    reference.backward()
    torch.testing.assert_close(loss, reference)
    for name, param in model.named_parameters():
        if param.grad is not None:
            torch.testing.assert_close(grads[name], param.grad)

    # The padding joined to the last record's segment changes that record's read states, and the loss.
    with torch.no_grad():
        merged = packed_loss(torch.tensor([4, 6, 4, 10]))
    assert not torch.allclose(merged, loss)


def test_lm_head_logits_are_float32_from_the_rows_of_the_tokens():
    torch.manual_seed(0)
    model = nn.Module()
    model.lm_head = nn.Linear(8, 32, bias=False).to(torch.bfloat16)
    hidden_states = torch.randn(6, 8).to(torch.bfloat16)
    positions, token_ids = [5, 2], [[7, 3, 20], [3]]

    logits = lm_head_logits(model, hidden_states, positions, token_ids)

    weight = model.lm_head.weight.double()
    exact = [hidden_states[p].double() @ weight[t] for p, tokens in zip(positions, token_ids) for t in tokens]
    assert logits.dtype == torch.float32
    # Closer than a logit rounded to bf16 would be
    torch.testing.assert_close(logits, torch.stack(exact).float())

    # A softcap of the LM head caps these logits too
    model.lm_head.softcap = 0.5
    capped = lm_head_logits(model, hidden_states, positions, token_ids)
    torch.testing.assert_close(capped, 0.5 * torch.tanh(logits / 0.5))


class HeadsConfig(PretrainedConfig):
    model_type = "decision_loss_heads_test"


class Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mix = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x + torch.tanh(self.mix(x))


class Backbone(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([Block(hidden), Block(hidden)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids, position_ids, *, seq_lens, seq_lens_are_pre_shard=False):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        # Each attention segment both ways: every token adds its segment's mean
        x = torch.cat([segment + segment.mean(0) for segment in x[0].split(seq_lens.tolist())])[None]
        return SimpleNamespace(last_hidden_state=self.norm(x))


class LMHeadStrategy(TokenStrategy):
    """Reads options from the LM head's logits at the first read token, plus the score head's mean over the reads."""

    def option_logits(self, model, hidden_states, input_ids, questions):
        lm_logits = model.lm_head(hidden_states[None])["logits"][0]
        scores = model.score_head(hidden_states)[:, 0].float()
        return torch.cat(
            [
                lm_logits[question.reads[0][0], : len(question.options)] + scores[slice(*question.reads[0])].mean()
                for question in questions
            ]
        )


class LMHeadRowsStrategy(TokenStrategy):
    """LMHeadStrategy, with the options' logits read by ``lm_head_logits`` from their rows of the LM head's weight."""

    def option_logits(self, model, hidden_states, input_ids, questions):
        positions = [question.reads[0][0] for question in questions]
        lm_logits = lm_head_logits(model, hidden_states, positions, [range(len(q.options)) for q in questions])
        scores = model.score_head(hidden_states)[:, 0].float()
        means = [scores[slice(*question.reads[0])].mean().expand(len(question.options)) for question in questions]
        return lm_logits + torch.cat(means)


class ScoreHeadStrategy(TokenStrategy):
    """Reads each question's options from the score head alone, never from the LM head."""

    def option_logits(self, model, hidden_states, input_ids, questions):
        scores = model.score_head(hidden_states)[:, 0].float()
        return torch.cat(
            [
                scores[slice(*question.reads[0])].mean() * torch.arange(len(question.options), dtype=torch.float32)
                for question in questions
            ]
        )


class HeadsModel(PreTrainedModelPrimeRL):
    """A decision model with a score head, and an LM head tied to the embeddings or not. The score head and a tied LM
    head are in the root FSDP unit; an untied LM head shares a unit with the final norm."""

    def __init__(self, config, freeze_lm_head: bool = False, vocab: int = 32, hidden: int = 8):
        super().__init__(config)
        self.model = Backbone(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        elif freeze_lm_head:
            self.lm_head.requires_grad_(False)  # a readout that never uses the untied LM head
        self.score_head = nn.Linear(hidden, 1)

    def init_buffers_post_meta(self):
        pass


def heads_model(tied: bool, strategy: type[SystemOneStrategy]) -> HeadsModel:
    torch.manual_seed(0)
    model = HeadsModel(HeadsConfig(tie_word_embeddings=tied), freeze_lm_head=strategy is ScoreHeadStrategy)
    inject_prime_lm_head(model)
    return model


def heads_loss(model: nn.Module, rank: int, strategy_cls: type[SystemOneStrategy]) -> torch.Tensor:
    """The loss of each rank's micro-batch: records r and s, with r's labels on rank 0 and s's on rank 1."""
    strategy = strategy_cls(None, CONFIG, READ_SPANS)
    rows = [ROW, SECOND] if rank == 0 else [{**SECOND, "id": "r"}, {**ROW, "id": "s"}]
    batch = _micro_batch(SystemOneSource(Dataset.from_list([ROW]), strategy), rows)
    inputs = batch["decision_loss_inputs"]
    loss_fn = partial(
        decision_loss, model, strategy=strategy, inputs=inputs, input_ids=batch["input_ids"], brier_weight=0.1
    )
    out = forward(
        model, batch["input_ids"], batch["position_ids"], seq_lens=batch["seq_lens"], hidden_states_fn=loss_fn
    )
    return out["loss"]


def fsdp_worker(rank: int, world: int, tied: bool, strategy, init_file: str, out_dir: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world))
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    parallel_dims_module.device_type = "cpu"  # the mesh is built on CUDA by default
    config = ModelConfig(name="heads", ep=1, compile=None)
    model = heads_model(tied, strategy)
    setup_fsdp(model, config, get_parallel_dims(config))  # prime-rl's real FSDP setup

    def grads() -> dict:
        return {name: p.grad.full_tensor() for name, p in model.named_parameters() if p.grad is not None}

    # As in the trainer: a step's backward, then validation (a no-grad forward, then reshard_module before clipping),
    # then the next step's backward on cleared gradients
    heads_loss(model, rank, strategy).backward()
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)
    with torch.no_grad():
        heads_loss(model, rank, strategy)
    reshard_module(model)
    steps = [grads()]
    model.zero_grad(set_to_none=True)
    heads_loss(model, rank, strategy).backward()
    steps.append(grads())
    torch.save(steps, os.path.join(out_dir, f"{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("strategy", [LMHeadStrategy, LMHeadRowsStrategy, ScoreHeadStrategy])
@pytest.mark.parametrize("tied", [True, False])
def test_decision_loss_under_fsdp(tied, strategy, tmp_path):
    world = 2
    args = (world, tied, strategy, str(tmp_path / "rendezvous"), str(tmp_path))
    mp.spawn(fsdp_worker, args=args, nprocs=world, join=True)

    # The same micro-batches on one process without FSDP, in bf16 as FSDP computes; FSDP averages over ranks.
    model = heads_model(tied, strategy).to(torch.bfloat16)
    sum(heads_loss(model, rank, strategy) for rank in range(world)).div(world).backward()
    expected = {name: param.grad.float() for name, param in model.named_parameters() if param.grad is not None}
    if not tied:  # a tied LM head's weight is listed as the embeddings'
        assert ("lm_head.weight" in expected) == (strategy is not ScoreHeadStrategy)
    for rank in range(world):
        for grads in torch.load(tmp_path / f"{rank}.pt"):
            assert grads.keys() == expected.keys()
            for name, grad in expected.items():
                torch.testing.assert_close(grads[name], grad, rtol=0.05, atol=0.02)
