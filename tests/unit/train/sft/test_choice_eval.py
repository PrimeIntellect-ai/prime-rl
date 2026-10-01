import json
import os
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch
from pydantic import ValidationError

from prime_rl.configs.sft import SFTChoiceEvalConfig, SFTConfig
from prime_rl.trainer.sft.choice import choice_logits
from prime_rl.trainer.sft.choice_eval import (
    MARKER_FILE,
    PREDICTIONS_FILE,
    ChoiceEvalSite,
    choice_eval_step_dir,
    choice_row_ids,
    due_choice_evals,
    merge_choice_predictions,
    score_choice_predictions,
    unscored_choice_evals,
    write_choice_predictions,
)
from prime_rl.trainer.sft.data import (
    PACKED_CHOICE_EVAL_SCHEMA_VERSION,
    PackedChoiceDataset,
    PackedChoiceEvalSet,
    setup_choice_eval_sets,
)
from prime_rl.trainer.world import reset_world
from prime_rl.utils.config import cli
from prime_rl.utils.cp import shard_for_cp

SEQ_LEN = 20
MAX_CHOICES = 4
VOCAB = 50

EVAL_BIN_SCHEMA = pa.schema(
    [
        ("input_ids", pa.list_(pa.int32())),
        ("seq_lens", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("choice_ids", pa.list_(pa.list_(pa.int32()))),
        ("row_ids", pa.list_(pa.int64())),
    ]
)


@pytest.fixture(autouse=True)
def fresh_world():
    yield
    reset_world()


def set_world(rank: int, world_size: int) -> None:
    reset_world()
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world_size)
    )


def make_eval_bin(prompts: list[tuple[list[int], list[int]]], first_row_id: int) -> dict:
    """A bin row from ``(prompt tokens, choice_ids)`` prompts, each read at its last token."""
    row = {name: [] for name in EVAL_BIN_SCHEMA.names}
    for offset, (tokens, choice_ids) in enumerate(prompts):
        row["input_ids"].extend(tokens)
        row["positions"].append(len(row["input_ids"]) - 1)
        row["seq_lens"].append(len(tokens))
        row["choice_ids"].append(choice_ids)
        row["row_ids"].append(first_row_id + offset)
    return row


def make_eval_bins(num_bins: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    bins = []
    for _ in range(num_bins):
        prompts = [
            ([rng.randrange(VOCAB) for _ in range(rng.randint(2, 6))], rng.sample(range(VOCAB), rng.randint(1, 4)))
            for _ in range(rng.randint(1, 3))
        ]
        bins.append(make_eval_bin(prompts, first_row_id=sum(len(b["row_ids"]) for b in bins)))
    return bins


def write_eval_export(path: Path, bins: list[dict], name: str = "bench-a", bins_per_file: int = 2) -> Path:
    path.mkdir(parents=True)
    files = []
    for start in range(0, len(bins), bins_per_file):
        file_name = f"bins-{len(files):05d}.arrow"
        with pa.ipc.new_file(path / file_name, EVAL_BIN_SCHEMA) as writer:
            writer.write_table(pa.Table.from_pylist(bins[start : start + bins_per_file], schema=EVAL_BIN_SCHEMA))
        files.append({"name": file_name, "num_bins": len(bins[start : start + bins_per_file])})
    manifest = {
        "schema_version": PACKED_CHOICE_EVAL_SCHEMA_VERSION,
        "name": name,
        "seq_len": SEQ_LEN,
        "num_bins": len(bins),
        "num_rows": sum(len(b["row_ids"]) for b in bins),
        "max_choices": MAX_CHOICES,
        "files": files,
        "source": {"kind": "matrix"},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def expected_eval_batch(row: dict, seq_len: int = SEQ_LEN) -> dict[str, list]:
    pad_len = seq_len - len(row["input_ids"])
    choice_ids = [[-1] * MAX_CHOICES for _ in range(seq_len)]
    row_ids = [-1] * seq_len
    for position, ids, row_id in zip(row["positions"], row["choice_ids"], row["row_ids"]):
        choice_ids[position][: len(ids)] = ids
        row_ids[position] = row_id
    return {
        "input_ids": [row["input_ids"] + [0] * pad_len],
        "position_ids": [[i for length in row["seq_lens"] for i in range(length)] + list(range(pad_len))],
        "seq_lens": row["seq_lens"][:-1] + [row["seq_lens"][-1] + pad_len],
        "choice_ids": [choice_ids],
        "row_ids": [row_ids],
    }


def assert_eval_batch_matches(batch: dict, row: dict) -> None:
    for key, expected in expected_eval_batch(row).items():
        torch.testing.assert_close(batch[key], torch.tensor(expected, dtype=batch[key].dtype), msg=key)


def test_every_rank_runs_the_same_forwards_and_real_slots_cover_each_bin_once(tmp_path: Path):
    bins = make_eval_bins(5)
    path = write_eval_export(tmp_path / "bench-a", bins)

    scored = []
    batches_by_rank = []
    for rank in range(8):
        set_world(rank, world_size=8)
        batches = list(PackedChoiceEvalSet(path, seq_len=SEQ_LEN, non_dp_size=2).batches())
        batches_by_rank.append(batches)

        # 4 data ranks over 5 bins: slot d + 4j, replaying bin slot % 5 past the last bin
        data_rank = rank // 2
        assert len(batches) == 2
        for forward, (batch, replay) in enumerate(batches):
            slot = data_rank + 4 * forward
            assert replay == (slot >= len(bins))
            assert_eval_batch_matches(batch, bins[slot % len(bins)])
            if not replay and rank % 2 == 0:
                scored.append(slot % len(bins))

    for rank in range(0, 8, 2):
        for (batch, replay), (peer_batch, peer_replay) in zip(batches_by_rank[rank], batches_by_rank[rank + 1]):
            assert replay == peer_replay
            for key in batch:
                torch.testing.assert_close(batch[key], peer_batch[key])
    assert sorted(scored) == list(range(len(bins)))


def test_eval_set_maps_choice_counts_by_row_id_and_accepts_a_prompt_ending_the_bin(tmp_path: Path):
    # The last prompt fills the bin, so its prediction is read at the bin's last token
    row = make_eval_bin([([1, 2, 3], [4, 5]), ([6, 7, 8, 9], [10, 11, 12])], first_row_id=0)
    row["row_ids"] = [1, 0]
    eval_set = PackedChoiceEvalSet(write_eval_export(tmp_path / "bench-a", [row]), seq_len=SEQ_LEN)

    assert row["positions"][-1] == len(row["input_ids"]) - 1
    assert eval_set.choice_counts.tolist() == [3, 2]
    ((batch, replay),) = list(eval_set.batches())
    assert not replay
    assert_eval_batch_matches(batch, row)


def test_eval_and_train_loaders_reject_each_others_exports(tmp_path: Path):
    path = write_eval_export(tmp_path / "bench-a", make_eval_bins(2))
    with pytest.raises(ValueError, match="schema 'simile-packed-choice-eval/v1'"):
        PackedChoiceDataset(path, seq_len=SEQ_LEN)

    manifest = json.loads((path / "manifest.json").read_text())
    (path / "manifest.json").write_text(json.dumps({**manifest, "schema_version": "simile-packed-choice/v2"}))
    with pytest.raises(ValueError, match="schema 'simile-packed-choice/v2'"):
        PackedChoiceEvalSet(path, seq_len=SEQ_LEN)


def with_row_ids(row_ids: list[int]) -> dict:
    row = make_eval_bin([([1, 2], [3]), ([4, 5], [6]), ([7, 8], [9])], first_row_id=0)
    row["row_ids"] = row_ids
    return row


@pytest.mark.parametrize(
    ("bins", "match"),
    [
        pytest.param([with_row_ids([0, 1, 1])], "row_ids do not cover 0..2 exactly once", id="duplicate-row"),
        pytest.param([with_row_ids([0, 1, 3])], "row_ids do not cover 0..2 exactly once", id="gap"),
        pytest.param(
            [{**make_eval_bin([([1, 2], [3])], 0), "positions": [2]}], "out-of-range positions", id="past-last-token"
        ),
        pytest.param(
            [{name: [] for name in EVAL_BIN_SCHEMA.names} | {"input_ids": [1], "seq_lens": [1]}],
            "has no rows",
            id="empty-bin",
        ),
        pytest.param([make_eval_bin([([1, 2], [3, 4, 5, 6, 7])], 0)], "outside 1..max_choices", id="too-many"),
    ],
)
def test_eval_set_rejects_a_malformed_export_at_startup(tmp_path: Path, bins: list[dict], match: str):
    with pytest.raises(ValueError, match=match):
        PackedChoiceEvalSet(write_eval_export(tmp_path / "bench-a", bins), seq_len=SEQ_LEN)


def test_setup_rejects_a_set_named_differently_from_its_export(tmp_path: Path):
    path = write_eval_export(tmp_path / "bench-a", make_eval_bins(2), name="bench-a")
    config = SFTChoiceEvalConfig(import_path="m.f", sets={"bench-b": {"path": path}})

    with pytest.raises(ValueError, match="the export of 'bench-a'"):
        setup_choice_eval_sets(config, seq_len=SEQ_LEN, non_dp_size=1)


def test_cp_shards_score_every_row_once_with_its_unsharded_logits(tmp_path: Path):
    # Rows read at positions 2, 9 and 19: 9 ends shard 0 and 19 is the bin's last token
    row = make_eval_bin(
        [([1, 2, 3], [4, 7]), ([8, 9, 10, 11, 12, 13, 14], [11, 12, 13]), ([15] * 10, [17])], first_row_id=0
    )
    row["row_ids"] = [2, 0, 1]
    assert row["positions"] == [2, 9, 19]
    ((batch, _),) = list(PackedChoiceEvalSet(write_eval_export(tmp_path / "bench-a", [row]), seq_len=SEQ_LEN).batches())
    hidden = torch.randn(1, SEQ_LEN, 8)
    weight = torch.randn(VOCAB, 8)
    full_logits = choice_logits(hidden, weight, batch["choice_ids"])
    full_rows = choice_row_ids(batch["choice_ids"], batch["row_ids"])
    assert full_rows.tolist() == [2, 0, 1]

    shard_rows, shard_logits = [], []
    for cp_rank in range(2):
        choice_ids, row_ids, hidden_shard = (
            shard_for_cp(tensor, cp_rank=cp_rank, cp_world_size=2)
            for tensor in (batch["choice_ids"], batch["row_ids"], hidden)
        )
        shard_rows.append(choice_row_ids(choice_ids, row_ids))
        shard_logits.append(choice_logits(hidden_shard, weight, choice_ids))

    assert [rows.tolist() for rows in shard_rows] == [[2, 0], [1]]
    merged = merge_choice_predictions(
        [(rows.numpy(), logits.numpy()) for rows, logits in zip(shard_rows, shard_logits)], num_rows=3
    )
    np.testing.assert_allclose(merged, full_logits[torch.argsort(full_rows)].numpy())


def test_merge_orders_rows_and_accepts_ranks_without_rows():
    logits = np.arange(12, dtype=np.float32).reshape(4, 3)
    parts = [
        (np.array([3, 0]), logits[[3, 0]]),
        (np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float32)),
        (np.array([2, 1]), logits[[2, 1]]),
    ]

    np.testing.assert_array_equal(merge_choice_predictions(parts, num_rows=4), logits)


@pytest.mark.parametrize(
    ("row_ids", "match"),
    [
        pytest.param([0, 1, 2, 2], "1 missing, 1 duplicated, 0 out of range", id="duplicate"),
        pytest.param([0, 1, 2], "1 missing, 0 duplicated, 0 out of range", id="missing"),
        pytest.param([0, 1, 2, 4], "1 missing, 0 duplicated, 1 out of range", id="out-of-range"),
    ],
)
def test_merge_rejects_predictions_that_do_not_cover_every_row_once(row_ids: list[int], match: str):
    part = (np.array(row_ids), np.zeros((len(row_ids), 2), dtype=np.float32))

    with pytest.raises(ValueError, match=match):
        merge_choice_predictions([part], num_rows=4)


def read_predictions(path: Path) -> pa.Table:
    return pa.ipc.open_file(pa.memory_map(str(path))).read_all()


def test_predictions_file_keeps_each_rows_real_choices_only(tmp_path: Path):
    logits = np.array([[0.5, -1.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, -2.0]], dtype=np.float32)
    path = tmp_path / PREDICTIONS_FILE

    write_choice_predictions(path, logits, np.array([2, 1, 3]))

    table = read_predictions(path)
    assert table.schema.field("row_id").type == pa.int64()
    assert table.schema.field("choice_logits").type == pa.list_(pa.float32())
    assert table.to_pydict() == {"row_id": [0, 1, 2], "choice_logits": [[0.5, -1.0], [2.0], [1.0, 3.0, -2.0]]}
    assert list(tmp_path.iterdir()) == [path]


def eval_config(eval_on_start: bool = False) -> SFTChoiceEvalConfig:
    return SFTChoiceEvalConfig(
        import_path="m.f",
        eval_on_start=eval_on_start,
        sets={
            "bench-a": {"path": "/m", "interval": 50},
            "bench-b": {"path": "/s", "interval": 100},
            "bench-c": {"path": "/u"},
        },
    )


@pytest.mark.parametrize(
    ("eval_on_start", "step", "site", "due"),
    [
        pytest.param(False, 0, "start", [], id="fresh-start"),
        pytest.param(True, 0, "start", ["bench-a", "bench-b", "bench-c"], id="fresh-start-eval-on-start"),
        pytest.param(True, 50, "start", ["bench-a"], id="resume-at-an-interval-step"),
        pytest.param(True, 70, "start", [], id="resume-between-intervals"),
        pytest.param(False, 49, "step", [], id="off-interval"),
        pytest.param(False, 100, "step", ["bench-a", "bench-b"], id="common-multiple"),
        pytest.param(False, 300, "step", [], id="last-step-left-to-final"),
        pytest.param(False, 300, "final", ["bench-a", "bench-b", "bench-c"], id="final"),
        pytest.param(False, 7, "final", ["bench-a", "bench-b", "bench-c"], id="final-off-interval"),
    ],
)
def test_due_choice_evals_follow_start_interval_and_final_cadence(
    eval_on_start: bool, step: int, site: ChoiceEvalSite, due: list[str]
):
    assert due_choice_evals(eval_config(eval_on_start), step, site, max_steps=300) == due


def gathered_parts(logits: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array([1]), logits[[1]]), (np.array([0]), logits[[0]])]


def test_scoring_logs_the_result_at_the_eval_step_and_marks_it_scored(tmp_path: Path):
    logits = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    output_dir = choice_eval_step_dir(tmp_path, "bench-a", 50)
    calls, logged = [], []

    def score(**kwargs):
        calls.append(kwargs)
        assert read_predictions(kwargs["predictions"]).to_pydict()["choice_logits"] == [[1.0, 2.0], [3.0]]
        return {"eval-bench-a/pass_rate": 0.5}

    score_choice_predictions(
        score,
        lambda metrics, step: logged.append((metrics, step)),
        name="bench-a",
        path=tmp_path / "evals" / "bench-a",
        step=50,
        output_dir=output_dir,
        parts=gathered_parts(logits),
        choice_counts=np.array([2, 1]),
        metrics={"time/choice_eval/bench-a": 1.5},
    )

    assert calls == [
        {
            "name": "bench-a",
            "path": tmp_path / "evals" / "bench-a",
            "step": 50,
            "predictions": output_dir / PREDICTIONS_FILE,
            "output_dir": output_dir,
        }
    ]
    row = {"eval-bench-a/pass_rate": 0.5, "time/choice_eval/bench-a": 1.5}
    assert logged == [(row, 50)]
    assert json.loads((output_dir / MARKER_FILE).read_text()) == row
    assert unscored_choice_evals(tmp_path, ["bench-a", "bench-c"], 50) == ["bench-c"]


def test_without_a_scoring_function_the_predictions_are_kept_and_the_step_marked(tmp_path: Path):
    output_dir = choice_eval_step_dir(tmp_path, "bench-a", 50)
    logged = []

    score_choice_predictions(
        None,
        lambda metrics, step: logged.append((metrics, step)),
        name="bench-a",
        path=tmp_path / "evals" / "bench-a",
        step=50,
        output_dir=output_dir,
        parts=gathered_parts(np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)),
        choice_counts=np.array([2, 1]),
        metrics={"time/choice_eval/bench-a": 1.5},
    )

    assert read_predictions(output_dir / PREDICTIONS_FILE).to_pydict()["choice_logits"] == [[1.0, 2.0], [3.0]]
    assert logged == [({"time/choice_eval/bench-a": 1.5}, 50)]
    assert json.loads((output_dir / MARKER_FILE).read_text()) == {"time/choice_eval/bench-a": 1.5}
    assert unscored_choice_evals(tmp_path, ["bench-a"], 50) == []


def failing_score(**kwargs):
    raise RuntimeError("scorer exited with 1")


@pytest.mark.parametrize(
    ("score", "parts"),
    [
        pytest.param(failing_score, None, id="scorer-fails"),
        pytest.param(lambda **kwargs: {}, [(np.array([0]), np.zeros((1, 2), dtype=np.float32))], id="row-missing"),
    ],
)
def test_a_failed_scoring_logs_nothing_and_leaves_the_step_unscored(tmp_path: Path, score, parts):
    logged = []

    score_choice_predictions(
        score,
        lambda metrics, step: logged.append((metrics, step)),
        name="bench-a",
        path=tmp_path / "evals" / "bench-a",
        step=50,
        output_dir=choice_eval_step_dir(tmp_path, "bench-a", 50),
        parts=parts or gathered_parts(np.zeros((2, 2), dtype=np.float32)),
        choice_counts=np.array([2, 1]),
        metrics={},
    )

    assert logged == []
    assert not (choice_eval_step_dir(tmp_path, "bench-a", 50) / MARKER_FILE).exists()
    assert unscored_choice_evals(tmp_path, ["bench-a"], 50) == ["bench-a"]


SFT_MODEL = {"model": {"name": "PrimeIntellect/Qwen3-0.6B"}}
PACKED_CHOICE = {
    "data": {"type": "packed_choice", "name": "/unused", "seq_len": SEQ_LEN},
    "loss": {"import_path": "m.loss"},
}
CHOICE_EVAL = {"import_path": "my_module.score", "sets": {"bench-a": {"path": "/evals/bench-a", "interval": 50}}}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"choice_eval": {**CHOICE_EVAL, "sets": {}}}, "at least 1 item", id="no-sets"),
        pytest.param(
            {"choice_eval": {**CHOICE_EVAL, "sets": {"../bench-a": {"path": "/m"}}}}, "match pattern", id="path-name"
        ),
        pytest.param({"choice_eval": {**CHOICE_EVAL, "sets": {".": {"path": "/m"}}}}, "match pattern", id="dot-name"),
        pytest.param(
            {"choice_eval": {**CHOICE_EVAL, "sets": {"bench-a": {"path": "/m", "interval": 0}}}},
            "greater than or equal to 1",
            id="zero-interval",
        ),
        pytest.param(
            {"choice_eval": {"sets": CHOICE_EVAL["sets"], "kwargs": {"timeout_s": 600}}},
            "choice_eval.kwargs needs choice_eval.import_path",
            id="kwargs-without-function",
        ),
    ],
)
def test_choice_eval_config_rejects(overrides: dict, match: str):
    with pytest.raises(ValidationError, match=match):
        SFTConfig.model_validate({**SFT_MODEL, **PACKED_CHOICE, **overrides})


def test_choice_eval_config_accepts_no_scoring_function():
    config = SFTConfig.model_validate({**SFT_MODEL, **PACKED_CHOICE, "choice_eval": {"sets": CHOICE_EVAL["sets"]}})

    assert config.choice_eval is not None
    assert config.choice_eval.import_path is None


def test_choice_eval_config_rejects_full_offload_with_any_data():
    full_offload = {"model": {"optim_cpu_offload": False, "full_offload": True}, "optim": {"max_norm": None}}
    assert SFTConfig.model_validate(full_offload).model.full_offload is not None

    with pytest.raises(ValidationError, match="\\[choice_eval\\] does not support model.full_offload"):
        SFTConfig.model_validate({**full_offload, "choice_eval": CHOICE_EVAL})


def test_choice_eval_fragment_composes_with_a_scalar_cli_override(tmp_path: Path):
    fragment = tmp_path / "choice_eval.json"
    fragment.write_text(
        json.dumps(
            {
                "choice_eval": {
                    **CHOICE_EVAL,
                    "kwargs": {"timeout_s": 600},
                    "sets": {
                        "bench-a": {"path": "/evals/bench-a", "interval": 50},
                        "bench-b": {"path": "/evals/bench-b", "interval": 50},
                        "bench-c": {"path": "/evals/bench-c", "interval": None},
                    },
                }
            }
        )
    )
    base = tmp_path / "base.json"
    base.write_text(json.dumps({**SFT_MODEL, **PACKED_CHOICE}))

    config = cli(SFTConfig, args=["@", str(base), "@", str(fragment), "--choice-eval.eval-on-start", "true"])

    assert config.choice_eval is not None
    assert config.choice_eval.eval_on_start
    assert config.choice_eval.kwargs == {"timeout_s": 600}
    assert {name: s.interval for name, s in config.choice_eval.sets.items()} == {
        "bench-a": 50,
        "bench-b": 50,
        "bench-c": None,
    }
