import gc
import json
import os
import random
from pathlib import Path

import pyarrow as pa
import pytest
import torch
from pydantic import ValidationError

from prime_rl.configs.sft import PackedChoiceDataConfig, SFTConfig, SFTDataConfig
from prime_rl.trainer.sft.choice import choice_logits, choice_loss_inputs, supervised_rows
from prime_rl.trainer.sft.data import (
    PACKED_CHOICE_SCHEMA_VERSION,
    PackedChoiceDataset,
    get_dataset_progress,
    packed_choice_collate,
    packed_choice_max_steps,
    packed_choice_mix_components,
    setup_dataloader,
)
from prime_rl.trainer.world import reset_world
from prime_rl.utils.cp import shard_for_cp

SEQ_LEN = 20
MAX_CHOICES = 4
VOCAB = 50
MIX_COMPONENTS = ["kl_choice", "l1_choice", "kl_full"]

BIN_SCHEMA = pa.schema(
    [
        ("input_ids", pa.list_(pa.int32())),
        ("seq_lens", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("choice_ids", pa.list_(pa.list_(pa.int32()))),
        ("target_probs", pa.list_(pa.list_(pa.float32()))),
        ("weights", pa.list_(pa.float32())),
        ("observed_ids", pa.list_(pa.int32())),
        ("mix_weights", pa.list_(pa.list_(pa.float32()))),
    ]
)


def make_bin(examples: list[tuple[list[int], int, list[int], list[float], float, list[float]]]) -> dict:
    """A bin row from ``(tokens, label_index, choice_ids, target_probs, weight, mix_weights)`` examples."""
    row = {name: [] for name in BIN_SCHEMA.names}
    for tokens, label_index, choice_ids, target_probs, weight, mix_weights in examples:
        row["positions"].append(len(row["input_ids"]) + label_index - 1)
        row["observed_ids"].append(tokens[label_index])
        row["input_ids"].extend(tokens)
        row["seq_lens"].append(len(tokens))
        row["choice_ids"].append(choice_ids)
        row["target_probs"].append(target_probs)
        row["weights"].append(weight)
        row["mix_weights"].append(mix_weights)
    return row


def make_bins(num_bins: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    bins = []
    for _ in range(num_bins):
        examples = []
        for _ in range(rng.randint(1, 3)):
            tokens = [rng.randrange(VOCAB) for _ in range(rng.randint(3, 6))]
            label_index = rng.randrange(1, len(tokens))
            choice_ids = rng.sample(range(VOCAB), rng.randint(1, MAX_CHOICES))
            tokens[label_index] = rng.choice(choice_ids)
            raw = [rng.random() + 0.1 for _ in choice_ids]
            mix = [rng.choice([0.0, 0.25, 0.5, 1.0]) for _ in MIX_COMPONENTS]
            examples.append((tokens, label_index, choice_ids, [p / sum(raw) for p in raw], rng.uniform(0.1, 2.0), mix))
        bins.append(make_bin(examples))
    return bins


def write_export(path: Path, bins: list[dict], split: str = "train", bins_per_file: int = 2) -> Path:
    path.mkdir(parents=True)
    files = []
    for start in range(0, len(bins), bins_per_file):
        name = f"bins-{len(files):05d}.arrow"
        with pa.ipc.new_file(path / name, BIN_SCHEMA) as writer:
            writer.write_table(pa.Table.from_pylist(bins[start : start + bins_per_file], schema=BIN_SCHEMA))
        files.append({"name": name, "num_bins": len(bins[start : start + bins_per_file])})
    manifest = {
        "schema_version": PACKED_CHOICE_SCHEMA_VERSION,
        "split": split,
        "seq_len": SEQ_LEN,
        "num_bins": len(bins),
        "num_examples": sum(len(b["positions"]) for b in bins),
        "num_tokens": sum(len(b["input_ids"]) for b in bins),
        "max_choices": MAX_CHOICES,
        "mix_components": MIX_COMPONENTS,
        "files": files,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def expected_tensors(row: dict, seq_len: int) -> dict[str, list]:
    pad_len = seq_len - len(row["input_ids"])
    choice_ids = [[-1] * MAX_CHOICES for _ in range(seq_len)]
    choice_targets = [[0.0] * MAX_CHOICES for _ in range(seq_len)]
    choice_weights = [0.0] * seq_len
    choice_mix_weights = [[0.0] * len(MIX_COMPONENTS) for _ in range(seq_len)]
    for position, ids, probs, weight, mix in zip(
        row["positions"], row["choice_ids"], row["target_probs"], row["weights"], row["mix_weights"]
    ):
        choice_ids[position][: len(ids)] = ids
        choice_targets[position][: len(probs)] = probs
        choice_weights[position] = weight
        choice_mix_weights[position] = mix
    return {
        "input_ids": [row["input_ids"] + [0] * pad_len],
        "position_ids": [[i for length in row["seq_lens"] for i in range(length)] + list(range(pad_len))],
        "seq_lens": row["seq_lens"][:-1] + [row["seq_lens"][-1] + pad_len],
        "target_ids": [row["input_ids"][1:] + [0] * (pad_len + 1)],
        "loss_mask": [[i in row["positions"] for i in range(seq_len)]],
        "choice_ids": [choice_ids],
        "choice_targets": [choice_targets],
        "choice_weights": [choice_weights],
        "choice_mix_weights": [choice_mix_weights],
    }


def assert_batch_matches(batch: dict, row: dict, seq_len: int = SEQ_LEN) -> None:
    for key, expected in expected_tensors(row, seq_len).items():
        torch.testing.assert_close(batch[key], torch.tensor(expected, dtype=batch[key].dtype), msg=key)
    assert batch["choice_mix_weights"].shape == (1, seq_len, len(MIX_COMPONENTS))


def set_world(rank: int, world_size: int) -> None:
    reset_world()
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world_size)
    )


def test_dataset_yields_padded_bins(tmp_path: Path):
    bins = make_bins(5)
    dataset = PackedChoiceDataset(write_export(tmp_path / "train", bins), seq_len=SEQ_LEN + 4, max_epochs=1)
    batches = [packed_choice_collate([sample]) for sample in dataset]

    assert len(batches) == len(bins)
    for batch, row in zip(batches, bins):
        assert_batch_matches(batch, row, seq_len=SEQ_LEN + 4)
        assert batch["mm_kwargs"] is None and batch["mm_token_type_ids"] is None
        # The hidden state at each supervised position predicts the observed answer token
        assert batch["input_ids"][0, torch.tensor(row["positions"]) + 1].tolist() == row["observed_ids"]


@pytest.mark.parametrize("rank", range(4))
def test_dataset_shards_bins_by_data_rank(tmp_path: Path, rank: int):
    bins = make_bins(7)
    path = write_export(tmp_path / "train", bins)
    set_world(rank, world_size=4)

    samples = list(PackedChoiceDataset(path, seq_len=SEQ_LEN, non_dp_size=2, max_epochs=1))

    data_rank = rank // 2
    expected = [bins[index] for index in range(data_rank, len(bins), 2)]
    assert len(samples) == len(expected)
    for sample, row in zip(samples, expected):
        assert_batch_matches(packed_choice_collate([sample]), row)


def test_dataset_repeats_without_max_epochs(tmp_path: Path):
    bins = make_bins(3)
    iterator = iter(PackedChoiceDataset(write_export(tmp_path / "train", bins), seq_len=SEQ_LEN))

    for row in bins + bins[:2]:
        assert_batch_matches(packed_choice_collate([next(iterator)]), row)


def test_dataloader_resumes_at_next_bin(tmp_path: Path):
    bins = make_bins(6)
    config = PackedChoiceDataConfig(name=write_export(tmp_path / "train", bins), seq_len=SEQ_LEN, batch_size=1)
    dataloader = setup_dataloader(PackedChoiceDataset(config.name, seq_len=SEQ_LEN), config)
    dataiter = iter(dataloader)
    for row in bins[:3]:
        assert_batch_matches(next(dataiter), row)
    state_dict = dataloader.state_dict()
    del dataiter, dataloader
    gc.collect()

    dataloader = setup_dataloader(PackedChoiceDataset(config.name, seq_len=SEQ_LEN), config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)
    assert_batch_matches(next(dataiter), bins[3])

    progress = get_dataset_progress(dataloader)
    assert (progress["step"], progress["epoch"]) == (4, 0)
    assert progress["num_samples"] == {"train": sum(len(row["seq_lens"]) for row in bins[:4])}
    assert progress["num_tokens"] == {"train": sum(len(row["input_ids"]) for row in bins[:4])}


def test_dataset_rejects_mismatched_export(tmp_path: Path):
    path = write_export(tmp_path / "train", make_bins(2))
    with pytest.raises(ValueError, match="above data.seq_len"):
        PackedChoiceDataset(path, seq_len=SEQ_LEN - 1)

    manifest = json.loads((path / "manifest.json").read_text())
    (path / "manifest.json").write_text(json.dumps({**manifest, "schema_version": "simile-packed-choice/v1"}))
    with pytest.raises(ValueError, match="schema 'simile-packed-choice/v1'"):
        PackedChoiceDataset(path, seq_len=SEQ_LEN)


def test_dataset_rejects_a_mix_row_of_the_wrong_width(tmp_path: Path):
    row = make_bins(1)[0]
    row["mix_weights"][0] = row["mix_weights"][0][:-1]

    with pytest.raises(ValueError, match="mix_weights row without 3 components"):
        next(iter(PackedChoiceDataset(write_export(tmp_path / "train", [row]), seq_len=SEQ_LEN)))


def test_max_steps_defaults_to_one_pass_over_full_batches(tmp_path: Path):
    config = PackedChoiceDataConfig(name=write_export(tmp_path / "train", make_bins(10)), batch_size=4)

    assert packed_choice_max_steps(config, None) == 2
    assert packed_choice_max_steps(config, 3) == 3
    with pytest.raises(ValueError, match="fewer than one batch"):
        packed_choice_max_steps(config.model_copy(update={"batch_size": 11}), None)


def test_mix_components_come_from_the_train_export_and_must_match_validation(tmp_path: Path):
    train = PackedChoiceDataConfig(name=write_export(tmp_path / "train", make_bins(2)))
    val = PackedChoiceDataConfig(name=write_export(tmp_path / "val", make_bins(2), split="val"))

    assert packed_choice_mix_components(train, val) == tuple(MIX_COMPONENTS)
    assert packed_choice_mix_components(train, None) == tuple(MIX_COMPONENTS)

    manifest = json.loads((val.name / "manifest.json").read_text())
    reordered = [MIX_COMPONENTS[1], MIX_COMPONENTS[0], *MIX_COMPONENTS[2:]]
    (val.name / "manifest.json").write_text(json.dumps({**manifest, "mix_components": reordered}))
    with pytest.raises(ValueError, match="has mix_components \\['l1_choice', 'kl_choice', 'kl_full'\\]"):
        packed_choice_mix_components(train, val)


def load_batch(path: Path, row: dict) -> dict:
    """The collated micro batch of a one-bin export holding ``row``."""
    return packed_choice_collate([next(iter(PackedChoiceDataset(write_export(path, [row]), seq_len=SEQ_LEN)))])


def test_choice_logits_match_full_vocabulary_logits(tmp_path: Path):
    row = make_bins(1, seed=3)[0]
    batch = load_batch(tmp_path / "train", row)
    hidden = torch.randn(1, SEQ_LEN, 8)
    weight = torch.randn(VOCAB, 8)

    full_logits = hidden[0] @ weight.T
    expected = torch.zeros(len(row["positions"]), MAX_CHOICES)
    for index, (position, ids) in enumerate(zip(row["positions"], row["choice_ids"])):
        expected[index, : len(ids)] = full_logits[position, ids]
    torch.testing.assert_close(choice_logits(hidden, weight, batch["choice_ids"]), expected)


def test_cp_shards_keep_rows_with_their_hidden_states(tmp_path: Path):
    # Answer positions 2, 9 and 14: position 9 is the last of shard 0 and its answer token opens shard 1
    row = make_bin(
        [
            ([1, 2, 3, 4, 5, 6], 3, [4, 7], [0.25, 0.75], 1.5, [1.0, 0.0, 0.0]),
            ([8, 9, 10, 11, 12, 13], 4, [11, 12, 13], [0.5, 0.25, 0.25], 0.5, [0.0, 1.0, 0.0]),
            ([14, 15, 16, 17, 18, 19], 3, [17], [1.0], 2.0, [0.5, 0.5, 0.0]),
        ]
    )
    assert row["positions"] == [2, 9, 14]
    batch = load_batch(tmp_path / "train", row)
    hidden = torch.randn(1, SEQ_LEN, 8)
    weight = torch.randn(VOCAB, 8)
    choice_tensors = ("choice_ids", "choice_targets", "choice_weights", "choice_mix_weights")
    full_logits = choice_logits(hidden, weight, batch["choice_ids"])
    full_inputs = choice_loss_inputs(full_logits, *(batch[key] for key in choice_tensors))

    shard_len = SEQ_LEN // 2
    shard_inputs = []
    for cp_rank in range(2):
        choice_ids, choice_targets, choice_weights, choice_mix_weights, hidden_shard = (
            shard_for_cp(tensor, cp_rank=cp_rank, cp_world_size=2)
            for tensor in (*(batch[key] for key in choice_tensors), hidden)
        )
        local_positions = torch.nonzero(supervised_rows(choice_ids)).flatten() + cp_rank * shard_len
        assert local_positions.tolist() == [p for p in row["positions"] if p // shard_len == cp_rank]
        assert batch["input_ids"][0, local_positions + 1].tolist() == [
            observed for p, observed in zip(row["positions"], row["observed_ids"]) if p // shard_len == cp_rank
        ]
        logits = choice_logits(hidden_shard, weight, choice_ids)
        shard_inputs.append(choice_loss_inputs(logits, choice_ids, choice_targets, choice_weights, choice_mix_weights))

    for key, value in full_inputs.items():
        torch.testing.assert_close(torch.cat([inputs[key] for inputs in shard_inputs]), value, msg=key)
    assert full_inputs["choice_counts"].tolist() == [2, 3, 1]
    torch.testing.assert_close(full_inputs["weights"], torch.tensor([1.5, 0.5, 2.0]))
    torch.testing.assert_close(full_inputs["target_probs"][1], torch.tensor([0.5, 0.25, 0.25, 0.0]))
    torch.testing.assert_close(
        full_inputs["mix_weights"], torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    )


SFT_MODEL = {"model": {"name": "PrimeIntellect/Qwen3-0.6B"}}
PACKED_DATA = {"type": "packed_choice", "name": "/unused", "seq_len": SEQ_LEN}
CHOICE_LOSS = {"import_path": "my_module.my_loss"}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"data": PACKED_DATA}, id="packed-without-loss"),
        pytest.param({"loss": CHOICE_LOSS}, id="loss-with-sft-data"),
        pytest.param({"data": PACKED_DATA, "loss": CHOICE_LOSS, "val": {"data": {"name": "x"}}}, id="sft-val"),
        pytest.param({"val": {"data": PACKED_DATA}}, id="packed-val-with-sft-data"),
        pytest.param({"data": {**PACKED_DATA, "micro_batch_size": 2, "batch_size": 2}, "loss": CHOICE_LOSS}, id="mbs"),
    ],
)
def test_packed_choice_config_rejects(overrides: dict):
    with pytest.raises(ValidationError):
        SFTConfig.model_validate({**SFT_MODEL, **overrides})


def test_packed_choice_config_accepts():
    config = SFTConfig.model_validate(
        {
            **SFT_MODEL,
            "data": PACKED_DATA,
            "loss": CHOICE_LOSS,
            "val": {"data": PACKED_DATA},
            "renderer": {"name": "default"},
            "scheduler": {"type": "cosine", "warmup_steps": 10},
        }
    )
    assert isinstance(config.data, PackedChoiceDataConfig) and isinstance(config.val.data, PackedChoiceDataConfig)
    assert config.max_steps is None

    untagged_val = SFTConfig.model_validate({**SFT_MODEL, "val": {"data": {"name": "x"}}})
    assert isinstance(untagged_val.val.data, SFTDataConfig)
